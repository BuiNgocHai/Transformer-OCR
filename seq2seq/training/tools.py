import os
import json 
import difflib
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Saver(object):
    """ Checkpoint class """
    def save_checkpoint(self, state, filepath='model.pt'):
        """ Save model checkpoints
        Args:
            state (model dict): model's state for saving
            filename (str): Checkpoint file path 
        Returns:
            filepath (str): model path 
        """
        torch.save(state, filepath)
        return filepath

    def load_checkpoint(self, filepath):
        """ Load model checkpoints
        Args:
            filepath (str): checkpoint file path
        Returns:
        """
        if DEVICE == torch.device('cpu'):
            checkpoint = torch.load(filepath, \
                map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(filepath)
        return checkpoint

class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""
    def __init__(self):
        self.reset()

    def add(self, v):
        self.n_count += 1
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def average(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        self.character = character
        if not '[blank]' in self.character:
            self.character = ['[blank]'] + self.character  # dummy '[blank]' token for CTCLoss (index 0)

        self.dict = {}
        for i, char in enumerate(self.character[1:]):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

    def encode(self, text, batch_max_length):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text).to(DEVICE), torch.IntTensor(length).to(DEVICE))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """
    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        self.character = character
        if (list_token[0] not in self.character \
            and list_token[1] not in self.character):
            self.character = list_token + self.character

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def encode(self, text, batch_max_length):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(DEVICE), torch.IntTensor(length).to(DEVICE))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

class AggregationCrossEntropyLoss(nn.Module):
    """
        "Implement Aggregation Cross-Entropy Loss"
        https://arxiv.org/abs/1904.08364
        Parameters:
            input_tensor: probs output tensor (a tensor with shape (batch_size, time_steps, class_size))
            anotation_number:a tensor contains number of K class frequences  (a tensor with shape (batch_size, class_size))
        Return:
            A tensor that is average loss over the batch 
    """
    def __init__(self, character):
        super(AggregationCrossEntropyLoss, self).__init__()

        self.character = character
        if '[blank]' not in self.character:
            self.character = ['[blank]'] + self.character

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def encode(self, text, *args):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
        """

        length = [len(s) for s in text]
        batch_text = torch.LongTensor(len(text), len(self.character)).fill_(0)

        for i, t in enumerate(text):
            for c in list(t):
                batch_text[i][self.dict[c]] += 1
            batch_text[i][0] = len(t)

        batch_text = batch_text.type(torch.float32)
        return (batch_text.to(DEVICE), torch.IntTensor(length).to(DEVICE))
    
    def decode(self, preds, *args):
        preds = F.softmax(preds, dim=-1) + 1e-10
        out_best = torch.max(preds, 2)[1].data.cpu().numpy()
        batch_size = preds.size(0)
        texts = [0] * batch_size

        for j in range(batch_size):
            texts[j] = ''.join([self.character[i] \
                for i in out_best[j][out_best[j]!=0]])
        return texts

    def decode_label(self, labels):
        batch_text = list()
        for i, t in enumerate(labels):
            index = [idx for idx, i in enumerate(t) if i != 0]
            text = ''.join([self.character[i] for i in index])
            batch_text.append(text)
        return batch_text

    def forward(self, probs, labels):
        """
        
        """
        # # assert len(labels) == 2  # labels must be 2 dimensional
        # device = torch.device("cuda" if self.cuda else "cpu")
        probs = probs.to(DEVICE)
        labels = labels.to(DEVICE)
        probs = F.softmax(probs, dim=-1)

        self.batch_size, self.time_steps, self.num_class = probs.size()
        probs = probs + 1e-10
        labels[:,0] = self.time_steps - labels[:,0]
        
        # Normalize Yk and Nk 
        norm_y_k = torch.sum(probs, dim=1).type(torch.FloatTensor).to(DEVICE) / self.time_steps
        norm_n_k = labels.type(torch.FloatTensor).to(DEVICE) / self.time_steps

        # Calculate loss based on batch 
        loss = (-torch.sum(torch.log(norm_y_k) * norm_n_k)) / self.batch_size

        return loss