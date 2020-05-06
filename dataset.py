import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable
import os
import argparse
import cv2
import numpy as np
from torchvision import datasets, models, transforms
import json
from data_loader.dataset import  DataLoader




def illegal(label):
    if len(label) > label_len-1:
        return True
    for l in label:
        if l not in vocab[1:-1]:
            return True
    return False

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f) 
    return data

class ListDataset(Dataset):
    def __init__(self, opt):
        data = load_json('train_all.json')
        self.img_path = []
        self.labels = []
        for key in data:
            self.img_path.append(key)
            self.labels.append(data[key])
        self.characters = None
        self.dataset_loader = DataLoader(opt, self.characters)
        print(len(self.dataset_loader.vocabulary))

        self.vocab = self.dataset_loader.vocabulary
        self.label_len = 148

        self.char2token = {"PAD":0}
        self.token2char = {0:"PAD"}
        for i, c in enumerate(self.vocab):
            self.char2token[c] = i+1
            self.token2char[i+1] = c

    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self, index):
        '''
        line: image path\tlabel
        '''
        #line = self.lines[index]
        #img_path, label_y_str = line.strip('\n').split('\t')
        img_path = self.img_path[index]
        label_y_str = self.labels[index]
        img = cv2.imread(img_path) / 255.
        # Channels-first
        img = np.transpose(img, (2, 0, 1))
        # As pytorch tensor
        img = torch.from_numpy(img).float()
        label = np.zeros(self.label_len, dtype=int)
        for i, c in enumerate(label_y_str):
            label[i] = self.char2token[c]
        label = torch.from_numpy(label)

        label_y = np.zeros(self.label_len, dtype=int)
        for i, c in enumerate(label_y_str):
            label_y[i] = self.char2token[c]
        label_y = torch.from_numpy(label_y) 
        return img, label_y, label

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, imgs, trg_y, trg, pad=0):
        self.imgs = Variable(imgs.cuda(), requires_grad=False)
        self.src_mask = Variable(torch.from_numpy(np.ones([imgs.size(0), 1, 36], dtype=np.bool)).cuda())
        if trg is not None:
            self.trg = Variable(trg.cuda(), requires_grad=False)
            self.trg_y = Variable(trg_y.cuda(), requires_grad=False)
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return Variable(tgt_mask.cuda(), requires_grad=False)

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, name):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.name = name
    def forward(self, x):
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name is self.name:
                b = x.size(0)
                c = x.size(1)
                return x.view(b, c, -1).permute(0, 2, 1)
        return None

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    """ Data params """
    parser.add_argument('--data_path', default='./data/', help='path to dataset')


    parser.add_argument('--train_file', default='train.json', help='name of train label file')
    parser.add_argument('--val_file', default='val.json', help='name of train label val')
    parser.add_argument('--test_file', default='test.json', help='name of train label test')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)

    """ Training params """
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    """ Data processing """
    parser.add_argument('--img_height', type=int, default=64, help='the height of the input image')
    parser.add_argument('--img_width', type=int, default=1500, help='the width of the input image, before=800')
    parser.add_argument('--normalize_text', action='store_false', help='normalizing text label')
    parser.add_argument('--padding', action='store_false', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--augment', action='store_false', help='whether to augment data or not')

    """ Model Architecture """
    parser.add_argument('--prediction', type=str, default='ctc', help='Prediction stage. ctc|attn|ace')
    opt = parser.parse_known_args()[0]

    listdataset = ListDataset(opt)
    dataloader = torch.utils.data.DataLoader(listdataset, batch_size=1, shuffle=False, num_workers=0)
    for epoch in range(1):
        for batch_i, (imgs, labels_y, labels) in enumerate(dataloader):
            print('pass')
            continue