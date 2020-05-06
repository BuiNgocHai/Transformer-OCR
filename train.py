import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import argparse
from dataset import ListDataset
from dataset import Batch
from dataset import Load_data
from model import make_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm

def run_epoch(dataloader, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, (imgs, labels_y, labels) in enumerate(dataloader):
        batch = Batch(imgs, labels_y, labels)
        out = model(batch.imgs, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens



def train(opt):
    load_data = Load_data(opt)
    char2token = load_data.char2token
    train_dataloader = load_data.train_loader
    val_dataloader = load_data.val_loader
    model = make_model(len(char2token))
    if opt.pretrained_path != None:
        model.load_state_dict(torch.load(opt.pretrained_path))
    model.cuda()
    criterion = LabelSmoothing(size=len(char2token), padding_idx=0, smoothing=0.1)
    criterion.cuda()
    model_opt = NoamOpt(model.tgt_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10000):
        model.train()
        run_epoch(train_dataloader, model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        test_loss = run_epoch(val_dataloader, model, 
              SimpleLossCompute(model.generator, criterion, None))
        print("test_loss", test_loss)
        torch.save(model.state_dict(), 'checkpoint/%08d_%f.pth'%(epoch, test_loss))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    """ Data params """
    parser.add_argument('--data_path', default='./data/', help='path to dataset')
    parser.add_argument('--train_file', default='train.json', help='name of train label file')
    parser.add_argument('--val_file', default='val.json', help='name of train label val')
    parser.add_argument('--test_file', default='test.json', help='name of train label test')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained models')
    """ Training params """
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    """ Data processing """
    parser.add_argument('--img_height', type=int, default=64, help='the height of the input image')
    parser.add_argument('--img_width', type=int, default=1500, help='the width of the input image, before=800')
    parser.add_argument('--normalize_text', action='store_false', help='normalizing text label')
    parser.add_argument('--padding', action='store_false', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--augment', action='store_false', help='whether to augment data or not')

    """ Model Architecture """
    parser.add_argument('--prediction', type=str, default='ctc', help='Prediction stage. ctc|attn|ace')
    opt = parser.parse_known_args()[0]

    train(opt)





