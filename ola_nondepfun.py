import json
import random
import math 
import re
import numpy as np
import torch, torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from functools import partial

class Struct():
    pass 

def get_accu(output,y):
    _, idxs   = output.max(1)
    n_correct = ((idxs-y)==0).sum().item()
    return n_correct/output.shape[1]    

def plot_list(plist, label='label'):
    plt.figure()
    plt.plot([x for x in range(len(plist))],plist,label=label)
    plt.legend()      
    plist[-1]

def onehencode(symbol, encoder):
    x = torch.zeros(len(encoder),1)
    x[encoder[symbol]] = 1.0
    return x.t()    
    
def yencode(string, encoder):
    return torch.Tensor([encoder[char] for char in y_str])

def onehdecode(X,decoder):
    assert(X.shape[-1] == len(decoder))
    string = ''
    for char in range(X.shape[0]):
        val, idx = torch.max(X[char],0)
        string += decoder[idx.item()]
    print(string)
    
def ydecode(Y,decoder):
    string = ''
    for char in range(Y.shape[0]): string += decoder[Y[char].item()]
    print(string)

def encodestr(string, encoder):
    x = torch.zeros((len(string),len(encoder)))
    x[[idx for idx in range(0,len(string))],[encoder[char] for char in string]] = 1
    return x

def change_char(s, p, r):
    return s[:p]+r+s[p+1:] 

def batch_strings(tweets,bs,sql=1):
    f"""creates a list of batchsize-list of strings of same length and sort each batch with longest string first."""    
    offset = -1*((len(tweets)/bs)*10%2!=0)    
    bch_strs = [] 
    for i in range(round(len(tweets)/bs)+offset):
        strings = tweets[i*bs:(i+1)*bs]
        strings.sort(key=len,reverse=True)
        pad_strings = pad(strings,sql)
        bch_strs.append(pad_strings)
    return bch_strs

def pad(str_list,sql=1,token='£'):
    f"""pad all strings in a list to max_len"""
    max_len = math.ceil((len(max(str_list, key=len)))/sql)*sql
    for idx, row in enumerate(str_list):        
        str_list[idx] = row + token*(max_len-len(row))
    if len(str_list) == 1: return str_list[0]    
    return str_list

def unpad(x,y,hidden):
    idx = (y != 0).nonzero()    
    if idx.shape[0] == 1: idx = idx[0]
    else: idx = idx.squeeze()
    if len(hidden.shape) > 2: return x[idx],y[idx],hidden[:,idx]
    return x[idx],y[idx],hidden[idx]

class Learner():
    def __init__(self, model, loss_fn, opt, data, lr):
        self.model, self.opt, self.loss_fn, self.data = model, opt, loss_fn, data
        self._lr     = opt.param_groups[0]['lr']
        self.hidden  = None    
        self.stats   = Struct()
        self.stats.valid_loss = []
        self.stats.valid_accu = [] 
        self.stats.train_loss = []
        self.stats.train_accu = []
        self.stats.train_mva_loss = []        
        self.n_epochs = 0.
        self.n_iters  = 0 
        self.stop    = False        
        
    @property
    def lr(self):
        return self._lr
    
    @lr.setter
    def lr(self,lr):
        self._lr = lr
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr        
            
    def one_batch(self, i, xb, yb):
        try:
            self.iter = i 
            self.xb,self.yb = xb,yb;                       self('begin_batch')
            self.pred = self.model(self.xb);               self('after_pred')
            self.loss = self.loss_fn(self.pred, self.yb);  self('after_loss')
            if not self.in_train: return
            self.loss.backwards();                         self('after_backward')
            self.opt.step();                               self('after_step')
            self.opt.zero_grad();
        except CancelBatchException:                       self('after_cancel_ batch')
        finally:                                           self('after_batch')


class Callback():
   
    def begin_fit(     self,learn): self.learn = learn;      return True
    def begin_epoch(   self,epoch): self.epoch = epoch;      return True
    def begin_batch(   self,xb,yb): self.xb,self.yb = xb,yb; return True
    def after_loss(    self,loss):  self.loss=loss;          return True
    def after_backward(self):                                return True
    def after_step(    self):                                return True
    def begin_validate(self):                                return True
    def after_epoch(   self):                                return True     
    def after_fit(     self):                                return True    

class CallbackHandler():
    def __init__(self,cbs=None):
        self.cbs = cbs if cbs else []
        self.cbs.sort(key=lambda x: x._order)

    def begin_fit(self, learn):
        self.learn = learn
        self.learn.in_train = True
        self.learn.stop = False
        res = True
        for cb in self.cbs: res = res and cb.begin_fit(learn)
        return res

    def after_fit(self):
        res = not self.learn.in_train
        for cb in self.cbs: res = res and cb.after_fit()
        return res
    
    def begin_epoch(self, epoch):
        self.learn.model.train()
        self.learn.in_train=True
        res = True
        for cb in self.cbs: res = res and cb.begin_epoch(epoch)
        return res

    def begin_validate(self):
        self.learn.model.eval()
        self.learn.in_train=False
        res = True
        for cb in self.cbs: res = res and cb.begin_validate()
        return res

    def after_epoch(self):
        res = True
        for cb in self.cbs: res = res and cb.after_epoch()
        return res
    
    def begin_batch(self, xb, yb):
        self.learn.in_train=True
        res = True
        for cb in self.cbs: res = res and cb.begin_batch(xb, yb)
        return res

    def after_loss(self, loss):
        res = self.learn.in_train
        for cb in self.cbs: res = res and cb.after_loss(loss)
        return res

    def after_backward(self):
        res = True
        for cb in self.cbs: res = res and cb.after_backward()
        return res

    def after_step(self):
        res = True
        for cb in self.cbs: res = res and cb.after_step()
        return res
    
    def do_stop(self):
        try:     return self.learn.stop
        finally: self.learn.stop = False    

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

torch.Tensor.ndim = property(lambda x: len(x.shape))

@annealer
def sched_lin(start, end, pos): return start + pos*(end-start)
@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
@annealer
def sched_no(start, end, pos):  return start
@annealer
def sched_exp(start, end, pos): return start * (end/start) ** pos

def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = torch.tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner

def print_sched(sched,t='label'):
    a = torch.arange(0, 100)
    p = torch.linspace(0.01,1,100)
    plt.plot(a, [sched(o) for o in p], label=t)