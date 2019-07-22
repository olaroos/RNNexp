import torch
import math
from typing import *
from functools import partial 
import matplotlib.pyplot as plt

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
    
class CounterCallback(Callback):    
    _order = 1
    
    def __init__(self, iters=None):        
        self.iters = iters

    def begin_fit(self,learn):
        super().begin_fit(learn)
        self.learn.n_epochs=0.
        self.learn.n_iter=0
        return True
    
    def after_step(self):
        if not self.learn.in_train: return
        if self.iters is not None:
            self.learn.n_epochs += 1./self.iters
        self.learn.n_iters  += 1
        if self.learn.n_iters == self.iters:
            self.learn.stop = True 
        return True
    
    def after_epoch(self):
        if self.iters is None:
            self.learn.n_epochs += 1
        return True
    
        
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