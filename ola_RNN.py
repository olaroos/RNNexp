import json
import random
import math 
import re
import numpy as np
import torch, torchvision
import torch.nn as nn
from functools import partial

from ola_nondepfun import * 
    
def cuda(input):
    if torch.cuda.is_available(): return input.cuda()
    return input

class TweetDataLoader():
    def __init__(self,data,tweets,bs,sql,shuffle=False):    
        self.tweets  = tweets
        self.bs      = bs         
        self.sql     = sql
        self.encoder = data.encoder
        self.decoder = data.decoder
        self.i       = -1
        self.shuffle = shuffle        
        
    def reset(self):
        if self.shuffle: random.shuffle(self.tweets)
        self.i  = -1
        
    def get_itter(self):
        return self.i
    
    def __iter__(self):  
        self.reset()
        while True:
            self.i += 1
            twt      = self.tweets[self.i*self.bs:(self.i+1)*self.bs]
            sbx,sby  = mk_tweetbatch(twt,self.encoder,self.bs,self.sql)
            sbloader = iter(SBDataLoader(sbx,sby))            
            try:
                while True:                
                    yield next(sbloader) 
            except StopIteration:
                pass            
            if self.i==round(len(self.tweets)/self.bs)-2: 
                break

class SBDataLoader():
    def __init__(self, sbx, sby): 
        self.sbx, self.sby = sbx, sby
    def __iter__(self):
        for j in range(self.sbx.shape[1]): yield cuda(self.sbx[:,j]), cuda(self.sby[:,j])

def mk_tweetbatch(tweets,encoder,bs,sql,symbol='£'):
    assert(math.floor(len(tweets)/bs)==len(tweets)/bs)
    bch       = batch_strings(tweets,bs,sql)[0]
    assert(math.floor(len(bch[0])/sql)==len(bch[0])/sql)            
    n_segment = int(len(bch[0])/sql)
    sbx       = torch.zeros(bs,n_segment,sql,len(encoder))
    sby       = torch.zeros(bs,n_segment,sql).long()
    for tweet in range(bs):
        """for target we don't use first char, compensate with one padded char"""
        y_str = bch[tweet][1:len(bch[tweet])]+symbol              
        chng_pos = len(bch[tweet])
        """if we find padded char, we know that tweet ended, remove last char of tweet"""        
        if re.search(symbol,bch[tweet]): chng_pos = re.search(symbol,bch[tweet]).span()[0]       
        x_str = change_char(bch[tweet],chng_pos-1,symbol)     
        for segment in range(n_segment):
            x = x_str[sql*segment:sql*(segment+1)]
            y = y_str[sql*segment:sql*(segment+1)] 
            sbx[tweet,segment] = encodestr(x,encoder)
            sby[tweet,segment] = torch.Tensor([encoder[char] for char in y])     
    """remove last batch if it contains only pad-elements"""            
    idx = sby[:,-1,0].nonzero()
    if idx.nelement() == 0: 
        sbx,sby = sbx[:,0:-2], sby[:,0:-2]
    idx = sby[:,-1,0].nonzero()
    if idx.nelement() == 0:
        print(y_str)
    return sbx,sby

def load_trumpdata(datapath, pad_tok='£', start_tok='^', end_tok='€'):

    van_tws, tws, van_tw_str, tw_str = [],[],'',''
    filenames = ['condensed_2018.json', 'condensed_2016.json', 'condensed_2017.json', 'condensed_2015.json']
    for fname in filenames:
        f = open(datapath+fname,"r")
        data = f.readline()
        f.close()
        data_tr = json.loads(data)
        for line in range(0,len(data_tr)):
            tweet      = data_tr[line]["text"].rstrip('\\')
            van_tw_str = van_tw_str + tweet 
            van_tws.append(tweet)            
    symbols = list(set(van_tw_str))  
    assert(pad_tok   not in symbols)
    assert(start_tok not in symbols)
    assert(end_tok   not in symbols)

    for tweet in van_tws:
        pad_tweet = start_tok + tweet + end_tok
        tw_str    = tw_str + pad_tweet            
        tws.append(pad_tweet)        
    symbols = [pad_tok, start_tok] + symbols + [end_tok]   
    decoder = {idx: symbols[idx] for idx in range(0,len(symbols))}
    encoder = {symbols[idx]: idx for idx in range(0,len(symbols))}        
    return tws, tw_str, decoder, encoder

def pp_trumpdata(filename, prop, bs=1):
    Data, train, valid, test = Struct(), Struct(), Struct(), Struct()        
    tweets, tweet_str, Data.decoder, Data.encoder = load_trumpdata(filename)    
    train.tweets = tweets[0:round(prop[0]*len(tweets))]
    train.tweet_str = tweet_str[0:round(prop[1]*len(tweet_str))]    
    valid.tweets = tweets[round(prop[0]*len(tweets)):round(prop[1]*len(tweets))]
    valid.tweet_str = tweet_str[round(prop[0]*len(tweet_str)):round(prop[1]*len(tweet_str))]    
    test.tweets  = tweets[round(prop[1]*len(tweets)):-1]
    test.tweet_str  = tweet_str[round(prop[1]*len(tweet_str)):-1]    

    train.batch_str = []
    stepsize = round(len(train.tweet_str)/bs-1)
    for i in range(0,bs):
        train.batch_str.append(train.tweet_str[i*stepsize:(i+1)*stepsize])
    valid.batch_str = [valid.tweet_str]
    
    Data.train, Data.valid, Data.test, Data.bs = train, valid, test, bs
    return Data

def get_valid_rnn(learn,itters=30):
    print(f"""getting validation""")    
    learn.model.eval()
    tot_loss = 0 
    tot_accu = 0 
    with torch.no_grad():
        hidden = learn.model.initHidden(learn.data.valid_dl.bs)
        for xb,yb in iter(learn.data.valid_dl): 
            output, hidden, loss, accu = learn.model.batch_forward(xb,yb,hidden,learn.loss_fn)
            if loss != 0: 
                tot_loss += loss.item()
                tot_accu += accu
            if learn.data.valid_dl.get_itter() == itters:
                return tot_loss/learn.data.valid_dl.get_itter(), tot_accu/learn.data.valid_dl.get_itter()
    return tot_loss/learn.data.valid_dl.get_itter(), tot_accu/learn.data.valid_dl.get_itter()

def generate_seq(model,Data,sql,symbol='^'):
    model.eval()
    with torch.no_grad():
        hidden = model.initHidden(1)
        result = symbol
        for i in range(sql):
            x = cuda(onehencode(symbol,Data.encoder))
            output, hidden = model.forward(x,hidden)        
            hidden = hidden.detach()
            
            prob     = np.exp(output[0].cpu().numpy())
            cum_prob = np.cumsum(prob)
            idx      = np.where(cum_prob - random.random() > 0)[0][0]
            symbol   = Data.decoder[idx]
            result  += symbol
    model.train()
    print(result)


def one_rnn_batch(xb,yb,cb):
    pred, cb.learn.hidden, loss, accu = cb.learn.model.batch_forward(xb,yb,cb.learn.hidden,cb.learn.loss_fn)
    if not cb.after_loss(loss): return    
    loss.backward()
    if not cb.after_backward(): return 
    cb.learn.opt.step()
    if not cb.after_step(): return
    cb.learn.opt.zero_grad()

def fit_rnn(epoches, learn, cb=None, itters=math.inf):
    hidden = learn.model.initHidden(learn.data.train_dl.bs)
    if not cb.begin_fit(learn):           return 
    for epoch in range(epoches):
        if not cb.begin_epoch(epoch):     return             
        for xb, yb in iter(learn.data.train_dl):   
            if not cb.begin_batch(xb,yb): return   
            one_rnn_batch(xb,yb,cb)
            if not cb.begin_validate():   return     
            if cb.do_stop():              break 
        if not cb.after_epoch():          return
    if not cb.after_fit():                return 
    return         
        
f""" Callback functions""" 

class ParamScheduler(Callback):
    _order=5
    def __init__(self, pname, sched_func): self.pname,self.sched_func = pname,sched_func

    def begin_fit(self,learn):
        super().begin_fit(learn)
        return True        
        
    def set_param(self):
        for pg in self.learn.opt.param_groups:
            pg[self.pname] = self.sched_func(self.learn.n_epochs)
        return True
    
    def begin_batch(self,xb,yb): 
        if self.learn.in_train: self.set_param()
        return True

class CounterCallback(Callback):    
    _order = 1
    
    def __init__(self, iters=None):        
        if iters is None:
            print(f"""FYI number of iterations in a Twitter epoch is random, % progress will not be correctly displayed.""")
        self.iters = iters

    def begin_fit(self,learn):
        super().begin_fit(learn)
        self.learn.n_epochs=0.
        if self.learn.n_iters is None or self.learn.n_iters != 0: 
            self.learn.n_iters=0                
        return True
    
    def after_step(self):
        if not self.learn.in_train: return
        if self.iters is not None:
            self.learn.n_epochs += 1./self.iters
        self.learn.n_iters  += 1
        if self.learn.n_iters % self.iters == 0:
            self.learn.stop = True 
        return True
    
    def after_epoch(self):
        if self.iters is None:
            self.learn.n_epochs += 1
        return True
    
class StatsCallback(Callback):
    _order = 10
    
    def __init__(self,lossbeta=None):
        if lossbeta is None: self.lossbeta = 0.99  
        else: self.lossbeta = lossbeta
        self.mva_loss = 0 
        
    def begin_fit(self,learn):
        super().begin_fit(learn)
        self.learn.stats.lrs = []
        return True

    def after_loss(self,loss):
        newloss      = loss.detach().cpu()
        self.learn.stats.train_loss.append(newloss)     
        
        self.mva_loss = self.mva_loss*self.lossbeta + (1-self.lossbeta)*newloss        
        self.learn.stats.train_mva_loss.append(self.mva_loss/(1-self.lossbeta**(self.learn.n_iters+1)))                     
        return True
    
    def after_step(self):
        self.learn.stats.lrs.append(self.learn.opt.param_groups[-1]['lr'])
        return True
        
    def begin_validate(self):
        if self.learn.n_iters%50 == 0:
            self.learn.in_train = False    
            loss, accu = get_valid_rnn(self.learn,itters=15)
            self.learn.stats.valid_loss.append(loss)
            self.learn.stats.valid_accu.append(accu)
            print(f"""finished: {self.learn.n_epochs}%""")
        return True
        
