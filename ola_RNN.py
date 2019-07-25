import json
import random
import math
import torch 
import re
import numpy as np

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


def cuda(input):
    if torch.cuda.is_available(): return input.cuda()
    return input

def encodestr(string, encoder):
    x = torch.zeros((len(string),len(encoder)))
    x[[idx for idx in range(0,len(string))],[encoder[char] for char in string]] = 1
    return x

def change_char(s, p, r):
    return s[:p]+r+s[p+1:] 

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

class Struct():
    pass 

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


def rnn_forward(learn,hidden,xb,yb):
    learn.model.train()
    if xb[0,0,1].item() == 1: hidden = learn.model.initHidden(xb.shape[0])                   
    loss = 0 
    for char in range(xb.shape[1]):
        x,y           = xb[:,char],yb[:,char]
        x,y,hidden    = unpad(x,y,hidden)
        output,hidden = learn.model.forward(x,hidden)
        loss += learn.loss_fn(output,y)                
    return output,hidden.detach(),loss/(char+1)

def unpad(x,y,hidden):
    idx = (y != 0).nonzero()        
    if idx.shape[0] == 1: idx = idx[0]
    else: idx = idx.squeeze()
    return x[idx],y[idx],hidden[idx]

def get_valid_rnn(learn,itters=30):
    print(f"""getting validation""")    
    learn.model.eval()
    tot_loss = 0 
    with torch.no_grad():
        hidden = learn.model.initHidden(learn.data.valid_dl.bs)
        for xb,yb in iter(learn.data.valid_dl): 
            output, hidden, loss = learn.model.batch_forward(xb,yb,hidden,learn.loss_fn)
            if loss != 0: tot_loss += loss.item()/xb.shape[0]
            if learn.data.valid_dl.get_itter() == itters: 
                return tot_loss/learn.data.valid_dl.get_itter()
        
    return tot_loss/learn.data.valid_dl.nb_itters()

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
    
class Learner():
    def __init__(self, model, loss_fn, opt, data, lr):
        self.model, self.opt, self.loss_fn, self.data = model, opt, loss_fn, data
        self._lr     = opt.param_groups[0]['lr']
        self.hidden  = None    
        self.stats   = Struct()
        self.stats.valid_loss = []
        self.stats.train_loss = []          
        self.n_epochs = 0.
        self.n_iters  = 0 
        
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
            
def one_rnn_batch(xb,yb,cb):
    pred, cb.learn.hidden, loss = cb.learn.model.batch_forward(xb,yb,cb.learn.hidden,learn.loss_fn)
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