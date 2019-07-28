import json 
def pad(str_list,sql=1,token='£'):
    f"""pad all strings in a list to max_len"""
    max_len = math.ceil(len(max(str_list, key=len))/sql)*sql
    for idx, row in enumerate(str_list):        
        str_list[idx] = row + token*(max_len-len(row))
    if len(str_list) == 1: return str_list[0]
    return str_list

class ParentDataLoader():
    def __init__(self, ds): 
        self.ds = ds
    def __iter__(self):    
        for i in range(len(self.ds)):
            iterator = iter(self.ds[i])
            yield next(iterator), True
            try:
                while True:                
                    yield next(iterator), False 
            except StopIteration:
                pass

class SBDataLoader():
    def __init__(self, sbx, sby): 
        self.sbx, self.sby = sbx, sby
    def __iter__(self):
        for j in range(self.sbx.shape[1]): yield cuda(self.sbx[:,j]), cuda(self.sby[:,j])



def load_trumpdata(datapath, pad_tok='£', start_tok='^', end_tok='€'):

    van_tws, tws, van_tw_str, tw_str = [],[],'',''
    filenames = ['condensed_2018.json', 'condensed_2016.json', 'condensed_2017.json', 'condensed_2018.json']
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
    symbols = [pad_tok] + symbols + [start_tok, end_tok]   
    decoder = {idx: symbols[idx] for idx in range(0,len(symbols))}
    encoder = {symbols[idx]: idx for idx in range(0,len(symbols))}        
    return tws, tw_str, decoder, encoder
class Struct():
    pass 

def pp_trumpdata(filename, prop, bsize=1):
    Data, train, valid, test = Struct(), Struct(), Struct(), Struct()        
    tweets, tweet_str, Data.decoder, Data.encoder = load_trumpdata(filename)    

    train.tweets = tweets[0:round(prop[0]*len(tweets))]
    train.tweet_str = tweet_str[0:round(prop[1]*len(tweet_str))]    
    valid.tweets = tweets[round(prop[0]*len(tweets)):round(prop[1]*len(tweets))]
    valid.tweet_str = tweet_str[round(prop[0]*len(tweet_str)):round(prop[1]*len(tweet_str))]    
    test.tweets  = tweets[round(prop[1]*len(tweets)):-1]
    test.tweet_str  = tweet_str[round(prop[1]*len(tweet_str)):-1]    

    train.batch_str = []
    stepsize = round(len(train.tweet_str)/bsize-1)
    for i in range(0,bsize):
        train.batch_str.append(train.tweet_str[i*stepsize:(i+1)*stepsize])
    valid.batch_str = [valid.tweet_str]
    
    Data.train, Data.valid, Data.test, Data.bsize = train, valid, test, bsize
    return Data
bs = 15
path = '/home/r2/Documents/RNNexp'
Data       = pp_trumpdata(path+"/data/trump/", [0.9,0.95], bs)
tweets = Data.train.tweets

def make_parentbatch(tweets, bs, sql, symbol='£'):
    f"""each parent-batch will have different numbers of sub-batches depending on how long the tweets are"""
    assert(len(tweets)/bs*10%2==0)
    bch_strs = batch_strings(tweets,bs,sql)
    parent_batches = []
    for pb in range(len(bch_strs)):
        bch       = bch_strs[pb]
        n_tweet   = bs
        n_segment = math.ceil(len(bch[0])/sql)
        sbx = torch.zeros(n_tweet,n_segment,sql,len(Data.decoder))
        sby = torch.zeros(n_tweet,n_segment,sql).long()

        for tweet in range(n_tweet):
            if re.search(symbol,bch[tweet]): position = re.search(symbol,bch[tweet]).span()[0]
            else:                            position = len(bch[tweet])
            x_str = change_char(bch[tweet],position-1,symbol)
            y_str = bch[tweet][1:len(bch[tweet])]+symbol                
            for segment in range(n_segment):
                x = x_str[sql*segment:sql*(segment+1)]
                y = y_str[sql*segment:sql*(segment+1)]  
                sbx[tweet,segment] = encodestr(x,Data.encoder)
                sby[tweet,segment] = torch.Tensor([Data.encoder[char] for char in y])                
                
        sb_ds = SBDataLoader(sbx, sby)
        parent_batches.append(sb_ds)
    return parent_batches

def batch_strings(tweets,bs,sql=1):
    f"""creates a list of batchsize-list of strings of same length and sort each batch with longest string first."""
    offset = -1*(int(len(tweets)/bs * 10) % 2 != 0)
    bch_strs = [] 
    for i in range(round(len(tweets)/bs)+offset):
        strings = tweets[i*bs:(i+1)*bs]
        strings.sort(key=len,reverse=True)
        pad_strings = pad(strings,sql)
        bch_strs.append(pad_strings)
    return bch_strs
import math
import torch
import re

def encodestr(string, encoder):
    x = torch.zeros((len(string),len(encoder)))
    x[[idx for idx in range(0,len(string))],[encoder[char] for char in string]] = 1
    return x

def onehencode(symbol, encoder):
    x = torch.zeros(len(encoder),1)
    x[encoder[symbol]] = 1.0
    return x.t()

def encode(string, encoder):
    return torch.Tensor([encoder[char] for char in y_str])

def onehdecode(X,decoder):
    string = ''
    for char in range(X.shape[0]):
        val, idx = torch.max(X[char],0)
        string += decoder[idx.item()]
    print(string)
    
def ydecode(Y,decoder):
    string = ''
    for char in range(Y.shape[0]): string += decoder[Y[char].item()]
    print(string)



def change_char(s, p, r):
    return s[:p]+r+s[p+1:] 
# %lprun -f make_parentbatch make_parentbatch(tweets,bs,sql=Params.sql)
Params = Struct()
Params.ni      = 3000
Params.ne      = 1
Params.hd_sz   = 150
Params.in_sz   = len(Data.encoder)
Params.sql     = 10
Params.iv_pr   = 200
Params.iv_pl   = 100
Params.n_e     = 1
Params.n_i     = 1000
Params.use_opt = True 
Params.lr      = 0.0005
Params.bs      = bs
make_parentbatch(tweets,bs,sql=Params.sql)
