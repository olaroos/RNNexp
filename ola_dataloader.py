import random
import math
import torch 
import re

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

def pad(str_list,sql=1,token='£'):
    f"""pad all strings in a list to max_len"""
    max_len = math.ceil(len(max(str_list, key=len))/sql)*sql
    for idx, row in enumerate(str_list):        
        str_list[idx] = row + token*(max_len-len(row))
    if len(str_list) == 1: return str_list[0]
    return str_list

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
    return sbx,sby

class TweetDataLoader():
    def __init__(self,data,tweets,bs,sql,shuffle=False):    
#         assert(math.floor(len(tweets)/bs)==len(tweets)/bs)
        self.tweets  = tweets
        self.bs      = bs         
        self.sql     = sql
        self.encoder = data.encoder
        self.decoder = data.decoder
        self.i       = -1
        self.ii      = 0 
        self.shuffle = shuffle        
        
    def reset(self):
        if self.shuffle: random.shuffle(self.tweets)
        self.i  = -1
        self.ii = 0
        
    def nb_itters(self):
        return self.ii 
    
    def __iter__(self):  
        self.reset()
        while True:
            self.i += 1
            twt      = self.tweets[self.i*self.bs:(self.i+1)*self.bs]
            sbx,sby  = mk_tweetbatch(twt,self.encoder,self.bs,self.sql)
            sbloader = iter(SBDataLoader(sbx,sby))            
            try:
                while True:                
                    self.ii+=1                    
                    yield next(sbloader) 
            except StopIteration:
                self.ii-=1
                pass            
            if self.i==round(len(self.tweets)/self.bs)-2: 
                break

class SBDataLoader():
    def __init__(self, sbx, sby): 
        self.sbx, self.sby = sbx, sby
    def __iter__(self):
        for j in range(self.sbx.shape[1]): yield cuda(self.sbx[:,j]), cuda(self.sby[:,j])

def batch_strings(tweets,bs,sql=1):
    f"""creates a list of batchsize-list of strings of same length and sort each batch with longest string first."""    
    """NOT SURE ABOUT THIS OFFSET, BUT THE PREVIOUS CODE ALWAYS MADE A 0"""
    offset = -1*((len(tweets)/bs)*10%2!=0)    
#     offset = -1*((math.floor(len(tweets)/bs)==len(tweets)/bs)==0)    
    bch_strs = [] 
    for i in range(round(len(tweets)/bs)+offset):
        strings = tweets[i*bs:(i+1)*bs]
        strings.sort(key=len,reverse=True)
        pad_strings = pad(strings,sql)
        bch_strs.append(pad_strings)
    return bch_strs