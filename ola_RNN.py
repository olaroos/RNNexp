import math
import torch

def rnn_forward(learn,hidden,xb,yb):
    learn.model.train()
    if xb[0,0,1].item() == 1: hidden = learn.model.initHidden(xb.shape[0])                   
    loss = 0 
    for char in range(xb.shape[1]):
        x,y = xb[:,char],yb[:,char]
        idx = (y != 0).nonzero()
        if idx.nelement() == 0: return output, hidden.detach(), loss/(char+1)
        x,y,hidden = unpad_rnn(x,y,hidden)
        output,hidden = learn.model.forward(x,hidden)
        loss += learn.loss_fn(output,y)                

    return output,hidden.detach(),loss/(char+1)

def unpad_rnn(x,y,hidden):
    idx = (y != 0).nonzero()        
    if idx.shape[0] == 1: idx = idx[0]
    else: idx = idx.squeeze()
    return x[idx],y[idx],hidden[idx]

def get_valid_rnn(learn,itters=30):
    print(f"""getting validation""")    
    learn.model.eval()
    tot_loss = 0 
    with torch.no_grad():
        hidden = learn.model.initHidden(15)
        for xb,yb in iter(learn.data.valid_dl): 
            output, hidden, loss = rnn_forward(learn,hidden,xb,yb)  
            if loss != 0: tot_loss += loss.item()/xb.shape[0]
            if learn.data.valid_dl.nb_itters() == itters: 
                return tot_loss/learn.data.valid_dl.nb_itters()
        
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