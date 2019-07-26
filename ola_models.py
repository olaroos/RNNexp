import torch, torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

def cuda(input):
    if torch.cuda.is_available(): return input.cuda()
    return input

class SGRU(nn.Module):
    def __init__(self, in_sz, hd_sz, n_stacks):
        super(SGRU,self).__init__()
        self.in_sz = in_sz
        self.hd_sz = hd_sz
        self.n_stacks = n_stacks
        self.GRUs = []
        for i in range(n_stacks):
            self.GRUs.append(cuda(GRU(in_sz,hd_sz)))
                
    def batch_forward(self,xb,yb,hds,loss_fn):
        if xb[0,0,1].item() == 1 or hds is None: hds = self.initHidden()                   
        # I want hidden outputs to have size [hd_sz, n_stacks]
        loss = 0
        for char in range(0,xb.shape[1]):
            predict = xb[:,char]
            for stack in range(self.n_stacks-1):
                print(f"""stack is {stack}, char is {char}""")
                gru = self.GRUs[stack]
                predict, hds[:,stack] = gru.forward(predict,hds[:,stack],True)
            predict, hds[:,stack+1] = gru.forward(predict,hds[:,stack+1],False)
            loss += loss_fn(predict,yb[:,char])
        return predict, hidds
        
    def initHidden(self, hd_sz=None): 
        if hd_sz is None: return cuda(torch.zeros(self.hd_sz,self.n_stacks))
        else: return cuda(torch.zeros(hd_sz,self.n_stacks))
    
    def parameters(self):
        for stack in range(n_stacks):
            for param in iter(self.GRUs[stack].parameters()):
                yield param
        
class GRU(nn.Module):
    def __init__(self, in_sz, hd_sz):
        super(GRU,self).__init__()
        self.in_sz = in_sz
        self.hd_sz = hd_sz

        self.x_lin = nn.Linear(self.in_sz,3*self.hd_sz)                
        self.h_lin = nn.Linear(self.hd_sz,3*self.hd_sz)
        
        self.up_sig = nn.Sigmoid()
        self.re_sig = nn.Sigmoid()
            
        self.o1      = nn.Linear(self.hd_sz+self.in_sz,self.in_sz)  

        self.softmax = nn.LogSoftmax(dim=1)   
        self.loss    = 0 
            
    def forward(self,input,hidden,to_stacked=False):        
        print(f"""input shape {input.shape}""")
        x = self.x_lin(input)   
        print(f"""x shape {x.shape} """)
        print(f"""h shape {hidden.shape} """)        
        h = self.h_lin(hidden)        
        print(f"""h shape {hidden.shape} """)        
        x_u,x_r,x_n = x.chunk(3,1)
        h_u,h_r,h_n = h.chunk(3,1)
        update_gate = self.up_sig(x_u+h_u)        
        reset_gate  = self.re_sig(x_r+h_r)
        new_gate    = torch.tanh(x_n + reset_gate * h_n)         
        h_new       = update_gate * hidden + (1 - update_gate) * new_gate 
        
        combined   = torch.cat((input,h_new),1)
        combined   = self.o1(combined)
        
        if to_stacked: return combined, h_new
        else: return self.softmax(combined), h_new
    
    def batch_forward(self,xb,yb,hidden,loss_fn):
        self.train()
        if xb[0,0,1].item() == 1: hidden = self.initHidden(xb.shape[0])                   
        loss = 0 
        for char in range(xb.shape[1]):
            x,y           = xb[:,char],yb[:,char]
            x,y,hidden    = unpad(x,y,hidden)
            if x.shape[0] == 0: break
            print(x.shape)
            print(hidden.shape)
            output,hidden = self.forward(x,hidden)
            loss += loss_fn(output,y)    
        return output,hidden.detach(),loss/(char+1)

    def initHidden(self, bs):
        return cuda(torch.zeros(bs,self.hd_sz))