import torch, torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

from ola_nondepfun import * 


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
            
    def forward(self,x,hidden_in):   
        hidden_out = cuda(torch.zeros(hidden_in.shape))        
        for stack in range(self.n_stacks-1):
            gru = self.GRUs[stack]
            x, hidden_out[stack] = gru.forward(x,hidden_in[stack],True)
        output, hidden_out[stack+1] = gru.forward(x,hidden_in[stack+1],False)            
        return output, hidden_out.detach()
        
    def batch_forward(self,xb,yb,hidden_in,loss_fn,calc_acc=False):
        if xb[0,0,1].item() == 1 or hidden_in is None: hidden_in = self.initHidden(xb.shape[0])                   
        accu = 0
        loss = 0
        for char in range(0,xb.shape[1]):
            x,y           = xb[:,char],yb[:,char]
            x,y,hidden_in = unpad(x,y,hidden_in)
            if x.shape[0] == 0: break 
            hidden_out = cuda(torch.zeros(hidden_in.shape))
            for stack in range(self.n_stacks-1):
                gru = self.GRUs[stack]
                x, hidden_out[stack] = gru.forward(x,hidden_in[stack],True)
            output, hidden_out[stack+1] = gru.forward(x,hidden_in[stack+1],False)
            hidden_in = hidden_out.clone()
            """divide by the bs used for the current character"""
            accu += get_accu(output,y)/x.shape[0]    
            """not sure if loss here is already averaged over bs...""" 
            loss += loss_fn(output,y)/x.shape[0]
        return output, hidden_out.detach(), loss/(char+1), accu/(char+1)
        
    def initHidden(self, bs): return cuda(torch.zeros(self.n_stacks,bs,self.hd_sz))
    
    def parameters(self):
        for stack in range(self.n_stacks):
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
        x = self.x_lin(input)   
        h = self.h_lin(hidden)           
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
        accu = 0 
        for char in range(xb.shape[1]):
            x,y           = xb[:,char],yb[:,char]
            x,y,hidden    = unpad(x,y,hidden)
            if x.shape[0] == 0: break
            output,hidden = self.forward(x,hidden)
            """divide by the bs used for the current character"""
            accu += get_accu(output,y)/x.shape[0]    
            """not sure if loss here is already averaged over bs...""" 
            loss += loss_fn(output,y)/x.shape[0]
        return output,hidden.detach(),loss/(char+1),accu/(char+1)

    def initHidden(self, bs):
        return cuda(torch.zeros(bs,self.hd_sz))
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN,self).__init__()
        self.hd_sz  = hidden_size
        self.in_sz  = input_size
        self.out_sz = output_size
        
        combined = input_size+hidden_size
        
        self.h1      = nn.Linear(combined, combined)  
        self.h1relu  = nn.ReLU(combined)
        
        self.h2      = nn.Linear(combined, hidden_size)

        self.o1      = nn.Linear(combined, combined)
        self.relu    = nn.ReLU(combined)

        self.o2      = nn.Linear(combined, combined)
        self.relu2   = nn.ReLU(combined)
        
        self.o3      = nn.Linear(combined, combined)
        self.relu3   = nn.ReLU(combined)
        
        self.o4      = nn.Linear(combined, combined)
        self.relu4   = nn.Linear(combined, combined)
        
        self.o5      = nn.Linear(combined, input_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)    
        
        hidden   = self.h1(combined)
        hidden   = self.h1relu(hidden)

        hidden   = self.h2(hidden)        
        hidden   = torch.tanh(hidden)
        
        output   = self.o1(combined)
        output   = self.relu(output)
        
        output   = self.o2(output)
        output   = self.relu2(output)
        
        output   = self.o3(output)
        output   = self.relu3(output)
        
        output   = self.o4(output)
        output   = self.relu4(output)
        
        output   = self.o5(output)
        output   = self.softmax(output)
        return output, hidden

    def batch_forward(self,xb,yb,hidden,loss_fn):
        self.train()
        if xb[0,0,1].item() == 1: hidden = self.initHidden(xb.shape[0])                   
        loss = 0
        accu = 0 
        for char in range(xb.shape[1]):
            x,y           = xb[:,char],yb[:,char]
            x,y,hidden    = unpad(x,y,hidden)
            if x.shape[0] == 0: break
            output,hidden = self.forward(x,hidden)
            """divide by the bs used for the current character"""
            accu += get_accu(output,y)/x.shape[0]    
            """not sure if loss here is already averaged over bs...""" 
            loss += loss_fn(output,y)/x.shape[0]
        return output,hidden.detach(),loss/(char+1),accu/(char+1)    
    
    def initHidden(self,bs):
        return cuda(torch.zeros(bs,self.hd_sz))  