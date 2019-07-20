class Callback():
    def begin_fit(     self,learn): self.learn = learn;      return True
    def after_fit(     self):                                return True
    def begin_epoch(   self,epoch): self.epoch = epoch;      return True
    def begin_validate(self):                                return True
    def after_epoch(   self):                                return True 
    def begin_batch(   self,xb,yb): self.xb,self.yb = xb,yb; return True
    def after_loss(    self,loss):  self.loss=loss;          return True
    def after_backward(self):                                return True
    def after_step(    self):                                return True

class CallbackHandler():
    def __init__(self,cbs=None):
        self.cbs = cbs if cbs else []

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
        res = self.in_train
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
    
class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.learn.n_epochs=0.
        self.learn.n_iter=0
    
    def after_batch(self):
        if not self.learn.in_train: return
        self.learn.n_epochs += 1./self.iters
        self.learn.n_iter   += 1
        
    def begin_epoch(self):
        self.learn.n_epochs=self.epoch
        self.model.train()
        self.learn.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.learn.in_train=False