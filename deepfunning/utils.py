'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-09-26 22:39:48
 * @desc 
'''
import torch




''' ------------Exponential Moving Average ------------'''

'''
How to use ？ Like Optimizer
- Initial
    ema=EMA(model,0.99)
- In Trianing
    def train():
        for i in epoches:
            for data in dataloader:
                ...
                optimizer.step()
                ema.update()
- In evaluation:
    def evaluate():
        ... test ...
        ema.apply_shadow()
        ... test ...
        ema.restore

'''
class EMA():
    def __init__(self,model,decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def update(self):
        for name,param in self.model.named_paramters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name]=new_average
    
    def register(self):
        for name,param in self.model.named_paramters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def apply_show(self):
        for name,param in self.model.named_paramters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name,param in self.model.named_paramters():
            if param.requires_grad:
                assert name in self.backup
                param.data =  self.backup[name]
        # self.backup=[]
        


