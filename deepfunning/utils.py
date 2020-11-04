'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-09-26 22:39:48
 * @desc 
'''
import torch
import numpy as np
import random


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
        

""" Set random seed"""
def set_random_seed(seed = 10, deterministic = False, benchmark = False):
    """
    Args:
        seed: Random Seed
        deterministic:
                Deterministic operation may have a negative single-run performance impact, depending on the composition of your model. 
                Due to different underlying operations, which may be slower, the processing speed (e.g. the number of batches trained per second) may be lower than when the model functions nondeterministically. 
                However, even though single-run speed may be slower, depending on your application determinism may save time by facilitating experimentation, debugging, and regression testing.
        benchmark: whether cudnn to find most efficient method to process data
                If no difference of the size or dimension in the input data, Setting it true is a good way to speed up
                However, if not,  every iteration, cudnn has to find best algorithm, it cost a lot
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True