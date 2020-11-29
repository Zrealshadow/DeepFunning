'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-11-29 15:02:50
 * @desc BaseWrapper for model wrapper in deeplearning
'''
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn


class BaseWrapper(object):
    """Base class for Wrapper
    """
    def self(self,model:nn.Module, config:Dict):
        super(self,BaseWrapper).__self__()
        self.device = torch.device(config['cuda']) if torch.cuda.is_available() else torch.device('cpu')

        self.model = model.to(self.device)
        self.loss_fn = config['loss_fn']
        self.start_epoch = 0
        self.checkpoint_epoch = config['checkpoint_epoch']
        self.print_step = config['print_step']
        self.init_lr = config['lr']
        self.best_model = self.model
        self.batch_size = config['batch_size']
        self.checkpoint_dir = config['checkpoint_dir']
        #data_loader
        self.train_dataloader = config['train_dataloader']
        #subclass have to overwrite
        self.optimizer = None
        self.best_score = None

    def __getattrfromconfig():
        pass 
    def train(self):
        raise NotImplementedError()
    
    def validation(self):
        raise NotImplementedError()

    def test_performance(self):
        raise NotImplementedError()

    def save_check_point(self, filename:str = None, epoch:int = None) -> None:
        """ save the key attribute in this wrapper
        key attribute:
        d = {
            'epoch' : epoch,
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'best_score' : best_score,
            'best_model' : best_pred
        }
        """
        dir_path = Path(self.checkpoint_dir)
        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        if filename is None:
            if epoch is None:
                filename = self.model.name+'_'+self.model.version+'_'+now+'_checkpoint.pth.tar'
            filename = self.model.name+'_'+self.model.version+'_'+'epoch:'+str(epoch)+'_'+now+'_checkpoint.pth.tar'
        
        filepath = Path.joinpath(dir_path, filename)
        d = {
            'epoch':self.start_epoches,
            'state_dict' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'best_model' : self.best_model.state_dict(),
            'best_score' : self.best_score
        }
        torch.save(d, filepath)
    
    def load_check_point(self,file_name:Optional[str] = None) -> None:
        """load the save model file
        Args:
            file_name: filename not filepath, it will concatenate the checkpoint_dir and filename 
                to create the whole path. It filename is none, it will choose the latest checkpoint file
                under the checkpoint directory.
        """
        dir_path = Path(self.checkpoint_dir)
        if file_name == None:
            import os
            flist = os.listdir(dir_path)
            if not flist:
                msg = 'No checkpoint file'
                raise ValueError(msg=msg)
            filepath = Path.joinpath(dir_path,max(flist))
        else:
            filepath = Path.joinpath(dir_path,file_name)
        
        checkpoint = torch.load(filepath, map_location='cpu')
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.best_model.load_state_dict(checkpoint['best_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_score = checkpoint['best_score']

 