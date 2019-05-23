#encoding:utf-8
from pathlib import Path
import numpy as np
import torch

class Model_Checkpoint(object):
    def __init__(self,checkpoint_dir,logger,
                 target,arch,
                 optimode='min',epoch_freq=1,initial_best=None,
                 only_save_best = True
                 ):
        if isinstance(checkpoint_dir,Path):
            self.base_dir =  checkpoint_dir
        else :
            self.base_dir = Path(checkpoint_dir)
        self.arch = arch
        self.logger = logger
        self.target = target
        self.epoch_freq = epoch_freq
        self.only_save_best = only_save_best

        if optimode=='min':
            self.monitor_op = np.less
            self.best = np.Inf
        else :
            self.monitor_op = np.greater()
            self.best = -np.Inf
        if initial_best:
            self.best = initial_best
        if only_save_best:
            self.model_name  = 'best_{}_model.pth'.format(arch)
    def epoch_step(self,state,current):
        if self.only_save_best:
            if self.monitor_op(current,self.best):
                self.logger.info(f"\nEpoch {state['epoch']} : {self.target} improved from {self.best} to {current}")
                self.best = current
                state['best'] = self.best
                best_path = self.base_dir / self.model_name
                self.logger.info(f"model with best score saved to disk {best_path}")
                torch.save(state,str(best_path))
        else:
            filename  = self.base_dir / Path(f"epoch_{state['epoch']}_{state[self.target]}_{self.arch}.pth")
            if state['epoch'] % self.epoch_freq==0:
                self.logger.info(f"Epoch {state['epoch']} saved to disk {filename}")
                torch.save(state,str(filename))



def restore_checkpoint(resume_path ,model,optimizer):
    if isinstance(resume_path,Path):
        resume_path = str(resume_path)
    checkpoint =  torch.load(resume_path)
    best = checkpoint['best']
    start_epoch = checkpoint['epoch']+1

    if model:
        model.load_state_dict(checkpoint['model_state'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return [model,optimizer,start_epoch,best]


