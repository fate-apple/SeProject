#encoding:utf-8
import os
import random
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
from collections import OrderedDict

def seed_everything(seed = 1024,device='cpu'):
    '''
    为random,numpy和Pytorch初始化seed
    :param seed:
    :param device:
    :return:
    '''
    #便于复现实验
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'cuda'in device:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class ProgressBar(object):
    def __init__(self,total,width=50):
        self.width  = width
        self.total = total

    def step(self,index,info,use_time):
        rate = int( (index+1) / self.total * 100 )
        if rate >100:
            rate = 100
        show_bar = f"[{ int(self.width*rate/100)*'>':<{self.width}s}] {rate}%"
        show_info = f"\r[training] {index+1}/{self.total} {show_bar} -{use_time:.2f}s/step"
        show_info += '--'.join([f' {key}: {value:.4f} ' for key,value in info.items()])
        print(show_info,end='')

def prepare_device(n_gpu_use,logger):
    if isinstance(n_gpu_use,int):
        n_gpu_use = range(n_gpu_use)
    n_gpu = torch.cuda.device_count()
    if len(n_gpu_use)>n_gpu:
        if(n_gpu==0):
            logger.warning("Warning: There is no Gpu available,training will be performed on Cpu")
        else:
            print('debug')
            logger.warning("Warning: {} Gpu configured,but only {} Gpu available".format(len(n_gpu_use,n_gpu)))
            logger.warning("Warning: There is not enough Gpu available")
    device = torch.device(f'cuda:{n_gpu_use[0]}' if len(n_gpu_use)>0 else 'cpu')
    device_ids = n_gpu_use
    return device,device_ids

def model_device(model,logger,n_gpu =0):
        device,device_ids = prepare_device(n_gpu,logger)
        if len(device_ids)>1:
            logger.info(f'{len(device_ids)} Gpu used for training')
            model = torch.nn.DataParallel(model,device_ids=device_ids)
        if len(device_ids)==1:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])
        model = model.to(device)
        return model,device

def summary_info(model , *inputs,batch_size = 1 ,show_input = True):
    summary = OrderedDict()
    hooks = []

    def register_hook(module):
        def hook(module,input,output=None):
            class_name  = str(module.__class__).split('.')[-1]
            module_idx = len(summary)

            mid =  f'{class_name}-{module_idx+1}'
            summary[mid] = OrderedDict()
            #summary[mid]['input_shape'] = [batch_size]+list(input[0].size())
            summary[mid]['input_shape'] = list(input[0].size())

            #TODO

            if not show_input and output is not None:
                raise NotImplementedError
                '''
                if isinstance(output,(list,tuple)):
                    for out in output:
                        if isinstance(out,torch.Tensor):
                            summary[mid]['output_shape'][[-1]+list(out[0].size())[1:]]
                '''

            params = 0
            if hasattr(module,'weight') and hasattr(module.weight,'size'):
                #return product of all inputs
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[mid]['trainable'] =  module.weight.requires_grad
            if hasattr(module,'bias') and hasattr(module.bias,'size'):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[mid]['params'] = params
        if not isinstance(module,nn.Sequential) and not isinstance(module,nn.ModuleList) and not (module==model):
            if show_input:
                #hooks.append(module.register_forword_pre_hook(hook))
                #The hook will be called every time before :func:`forward` is invoked.
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    '''applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        as well as self. Typical use includes initializing the parameters of a model'''
    model.apply(register_hook)
    model(*inputs)

    print(f"{'-'*25}")
    for h in hooks:
        h.remove()
    if show_input:
        line_new = f"{'layer(type)':>25}  {'input shape':>25}  {'param #':>15}"
    print(line_new)
    print(f"{'-'*25}")

    total_params,total_output,trainable_params = 0,0,0
    for layer in summary:
        if show_input:
            params = f"{summary[layer]['params']:,}"
            line_new = f"{layer:>25}  {str(summary[layer]['input_shape']):>25} {params:>15}"
            total_output += np.prod(summary[layer]['input_shape'])
        else :
            params = f"{summary[layer]['params']:,}"
            line_new = f"{layer:>25}  {str(summary[layer]['output_shape']):>25} {params:>15}"
            total_output += np.prod(summary[layer]['output_shape'])
        print(line_new)
        total_params += summary[layer]['params']
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params +=summary[layer]['params']

    print('='*25)
    print(f"total_params    :   {total_params}")
    #print(f"total_outputs   :   {total_output}")
    print(f"trainable_params    :   {trainable_params}")
    print('-'*25)

def load_bert(model_path,model = None,optimizer = None):
    '''
    加载模型
    :param resume_path:
    :param model:
    :param optimizer:
    :return:
    '''
    if isinstance(model_path,Path):
        model_path = str(model_path)
    print(f"load model from {model_path}")
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['model_state']

    best = checkpoint['best']
    start_epoch = checkpoint['epoch'] + 1
    if model:
        model.load_state_dict(state_dict)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return [model,optimizer,best,start_epoch]

Causal_Cue_Words = ['affected by','affect','affects',
                    'and consequently','and hence',
                    'as a consequence of','as a consequence','as a result of',
                    'because of','because',
                    'bring on','brings on','brought on',
                    'cause','caused by','caused','causes','causing',
                    'consequently',
                    'decreased by','decrease','decreases',
                    'due to',
                    'effect of',
                    'for this reson alone',
                    'gave rise to','give rise to','given rise to','giving rise to',
                    'hence',
                    'in consequence of',
                    'in reseponse to',
                    'increase','increases','increased by',
                    'induce','inducing','induced',
                    'lead to','leading to','leads to','led to',
                    'on account of',
                    'owing to',
                    'reason for','reason of','resons for','reasons of',
                    'result from','resulting from','resulted from','results from',
                    'so that',"that's why",'the result is',
                    'thereby','therefor','thus']
        









