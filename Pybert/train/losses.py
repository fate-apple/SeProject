#encoding: utf-8
'''
@time: 2019/5/4 15:27
@desc:warpper of pytorch loss func
'''

from torch.nn import  BCEWithLogitsLoss,CrossEntropyLoss

class BCEWithLogits(object):
    def __init__(self):
        self.fn = BCEWithLogitsLoss()
    def __call__(self,target,logits):
        loss  =  self.fn(input =logits,target = target)
        return loss

class CrossEntropy(object):
    def __init__(self):
        self.fn = CrossEntropyLoss(ignore_index=0)
    def __call__(self,logits,target):
        loss  =  self.fn(input =logits,target = target,)
        return loss
