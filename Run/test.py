#encoding: utf-8
'''
@time: 2019/5/22 16:55
@desc:
'''
import  re
import numpy as np
import torch
x = np.array([[1,2],[3,4],[5,6]])
a = torch.FloatTensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12]]])
b = a.softmax(-1)
c =a[:,1:-1,:]
r,_ = x.shape
l = [i for i in range(r) if i!=1]
y =x[l,1]
print('test')

