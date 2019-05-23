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
for temp_a in a:
    print('debug')
b = a.softmax(-1)
c =a[:,1:-1,:]
r,_ = x.shape
ignore = [0]
l = [i for i in range(r) if  i not in ignore]
y =x[l,1]

dict ={1:'a',2:'b'}
l  = dict.values()

a= a.view(a.size(1),a.size(-1))
l=[i for i in range(a.size(0)) if i != 0]
logits = a[l,...]

a = torch.FloatTensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
b = torch.FloatTensor([[1,2,3,4],[5,6,7,8],[9,10,11,12]])


a =[1,2]
b= [2,3]
for i,batch in enumerate(zip(a,b)):
    print('debug')
d = [[]*5]
d2 = []*5
print('test')

