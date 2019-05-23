#encoding: utf-8
'''
@time: 2019/5/12 11:03
@desc:
'''
source = open('source_BIO_2014_cropus.txt','r',encoding='utf-8')
target = open('target_BIO_2014_cropus.txt','r',encoding='utf-8')
temp_source = open('temp_source_BIO_2014_cropus.txt','w',encoding='utf-8')
temp_target = open('temp_target_BIO_2014_cropus.txt','w',encoding='utf-8')

for i in range(10):
    line1 = source.readline()
    line2 = target.readline()
    temp_source.write(line1)
    temp_target.write(line2)
