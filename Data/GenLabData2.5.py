#encoding: utf-8
'''
@time: 2019/5/9 19:21
@desc:
'''
import pymongo
import tqdm
db_dir = 'G:/freebase/'
preidicate_file = open(db_dir+'predicate_4.txt','r',encoding="utf-8")
conn_mongo = pymongo.MongoClient()
collection = conn_mongo.freebase.event
event_type = {}

event_type_info3 = open(db_dir+'event_type_info3.txt','r',encoding="utf-8")
event_type_info4 = open(db_dir+'event_type_info4.txt','w',encoding="utf-8")
summary={}
for line in event_type_info3:
    event_type = line.split('\t')[0].split(':')[0]
    total = int(line.split('\t')[0].split(':')[1])
    args = line.split('\t')[1:]
    args = [[arg.split(':')[0],int(arg.split(':')[1])] for arg in args]
    args = sorted(args,key=lambda d:d[1],reverse=True)
    args = [arg for arg in args if arg[1]/total>0.75]
    summary[event_type] = {}
    summary[event_type]['total'] = total
    summary[event_type]['line'] = f"{event_type}:{total}\t"+'\t'.join([f"{arg[0]}:{arg[1]}" for arg in args])+'\n'
    summary[event_type]['args'] = args

vs = [ v for v in sorted(summary.values(),key=lambda d:d['total'],reverse=True)]
count =0
for v in vs:
    if v['total']>1000and v['args'][2][1]/v['total']>0.75:
        count+=1
        event_type_info4.write(v['line'])
print(count)
