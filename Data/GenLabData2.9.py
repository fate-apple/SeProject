#encoding: utf-8
'''
@time: 2019/5/9 23:35
@desc: sample and 21type in github
'''
import pymongo
import tqdm
db_dir = 'G:/freebase/'
preidicate_file = open(db_dir+'predicate_4.txt','r',encoding="utf-8")
conn_mongo = pymongo.MongoClient()
collection = conn_mongo.freebase.event
event_type = {}



event_type_sample = open(db_dir+'event_instances_sample2.txt','r',encoding="utf-8")
key_argument_list_by_KR = open(db_dir+'key_argument_list_by_KR.txt','r',encoding="utf-8")
event_type_info = open(db_dir+'event_type_info.txt','r',encoding="utf-8")
event_type_info5 = open(db_dir+'event_type_info5.txt','w',encoding="utf-8")
event_list=[]
while(True):
    line = event_type_sample.readline()
    if  line!='':
        try:
            event = line.split('\t')[0].split(':')[1]
            event_list.append(event)
        except:
            pass
    else:
        break
while(True):
    line = key_argument_list_by_KR.readline()
    if  line!='':
        try:
            event = line.strip()
            if len(event.split('.'))>1:
                event_list.append(event)
        except:
            pass
    else:
        break
summary={}
for line in event_type_info:
    event_type = line.split('\t')[0].split(':')[0]
    if event_type in event_list:
        total = int(line.split('\t')[0].split(':')[1])
        args = line.split('\t')[1:]
        args = [[arg.split(':')[0],int(arg.split(':')[1])] for arg in args]
        args = sorted(args,key=lambda d:d[1],reverse=True)
        args = [arg for arg in args if arg[1]/total>0.75]
        summary[event_type] = {}
        summary[event_type]['total'] = total
        summary[event_type]['line'] = f"{event_type}:{total}\t"+'\t'.join([f"{arg[0]}:{arg[1]}" for arg in args])+'\n'
        summary[event_type]['args'] = args
    else:
        print(event_type)

vs = [ v for v in sorted(summary.values(),key=lambda d:d['total'],reverse=True)]
count =0
for v in vs:
        count+=1
        event_type_info5.write(v['line'])
print(count)
