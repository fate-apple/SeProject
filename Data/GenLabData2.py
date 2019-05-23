#encoding: utf-8
'''
@time: 2019/5/7 21:57
@desc:
'''
import pymongo
import tqdm
db_dir = 'G:/freebase/'
preidicate_file = open(db_dir+'predicate_4.txt','r',encoding="utf-8")
conn_mongo = pymongo.MongoClient()
collection = conn_mongo.freebase.event
event_type = {}
event_type_file3 = open(db_dir+'event_type3.txt','r',encoding="utf-8")
event_type_info = open(db_dir+'event_type_info.txt','r',encoding="utf-8")
#event_type_info2 = open(db_dir+'event_type_info2.txt','w',encoding="utf-8")
event_type_info3 = open(db_dir+'event_type_info3.txt','w',encoding="utf-8")


while(True):
    line = event_type_file3.readline()
    if  line:
        event = line.split('\t')[0].rstrip()
        args = line.rsplit('\t')[1:]
        event_type[event] = [arg.strip() for arg in args]
    else:
        break
summary={}
count=0
'''
for event,args in event_type.items():
    t1 = event.split('.')[0]
    t2 = event.split('.')[1]
    instances = collection.find({f"{event}":{'$exists':True}})
    total = 0
    summary[event]={}

    for arg in args:
        summary[event][arg] = 0
    for instance in instances:
        total+=1
        for arg in args:
            if instance[t1][t2].get(arg):
                        summary[event][arg]+=1
    summary[event]['total']=total
    line = f"{event}:{total  }\t"+'\t'.join(["{}:{}".format(arg,summary[event][arg]) for arg in args])
    print(str(count)+line)
    summary[event]['line']=line
    count+=1
    event_type_info.write(line+'\n')
'''

for line in event_type_info:
    event_type = line.split('\t')[0].split(':')[0]
    total = int(line.split('\t')[0].split(':')[1])
    args = line.split('\t')[1:]
    args = [[arg.split(':')[0],int(arg.split(':')[1])] for arg in args]
    args = sorted(args,key=lambda d:d[1],reverse=True)
    summary[event_type] = {}
    summary[event_type]['total'] = total
    summary[event_type]['line'] = f"{event_type}:{total}\t"+'\t'.join([f"{arg[0]}:{arg[1]}" for arg in args])+'\n'
    summary[event_type]['args'] = args

vs = [ v for v in sorted(summary.values(),key=lambda d:d['total'],reverse=True)]
#summary= sorted(summary.items(), key=lambda d:d[1]['total'],reverse=True)
count =0
for v in vs:
    if v['total']>1000 and v['args'][2][1]/v['total']>0.5:
        count+=1
        event_type_info3.write(v['line'])
print(count)









