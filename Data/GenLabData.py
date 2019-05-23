import gzip
import pymysql
import pymongo
from tqdm import tqdm
import time

db_dir = 'G:/freebase/'
preidicate_file = open(db_dir+'predicate_4.txt','r',encoding="utf-8")
conn_mongo = pymongo.MongoClient()
collection = conn_mongo.freebase.event
event_type = {}


event_type_info4 = open(db_dir+'event_type_info5.txt','r',encoding="utf-8")
event_instances = open(db_dir+'event_instances.txt','w',encoding="utf-8")
event_instances_sample = open(db_dir+'event_instances_sample_new.txt','w',encoding="utf-8")
while(True):
    line = event_type_info4.readline()
    if  line:
        event = line.split('\t')[0].split(':')[0]
        args = line.split('\t')[1:]
        args = [arg.split(':')[0] for arg in args]
        event_type[event] = [arg.strip() for arg in args]
    else:
        break

count=0
for event,args in event_type.items():
        print(count)
        count+=1
        instances = collection.find({event:{'$exists':True}},limit=2)
        for instance in instances:
            try:
                t1 = event.split('.')[0]
                t2 = event.split('.')[1]
                line = f"{instance['_id']}:{event}"
                for arg in args:
                            line+=f"\t{arg}:{instance[t1][t2][arg]}"
                event_instances_sample.write(line+'\n')
            except :
                continue

'''
instances = collection.find()
for index,instance in tqdm(enumerate(instances)):
    try:
        for event,args in event_type.items():
            t1 = event.split('.')[0]
            t2 = event.split('.')[1]
            line = f"{instance['_id']}:{event}"
            for arg in args:
                        line+=f"\t{arg}:{instance[t1][t2][arg]}"
            event_instances.write(line+'\n')
    except :
        continue
'''


'''
while(True):
    line = event_type_file3.readline()
    if  line:
        event = line.split('\t')[0].rstrip()
        args = line.rsplit('\t')[1:]
        event_type[event] = [arg.strip() for arg in args]
    else:
        break
#instances = collection.find({"american_football.football_conference":{'$exists':True}})

instances = collection.find()
global_count=0
for index,instance in tqdm(enumerate(instances)):
    if index > 30400736:
        for event,args in event_type.items():
            t1 = event.split('.')[0]
            t2 = event.split('.')[1]
            if instance.get(t1) and instance[t1].get(t2) and len(instance[t1][t2])>2:
                line = f"{event}\t{instance['_id']}"
                for arg in args:
                    if instance[t1][t2].get(arg):
                        line+=f"\t{arg}:{instance[t1][t2][arg]}"
                event_type_file4.write(line+'\n')

'''
'''
for event,args in tqdm(event_type.items()):
    #instances = collection.find({event:{'$exists':True}})
    instances = collection.find({event:{'$exists':True}})
    t1 = event.split('.')[0]
    t2 = event.split('.')[1]
    for instance in instances:
        if instance.get(t1) and instance[t1].get(t2) and len(instance[t1][t2])>2:
            line = f"{event}\t{instance['_id']}"
            for arg in args:
                if instance[t1][t2].get(arg):
                    line+=f"\t{arg}:{instance[t1][t2][arg]}"
            event_type_file4.write(line+'\n')
'''

'''
batch_size = 1
start=0
event_type_l=[]
total = 48853338
for event,args in event_type.items():
    event_type_l.append((event,args))
while start+batch_size<=len(event_type):
    start_time = time.time()
    sql=[]
    #for t in event_type_l[start:start+batch_size]:
    for event,arg in event_type_l[start:start+batch_size]:
        #event,args =t
        sql.append({event:{'$exists':True}})
    #instances = collection.find({event:{'$exists':True}})
    instances = collection.find({'$or':sql})
    instances_count = instances.count()
    for index,instance in tqdm(enumerate(instances)):
        for t in event_type_l[start:start+batch_size]:
            event,args = t
            t1 = event.split('.')[0]
            t2 = event.split('.')[1]
            if instance.get(t1) and instance[t1].get(t2) and len(instance[t1][t2])>2:
                line = f"{event}\t{instance['_id']}\t"
                for arg in args:
                    if instance[t1][t2].get(arg):
                        line+=f"\t{arg}:{instance[t1][t2][arg]}"
                event_type_file4.write(line+'\n')
    time_cost = (time.time()-start_time)
    log_file.write(f"batch_size ; {batch_size} iter :{start}-{start+batch_size-1} time/iter : {time_cost}")
    print(f"batch_size ; {batch_size} iter :{start}-{start+batch_size} time/iter : {time_cost}"
          f"predict_total : {total/instances_count*time_cost}")
    start +=batch_size
    batch_size = min(len(event_type)-start,batch_size+1)
'''








