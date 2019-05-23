#encoding:utf-8
import pandas as pd
import tqdm
import random
import Pybert.utils.utils
import json

class DataTransformer(object):
    def __init__(self,logger,seed,encoding='utf-8'):
        self.seed = seed
        self.logger = logger
        self.encoding = encoding

class LabelDT(DataTransformer):
    def __init__(self,logger,seed,add_unk=True):
        super(LabelDT,self).__init__(logger,seed)
        self.add_unk = add_unk

    def text_write(self,path,data,encoding='utf-8'):
        '''

        :param path: the path of file
        :param data:  data with format (sentence,data)
        :param encoding:
        :return:
        '''
        with open(path,'w',endoding=encoding) as fw:
            for sentence,target in tqdm(data,desc = 'write sentence and target jointly to disk'):
                target = [str(x) for x in target]
                line = 't'.join([sentence,','.join(target)])
                fw.write(line+'\n')


    def read_data(self,raw_data_path,preprocessor=None,is_train=True):
        '''
        读取并预处理数据,sample:
        "000103f0d9cfb60f",
        "D'aww! He matches this background colour I'm seemingly stuck with. Thanks.  (talk) 21:51, January 11, 2016 (UTC)",
        0,0,0,0,0,0
        :param raw_data_path:
        :param preprocessor:
        :param is_train:
        :return: targets
        '''
        targets,sentences = [],[]
        f = open(raw_data_path,'r',encoding='utf-8')
        for line in tqdm(f):
            if is_train:
                target = line[2:]
            else:
                target  = [-1,-1]
            sentence = str(line[1])

            if preprocessor:
                if(isinstance(preprocessor,list)):
                    for p in preprocessor:
                        sentence = p(sentence)
                else :
                    sentence = preprocessor(sentence)
            if sentence:
                targets.append(target)
                sentences.append(sentences)
            return targets,sentences

    def SplitDatasetTV(self,x,y,valid_size,
                           stratify = False,
                           shuffle=True,
                           save = True,
                           train_path = None,
                           valid_path=None):
            '''
            将原始数据集划分为valid和train部分
            :param self:
            :param x:
            :param y:
            :param valid_size: 验证集所占比例,0~1
            :param stratify:
            :param shuffle:
            :param save:
            :param train_path:
            :param valid_path:
            :return:
            '''
            self.logger.info('Split Dateset to train and valid')

            if stratify:
                #split by lab_yi Proportionally
                num_classes = len(list(set(y)))
                trainData,validData = []
                bucket = [ [] for _ in range(num_classes)]
                for xi,yi in tqdm(zip(x,y),desc='bucket'):
                    bucket[int(yi)].append((xi,yi))
                del x,y
                for lab_yi in tqdm(bucket,desc='split Proportionally'):
                    n = len(lab_yi)
                    if n==0:
                        continue
                    validPart = n*valid_size
                    if shuffle:
                        random.shuffle(lab_yi)
                    validData.extend(lab_yi[:validPart])
                    trainData.extend(lab_yi[validPart:])
                if shuffle:
                    random.shuffle(trainData)
            else:
                data = []
                for xi,yi in tqdm(zip(x,y),desc = 'together'):
                    data.append((xi,yi))
                del x,y
                n = len(data)
                validPart = int(n*valid_size)
                if shuffle:
                    random.shuffle(data)
                validData = data[:validPart]
                trainData = data[validPart:]
                if shuffle:
                    random.shuffle(data)

            if save:
                self.text_write(filename=train_path,data = trainData)
                self.text_write(filename=valid_path,data = validData)

            return trainData,validData
    '''
SOCCER NN B-NP O
- : O O
JAPAN NNP B-NP B-LOC
GET VB B-VP O
LUCKY NNP B-NP O
WIN NNP I-NP O
, , O O
CHINA NNP B-NP B-PER
IN IN B-PP O
SURPRISE DT B-NP O
DEFEAT NN I-NP O
. . O O
'''
class Conll2003DT(DataTransformer):

    def __init__(self,logger,seed):
        super(Conll2003DT,self).__init__(logger,seed)

    def read_data(self,path):
        l_words,l_ners = [],[]
        entries = open(path,'r',encoding=self.encoding).read().strip().split('\n\n')
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            ners = [line.split()[-1] for line in entry.splitlines()]
            l_words.append(['[CLS]']+words+['[SEP]'])
            l_ners.append(['<PAD>']+ners+['<PAD>'])
        return (l_words,l_ners)

class BIO2014DT(DataTransformer):

    def __init__(self,logger,seed):
        super(BIO2014DT,self).__init__(logger,seed)

    def read_data(self,source_path,target_path):
        max_len = 254
        l_words,l_ners = [],[]
        count =0
        with open(source_path,'r',encoding='utf-8') as f1,open(target_path,'r',encoding='utf-8') as f2:
            for line1, line2 in zip(f1,f2):
                words = line1.strip('\n').split()
                ners = line2.strip('\n').split()
                if count <2000:
                    print(f"{count} :   {len(words)}")
                count+=1

                l_words.append(['[CLS]']+words+['[SEP]'])
                #l_ners.append(['<PAD>']+ners+['<SEP>'])
                l_ners.append(['<PAD>']+ners+['<PAD>'])
        return (l_words,l_ners)

    def SplitDatasetTV(self,x,y,valid_size,
                           stratify = False,
                           shuffle=True,
                           save = True,
                           train_path = None,
                           valid_path=None):
            '''
            将原始数据集划分为valid和train部分
            :param self:
            :param x:
            :param y:
            :param valid_size: 验证集所占比例,0~1
            :param stratify:
            :param shuffle:
            :param save:
            :param train_path:
            :param valid_path:
            :return:
            '''
            self.logger.info('Split Dateset to train and valid')

            if stratify:
                #split by lab_yi Proportionally
                num_classes = len(list(set(y)))
                trainData,validData = []
                bucket = [ [] for _ in range(num_classes)]
                for xi,yi in tqdm(zip(x,y),desc='bucket'):
                    bucket[int(yi)].append((xi,yi))
                del x,y
                for lab_yi in tqdm(bucket,desc='split Proportionally'):
                    n = len(lab_yi)
                    if n==0:
                        continue
                    validPart = n*valid_size
                    if shuffle:
                        random.shuffle(lab_yi)
                    validData.extend(lab_yi[:validPart])
                    trainData.extend(lab_yi[validPart:])
                if shuffle:
                    random.shuffle(trainData)
            else:
                data = []
                for xi,yi in tqdm(zip(x,y),desc = 'together'):
                    data.append((xi,yi))
                del x,y
                n = len(data)
                validPart = int(n*valid_size)
                if shuffle:
                    random.shuffle(data)
                validData = data[:validPart]
                trainData = data[validPart:]
                if shuffle:
                    random.shuffle(data)

            if save:
                self.text_write(filename=train_path,data = trainData)
                self.text_write(filename=valid_path,data = validData)

            return trainData,validData

    def text_write(self,path,data,encoding='utf-8'):
        '''

        :param path: the path of file
        :param data:  data with format (sentence,data)
        :param encoding:
        :return:
        '''
        with open(path,'w',encoding=encoding) as fw:
            for sentence,target in data:
                for word,ner in zip(sentence,target):
                    fw.write(word+'\t'+ner+'\n')
                fw.write('\n')

class MnliDT(DataTransformer):

    def __init__(self,logger,seed,label=None):
        super(MnliDT,self).__init__(logger,seed)
        self.LABEL_MAP = label

    def read_data(self,data_path):
        #fw = open('temp_dict.txt','w',encoding='utf-8')
        data = []
        types =[]
        with open(data_path) as f:
            for line in f:
                loaded_example = json.loads(line)
                if loaded_example["gold_label"] not in self.LABEL_MAP:
                    continue
                loaded_example["label"] = self.LABEL_MAP[loaded_example["gold_label"]]
                #if loaded_example['genre'] not in types:
                #    types.append(loaded_example['genre'])
                data.append(loaded_example)
            random.seed(1)
            random.shuffle(data)
        #for i,t in enumerate(types):
        #    fw.write(f"{t}:{i}\n")
        return data


    def SplitDatasetTV(self,data,valid_size,
                       stratify = False,
                       shuffle=True,
                       save = True,
                       train_path = None,
                       valid_path=None):
        '''
        将原始数据集划分为valid和train部分
        :param self:
        :param x:
        :param y:
        :param valid_size: 验证集所占比例,0~1
        :param stratify:
        :param shuffle:
        :param save:
        :param train_path:
        :param valid_path:
        :return:
        '''
        self.logger.info('Split Dateset to train and valid')

        if stratify:
            num_classes = len(self.LABEL_MAP)
            trainData,validData = [],[]
            bucket = [ [] for _ in range(num_classes)]
            for d in tqdm.tqdm(data):
                bucket[self.LABEL_MAP[d['gold_label']]].append(d)
            del data
            for label in tqdm.tqdm(bucket,desc='split Proportionally'):
                n = len(label)
                if n==0:
                    continue
                validPart = int(n*valid_size)
                if shuffle:
                    random.shuffle(label)
                validData.extend(label[:validPart])
                trainData.extend(label[validPart:])
            if shuffle:
                random.shuffle(trainData)
        else:

            n = len(data)
            validPart = int(n*valid_size)
            if shuffle:
                random.shuffle(data)
            validData = data[:validPart]
            trainData = data[validPart:]
            if shuffle:
                random.shuffle(data)

        if save:
            self.text_write(path=train_path,data = trainData)
            self.text_write(path=valid_path,data = validData)

        return trainData,validData

    def text_write(self,path,data,encoding='utf-8'):
        '''

        :param path: the path of file
        :param data:  data with format (sentence,data)
        :param encoding:
        :return:
        '''
        with open(path,'w',encoding=encoding) as fw:
            for d in data:
                json.dump(d,fw)
                fw.write('\n')

class ontonotes5DT(DataTransformer):

    def __init__(self,logger,seed):
        super(ontonotes5DT,self).__init__(logger,seed)

    def read_data(self,path):
        l_words,l_ners = [],[]
        entries = open(path,'r',encoding=self.encoding).read().strip().split('\n\n')
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            ners = [line.split()[-1] for line in entry.splitlines()]
            l_words.append(['[CLS]']+words[:254]+['[SEP]'])
            l_ners.append(['<PAD>']+ners[:254]+['<PAD>'])
        return (l_words,l_ners)







