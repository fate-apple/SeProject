#encoding:utf-8
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import gc
import torch
from pytorch_pretrained_bert import BertTokenizer


class LabelExample(object):
    def __init__(self,eid,sentence,labels=None):
        '''
        Example for event detection(labelling)
        :param eid: the unique id of the example with format: type-num.
        :param label: the list of event label like [7,31]
        '''
        self.eid = eid
        self.sentence = sentence
        self.labels = labels
class LabelDataset(Dataset):
    def __init__(self,data,max_seq_len,tokenizer,seed,example_type):
        '''
        Dataset for event labeling
        :param example_type:
        '''
        self.data = data
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.seed = seed
        self.example_type = example_type
        self.examples = self.build_examples()
    def __len__(self):
        return len(self.examples)
    def __getitem__(self,index):
        return self.process(index)

    def process(self,index):
        example = self.examples[index]
        feature = self.build_features(example)
        input_ids,input_masks,segement_ids,label_ids = feature
        return np.array(input_ids),np.array(input_masks),np.array(label_ids)

    def build_feature(self,example):
        tokens = self.tokenizer.tokenize(example.sentence)
        if len(tokens) >self.max_seq_len -2:
            tokens = tokens[:self.max_seq_len-2]
        # add identifier in begin and end of sentence
        tokens = ['[CLS]']+tokens+['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_masks = [1]*len(input_ids)
        padding = [0]*(self.max_seq_len-len(input_ids))

        input_ids += padding
        input_masks +=padding
        labels = example.labels

        return (input_ids,input_masks,labels)


    def build_examples(self):
        '''
        init all examples
        :return:
        '''
        if isinstance(self.data,Path):
            lines = self.read_data(data_path = self.data)
        else :
            lines = self.data
        examples = []
        for i,line in enumerate(lines):
            eid = f'{self.example_type}-{i}'
            sentence  = line[0]
            labels = line[1]
            if isinstance(labels,str):
                labels = [np.float32(x) for x in labels.split(',')]
            else:
                labels = [np.float32(x) for x in list(labels)]
            #text_b = None
            example = LabelExample(eid = eid,sentence= sentence,labels=labels)
            self.examples.append(example)
        del lines,self.data
        gc.collect()
        return examples



class NerDataset(Dataset):
    def __init__(self,data,max_seq_len,ner2idx,seed=0,example_type='Unk',
                 tokenizer=BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)):

        self.l_words,self.l_ners = data
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.tokenizer = tokenizer
        self.example_type = example_type
        self.ner2idx = ner2idx
    def __len__(self):
        return len(self.l_words)
    def __getitem__(self, index):
        words,ners = self.l_words[index],self.l_ners[index]

        l_tokenids,l_nerids,l_input_masks = [],[],[]
        l_is_begin =[]
        for w,n in zip(words,ners):
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            #TODO :pad in each batch
            #Done!
            #if len(tokens) >self.max_seq_len :
            #    tokens = tokens[:self.max_seq_len-1]+['[SEP']
            tokenids = self.tokenizer.convert_tokens_to_ids(tokens)
            is_begin = [1] + [0]*(len(tokens)-1)
            #tags = [n]+['<PAD>']*(len(tokens)-1)
            tags = [n]*len(tokens)
            nerids = [self.ner2idx[tag] for tag in tags]
            input_masks = [1]*len(tokenids)

            l_tokenids.extend(tokenids)
            l_input_masks.extend(input_masks)
            l_nerids.extend(nerids)
            l_is_begin.extend(is_begin)

        assert len(l_tokenids)==len(l_nerids)==len(l_is_begin), \
            f"len(x)={len(l_tokenids)}, len(y)={len(l_nerids)}, len(is_heads)={len(l_is_begin)}"

        seq_len = len(l_tokenids)

        sentence = ' '.join(words)
        ners = ' '.join(ners)
        #bertmodel.forward need longtensor
        return l_tokenids,l_input_masks,l_nerids,l_is_begin,sentence,ners,seq_len

    def pad(batch):
        f = lambda x: [sample[x] for sample in batch]
        #input_ids = f(0)
        token_type_ids =None
        #attention_mask = f(1)
        #l_nerids = f(2)
        #l_is_begin =f(3)
        sentence = f(4)
        ners = f(5)
        seqlens = f(-1)

        max_seq_len = np.array(seqlens).max()

        #f = lambda  l,max_seq_len : [x+[0]*(max_seq_len-len(x))for x in l]
        f = lambda x, seqlen: [sample[x] + [0] * (max_seq_len - len(sample[x])) for sample in batch]
        g= torch.LongTensor
        input_ids = g(f(0,max_seq_len))
        attention_mask = g(f(1,max_seq_len))
        y = g(f(2,max_seq_len))
        l_is_begin =g(f(3,max_seq_len))
        #copy from BertModel.forward()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)


        return input_ids,token_type_ids,attention_mask,y,l_is_begin,sentence,ners


class MnliDataset(Dataset):
    def __init__(self,data,max_seq_len,label2idx,seed=0,example_type='Unk',
                 tokenizer=BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)):
        self.data = data
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.tokenizer = tokenizer
        self.example_type = example_type
        self.label2idx = label2idx
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        label,textA,textB  = self.data[index]['gold_label'],self.data[index]['sentence1'],self.data[index]['sentence2']

        input_ids,token_type_ids,attention_mask,y = [],[],[],[]
        tokensA = ['[CLS]'] + self.tokenizer.tokenize(textA)[:126] + ['[SEP]']
        tokensB = self.tokenizer.tokenize(textB)[:127]+ ['[SEP]']
        tokens =  tokensA +  tokensB
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        token_type_ids = [0] * len(tokensA)+[1]*len(tokensB)
        attention_mask = [1]*len(input_ids)
        #label_id = [0]*len(self.label2idx)
        #label_id[self.label2idx[label]] = 1
        label_id = self.label2idx[label]


        #.extend(label_id)
        y = label_id
        seq_len = len(input_ids)

        return input_ids,token_type_ids,attention_mask,y,seq_len

    def pad(batch):

        f = lambda x: [sample[x] for sample in batch]
        seqlens = f(-1)
        y = f(-2)
        max_seq_len = np.array(seqlens).max()

        #f = lambda  l,max_seq_len : [x+[0]*(max_seq_len-len(x))for x in l]
        f = lambda x, seqlen: [sample[x] + [0] * (max_seq_len - len(sample[x])) for sample in batch]
        g= torch.LongTensor
        input_ids = g(f(0,max_seq_len))
        token_type_ids = g(f(1,max_seq_len))
        attention_mask = g(f(2,max_seq_len))
        y = g(y)

        #copy from BertModel.forward()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)


        return input_ids,token_type_ids,attention_mask,y

class BIODataset(Dataset):
    def __init__(self,data,max_seq_len,seed=0,example_type='Unk',
                 tokenizer=BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)):
        self.l_words,self.l_ners = data
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.tokenizer = tokenizer
        self.example_type = example_type

    def __len__(self):
        return len(self.l_words)
    def __getitem__(self, index):
        words,ners = self.l_words[index],self.l_ners[index]
        input_ids,token_type_ids,attention_mask,l_is_begin = [],[],[],[]
        l_tags =[]
        for w,n in zip(words,ners):
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            tokenids = self.tokenizer.convert_tokens_to_ids(tokens)
            is_begin = [1] + [0]*(len(tokens)-1)
            tags = [n]*len(tokens)
            input_masks = [1]*len(tokenids)

            input_ids.extend(tokenids)
            attention_mask.extend(input_masks)
            l_is_begin.extend(is_begin)
            l_tags.extend(tags)
        token_type_ids = None
        seq_len = len(input_ids)

        pos_v = l_tags.index('V')
        pos_arg0 = l_tags.index('B-ARG0')
        try:
            pos_arg1 = l_tags.index('B-ARG1')
        except:
            print('debug')
        for j in range(pos_arg0+1,seq_len+1):
            if j<seq_len and l_tags[j] in ['B-ARG0','I-ARG0']:
                j+=1
                continue
            else :
                pos_arg0_end =  j
                break
        for j in range(pos_arg1+1,seq_len+1):
            if j<seq_len and l_tags[j] in ['B-ARG1','I-ARG1']:
                j+=1
                continue
            else :
                pos_arg1_end =  j
                break
        y = [pos_v,pos_arg0,pos_arg0_end,pos_arg1,pos_arg1_end]

        sentence = ' '.join(words)
        return input_ids,token_type_ids,attention_mask,y,l_is_begin,sentence,seq_len

    def pad(batch):
        f = lambda x: [sample[x] for sample in batch]
        seqlens = f(-1)
        y = f(3)
        max_seq_len = np.array(seqlens).max()

        f = lambda x, seqlen: [sample[x] + [0] * (max_seq_len - len(sample[x])) for sample in batch]
        g= torch.LongTensor
        input_ids = g(f(0,max_seq_len))
        #token_type_ids = g(f(1,max_seq_len))
        attention_mask = g(f(2,max_seq_len))
        l_is_begin = g(f(4,max_seq_len))
        y = g(y)
        token_type_ids = torch.zeros_like(input_ids)

        return (input_ids,token_type_ids,attention_mask,y,l_is_begin)













