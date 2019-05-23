#encoding: utf-8
'''
@time: 2019/5/19 21:15
@desc:
'''
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from Config.BasicConfig import Mnli_configs
import torch
from  pathlib import Path
from Pybert.utils.utils import ProgressBar,model_device,summary_info,load_bert
import numpy as np
import tqdm
from Pybert.train.metric import F1Score,mnli_simple_accuracy
from pytorch_pretrained_bert.tokenization import BertTokenizer
from Data.Dataset import *


class Predictor(object):
    def __init__(self,
                 model,
                 logger,
                 model_path,
                 config
                 ):
        self.model = model
        self.logger = logger
        self.width = 30
        self.config = config
        n_gpu = config['common']['n_gpu']
        self.model, self.device = model_device(n_gpu= n_gpu, model=self.model, logger=self.logger)
        if model_path:
            loads = load_bert(model_path=model_path,model = self.model)
            self.model = loads[0]
        self.result={}
        self.outputs,self.targets =[],[]
        self.tokenizer = BertTokenizer(vocab_file=config['model']['pretrained']['bert_vocab_path'],
                              do_lower_case=config['train']['do_lower_case'])
        
    def PreprocessText(self,textA,textB=None):
        input_ids,token_type_ids,attention_mask,y = [],[],[],[]

        if textB: #mnli
            debug_words = textB.split()
            tokens = ['[CLS]'] + self.tokenizer.tokenize(textA)[:254] + ['[SEP]']
            token_type_ids = [0] * len(tokens)
            '''
            if textB:
                tokensB = self.tokenizer.tokenize(textB)[:255]+ ['[SEP]']
                tokens += tokensB
                token_type_ids+=[1]*len(tokensB)
            '''
            tokensB = self.tokenizer.tokenize(textB)[:255]+ ['[SEP]']
            tokens += tokensB
            token_type_ids+=[1]*len(tokensB)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1]*len(input_ids)
            batch = (input_ids,token_type_ids,attention_mask)
            batch = tuple(torch.LongTensor(t).to(self.device).view(1,-1) for t in batch)
        else:
            words = ['[CLS]']+textA.split()[:254]+['[SEP]']
            input_ids,l_is_begin,attention_mask = [],[],[]
            for w in words:
                tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                tokenids = self.tokenizer.convert_tokens_to_ids(tokens)
                is_begin = [1] + [0]*(len(tokens)-1)
                input_masks = [1]*len(tokenids)

                input_ids.extend(tokenids)
                attention_mask.extend(input_masks)
                l_is_begin.extend(is_begin)

            input_ids,attention_mask,l_is_begin = tuple(torch.LongTensor(t).to(self.device).view(1,-1) for t in (input_ids,attention_mask,l_is_begin))
            token_type_ids = torch.zeros_like(input_ids)
            batch = (input_ids,token_type_ids,attention_mask,l_is_begin)
        return batch

class MnliPredictor(Predictor):
    def __init__(self,
                 model,
                 logger,
                 model_path,
                 config,
                 criterion=None
                 ):
        self.metrics = [F1Score(average='micro',task_type='multiclass',normalizate=False,only_head=False),
                        mnli_simple_accuracy()]
        self.criterion = criterion
        super(MnliPredictor,self).__init__(model,logger,model_path,config)

    def predict(self,textA,textB):
        self.model.eval()
        with torch.no_grad():
            batch =   self.PreprocessText(textA,textB)
            #batch = tuple(torch.LongTensor(t).to(self.device).view(1,-1) for t in batch)
            input_ids, token_type_ids, attention_mask= batch
            logits = self.model(input_ids, token_type_ids, attention_mask)
        return logits
    def show_info(self,batch_id,n_batch):
        recv_per = int(100 * (batch_id + 1) / n_batch)
        show_bar = f"\r[predict]{batch_id+1}/{n_batch}[{int(self.width * recv_per / 100) * '>':<{self.width}s}]{recv_per}%"
        print(show_bar,end='')

    def test(self,data):
        self.model.eval()
        n_batch = len(data)
        y_hat = []
        with torch.no_grad():
            for step, batch in enumerate(data):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, token_type_ids, attention_mask, y = batch
                logits = self.model(input_ids, token_type_ids, attention_mask)
                logits_sigm = logits.sigmoid()
                self.show_info(step,n_batch)
                self.outputs.append(logits.cpu().detach())
                y_hat.append(logits.argmax(-1).cpu().detach())
                self.targets.append(y.cpu().detach())
            self.outputs = torch.cat(self.outputs,dim=0).cpu().detach()
            self.targets = torch.cat(self.targets,dim=0).cpu().detach()
            y_hat = torch.cat(y_hat,dim=0).cpu().detach()
            loss = self.criterion(target= self.targets,logits = self.outputs)
            for metric in self.metrics:
                metric(logits=self.outputs,target = self.targets)
                value = metric.value()
                if value :
                    self.result[f"{metric.name()}"] = value
            self.result['loss'] = loss
            show_info = '\n'+'  -  '.join([f" {key} : {value:.4f} "for key,value in self.result.items()])
            self.logger.info(show_info)
        return loss

class NerPredictor(Predictor):
    def __init__(self,
                 model,
                 logger,
                 model_path,
                 config,
                 criterion=None
                 ):

        self.metrics = [F1Score(average='micro',task_type='multiclass',normalizate=False,only_head=True)]
        self.criterion = criterion
        super(NerPredictor,self).__init__(model,logger,model_path,config)
        self.ner2id = {n:i for i,n in enumerate(config['Ner'])}

    def predict(self,textA):
        self.model.eval()
        words = ['[CLP]']+textA.split()[:254]+['[SEP]']
        id2ner  = {i:n for i,n in enumerate(self.config['Ner'])}
        with torch.no_grad():
            batch =   self.PreprocessText(textA)
            #batch = tuple(t.to(self.device) for t in batch)
            input_ids, token_type_ids, attention_mask,is_heads = batch
            logits = self.model(input_ids, token_type_ids, attention_mask)
            y_hat = logits.argmax(-1).view(-1).tolist()
            is_heads = is_heads.view(-1).tolist()
            input_ids = input_ids.view(-1).tolist()

            #is_heads[0]  =0
            #is_heads[-1] = 0
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            ners= [id2ner[i] for i in y_hat]
            logits = logits.view(logits.size(1),logits.size(2)).softmax(-1).cpu().detach().numpy()
            pos = [i for i,_ in enumerate(input_ids) if is_heads[i]==1]
            logits = logits[pos]


            i =0
            n = len(ners)
            m = len(input_ids)
            pos += [m]
            output = {'PER':[],'LOC':[],'ORG':[],'MISC':[]}

            while(i<n):
                word = ''
                if ners[i]=='<PAD>' or  ners[i]=='O':
                    i+=1
                    continue
                elif ners[i] == 'B-PER':
                    for j in range(i+1,n+1):
                        if 'I' in ners[j]:
                            continue
                        else:
                            #word = ' '.join([''.join(self.tokenizer.convert_ids_to_tokens(input_ids[pos[a]:pos[a+1]]))
                            #                 for a in range(i,j)])
                            word = ' '.join([words[i] for i in range(i,j)])
                            i =j
                            output['PER'].append(word)
                            break
                elif ners[i] == 'B-LOC':
                    #word += self.tokenizer.convert_ids_to_tokens(input_ids[i])
                    for j in range(i+1,n):
                        #if ners[j]in['I-LOC','B-LOC']:
                        if 'I' in ners[j]:
                            continue
                        else:
                            word = ' '.join([words[i] for i in range(i,j)])
                            i =j
                            output['LOC'].append(word)
                            break
                elif ners[i] == 'B-MISC':
                    for j in range(i+1,n):
                        if 'I' in ners[j]:
                            continue
                        else:
                            word = ' '.join([words[i] for i in range(i,j)])
                            i =j
                            output['MISC'].append(word)
                            break
                elif ners[i] == 'B-ORG':
                    for j in range(i+1,n):
                        #if ners[j]in['I-ORG','B-ORG']:
                        if 'I' in ners[j]:
                            continue
                        else:
                            word = ' '.join([words[i] for i in range(i,j)])
                            i =j
                            output['ORG'].append(word)
                            break
                else:
                    i+=1

        return logits,output

    def show_info(self,batch_id,n_batch):
        recv_per = int(100 * (batch_id + 1) / n_batch)
        show_bar = f"\r[predict]{batch_id+1}/{n_batch}[{int(self.width * recv_per / 100) * '>':<{self.width}s}]{recv_per}%"
        print(show_bar,end='')

    def test(self,data):
        self.model.eval()
        n_batch = len(data)
        y_hat = []
        with torch.no_grad():
            for step, batch in enumerate(data):
                #batch = tuple(t.to(self.device) for t in batch)
                input_ids,token_type_ids,attention_mask,y,l_is_begin,sentence,ners = batch
                input_ids,token_type_ids,attention_mask,y = tuple(t.to(self.device) for t in (input_ids,token_type_ids,attention_mask,y))
                logits = self.model(input_ids, token_type_ids, attention_mask)
                y_hat = logits.argmax(-1)
                #----------------------------loss----------------------------
                logits = logits.view(-1, logits.shape[-1])
                y,y_hat,l_is_begin = y.view(-1),y_hat.view(-1),l_is_begin.view(-1)
                self.show_info(step,n_batch)
                self.outputs.append(logits.cpu().detach())
                self.targets.append(y.cpu().detach())

            self.outputs = torch.cat(self.outputs,dim=0).cpu().detach()
            self.targets = torch.cat(self.targets,dim=0).cpu().detach()
            loss = self.criterion(target= self.targets,logits = self.outputs)
            for metric in self.metrics:
                metric(logits=self.outputs,target = self.targets)
                value = metric.value()
                if value :
                    self.result[f"{metric.name()}"] = value
            self.result['loss'] = loss
            show_info = '\n'+'  -  '.join([f" {key} : {value:.4f} "for key,value in self.result.items()])
            self.logger.info(show_info)
        return loss

class BIOPredictor(Predictor):
    def __init__(self,
                 model,
                 logger,
                 model_path,
                 config,
                 criterion=None
                 ):

        self.metrics = [F1Score(average='micro',task_type='multiclass',normalizate=False,only_head=True),
                        mnli_simple_accuracy()]
        self.criterion = criterion
        self.threshold = 0.1
        super(BIOPredictor,self).__init__(model,logger,model_path,config)

    def predict(self,textA):
        self.model.eval()
        words = ['[CLP]']+textA.split()[:254]+['[SEP]']
        id_ner  = {i:n for i,n in enumerate(self.config['Ner'])}
        ner2id = {n:i for i,n in enumerate(self.config['Ner'])}
        with torch.no_grad():
            batch =   self.PreprocessText(textA)
            #batch = tuple(t.to(self.device) for t in batch)
            input_ids, token_type_ids, attention_mask,is_heads = batch
            logits = self.model(input_ids, token_type_ids, attention_mask)
            y_hat = logits.argmax(-1).view(-1).tolist()
            is_heads = is_heads.view(-1).tolist()
            input_ids = input_ids.view(-1).tolist()

            #is_heads[0]=0
            #is_heads[-1]=0
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            ners= [id_ner[i] for i in y_hat]
            logits = logits.view(logits.size(1),logits.size(2)).softmax(-1).cpu().detach().numpy()
            pos = [i for i,_ in enumerate(input_ids) if is_heads[i]==1]


            logits = logits[pos]
            r,c = logits.shape
            pos_v,pos_arg0,pos_arg1,pos_arg2 = -1,-1,-1,-1
            output = {'V':'','ARG0':'','ARG1':'','ARG2':''}
            des = ''

            pos = pos+[len(logits)]
            n = len(pos)
            '''
            if np.max(logits[...,ner2id['V']]) >0:
                pos_v = logits[pos[:-1],ner2id['V']].argmax()
            if pos_v>0:
                #n= len(pos)
                for j in range(pos_v+1,n):
                        if j<n-1 and ners[j] in ['V']:
                            continue
                        else:
                            word = ' '.join([words[i] for i in range(pos_v,j)])
                            debug_l  = range(ori_pos[pos_v],ori_pos[j])
                            pos = [i for i in pos if i not in debug_l]
                            output['V']=  word
                            break
            if np.max(logits[...,ner2id['B-ARG0']]) >0:
                debug_logit = logits[pos[:-1],ner2id['B-ARG0']]
                pos_arg0 = debug_logit.argmax()
            if pos_arg0>0:
                #n= len(pos)
                for j in range(pos_arg0+1,n):
                        if j<n-1 and ners[j] in ['I-ARG0']:
                            continue
                        else:
                            #word = ' '.join([''.join(self.tokenizer.convert_ids_to_tokens(input_ids[ori_pos[a]:ori_pos[a+1]]))
                            #                for a in range(pos_arg0,j)])
                            word = ' '.join([words[i] for i in range(pos_arg0,j)])
                            debug_l  = range(ori_pos[pos_v],ori_pos[j])
                            pos = [i for i in pos if i not in range(ori_pos[pos_arg0],ori_pos[j])]
                            output['ARG0']=  word
                            break
            if np.max(logits[...,ner2id['B-ARG1']]) >0:
                pos_arg1 = logits[pos[:-1],ner2id['B-ARG1']].argmax()
            if pos_arg1>0:
                #n= len(pos)
                for j in range(pos_arg1+1,n):
                        if j<n-1 and  ners[j] in ['I-ARG1']:
                            continue
                        else:
                            #word = ' '.join([''.join(self.tokenizer.convert_ids_to_tokens(input_ids[ori_pos[a]:ori_pos[a+1]]))
                            #                 for a in range(pos_arg1,j)])
                            word = ' '.join([words[i] for i in range(pos_arg1,j)])
                            pos = [i for i in pos if i not in range(ori_pos[pos_arg1],ori_pos[j])]
                            output['ARG1']=  word
                            break
            if np.max(logits[...,ner2id['B-ARG2']]) >self.threshold:
                pos_arg2 = logits[pos[:-1],ner2id['B-ARG2']].argmax()
            if pos_arg2>0:
                #n= len(pos)
                for j in range(pos_arg2+1,n):
                        if j<n-1 and ners[j] in ['I-ARG2']:
                            continue
                        else:
                            word = ' '.join([words[i] for i in range(pos_arg2,j)])
                            pos = [i for i in pos if i not in range(ori_pos[pos_arg2],ori_pos[j])]
                            logits = logits[pos]
                            output['ARG2']=  word
                            break
            '''
            ignore = []
            pos_v = logits[...,ner2id['V']].argmax()
            ignore.append(pos_v)
            debug_l  =[i for i in range(r)if i not in ignore]
            pos_arg0 = logits[debug_l,ner2id['B-ARG0']].argmax()
            ignore.append(pos_arg0)
            debug_l  =[i for i in range(r)if i not in ignore]
            pos_arg1 = logits[debug_l,ner2id['B-ARG1']].argmax()
            if pos_v>0:
                for j in range(pos_v+1,n):
                        if j<n-1 and ners[j] in ['V']:
                            continue
                        else:
                            word = ' '.join([words[i] for i in range(pos_v,j)])
                            words.pop(pos_v)
                            output['V']=  word
                            break
            if pos_arg0>=0:
                for j in range(pos_arg0+1,n):
                        if j<n-1 and ners[j] in ['I-ARG0']:
                            continue
                        else:
                            word = ' '.join([words[i] for i in range(pos_arg0,j)])
                            words.pop(pos_arg0)
                            output['ARG0']=  word
                            break
            if pos_arg1>=0:
                for j in range(pos_arg1+1,n):
                        if j<n-1 and  ners[j] in ['I-ARG1']:
                            continue
                        else:
                            word = ' '.join([words[i] for i in range(pos_arg1,j)])
                            output['ARG1']=  word
                            break
        des = ' '.join([output['ARG0'],output['V'],output['ARG1'],output['ARG2']])
        #print(des)
        return logits,output,des

    def show_info(self,batch_id,n_batch):
        recv_per = int(100 * (batch_id + 1) / n_batch)
        show_bar = f"\r[predict]{batch_id+1}/{n_batch}[{int(self.width * recv_per / 100) * '>':<{self.width}s}]{recv_per}%"
        print(show_bar,end='')

    def test(self,data):
        self.model.eval()
        n_batch = len(data)
        y_hat = []
        with torch.no_grad():
            for step, batch in enumerate(data):
                #batch = tuple(t.to(self.device) for t in batch)
                input_ids,token_type_ids,attention_mask,y,l_is_begin,sentence,ners = batch
                input_ids,token_type_ids,attention_mask,y = tuple(t.to(self.device) for t in (input_ids,token_type_ids,attention_mask,y))
                logits = self.model(input_ids, token_type_ids, attention_mask)

                y_hat = logits.argmax(-1)
                #logits = logits[:,1:-1,...].view(-1, logits.shape[-1])
                #y,y_hat,l_is_begin = y[:,1:-1,...].view(-1),y_hat[:,1:-1,...].view(-1),l_is_begin[:,1:-1,...].view(-1)
                logits = logits.view(-1, logits.shape[-1])
                y,y_hat,l_is_begin = y.view(-1),y_hat.view(-1),l_is_begin.view(-1)

                self.show_info(step,n_batch)
                self.outputs.append(logits.cpu().detach())
                #y_hat.append(y_hat.cpu().detach())
                self.targets.append(y.cpu().detach())
            self.outputs = torch.cat(self.outputs,dim=0).cpu().detach()
            self.targets = torch.cat(self.targets,dim=0).cpu().detach()
            loss = self.criterion(target= self.targets,logits = self.outputs)
            for metric in self.metrics:
                metric(logits=self.outputs,target = self.targets)
                value = metric.value()
                if value :
                    self.result[f"{metric.name()}"] = value
            self.result['loss'] = loss
            show_info = '\n'+'  -  '.join([f" {key} : {value:.4f} "for key,value in self.result.items()])
            self.logger.info(show_info)
        return loss


