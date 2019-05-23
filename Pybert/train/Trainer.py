import time
import torch
from Pybert.utils.utils import ProgressBar,model_device,summary_info
from Pybert.callback.model_checkpoint import  restore_checkpoint
import tqdm
class Trainer(object):
    def __init__(self,train_configs):
        self.start_epoch = 1
        self.global_step = 1
        #self.empty_cache=train_configs['empty_cache']
        self.n_gpu = train_configs['n_gpu']
        self.model = train_configs['model']
        self.epochs = train_configs['epochs']
        self.logger = train_configs['logger']
        self.verbose = train_configs['verbose']
        self.criterion = train_configs['criterion']
        self.optimizer = train_configs['optimizer']
        self.lr_scheduler = train_configs['lr_scheduler']
        self.early_stop = train_configs['early_stop']
        self.epoch_metrics = train_configs['epoch_metrics']
        self.batch_metrics = train_configs['batch_metrics']
        self.model_checkpoint = train_configs['model_checkpoint']
        self.training_monitor = train_configs['training_monitor']
        self.gradient_accumulation_steps = train_configs['gradient_accumulation_steps']

        self.model ,self.device = model_device(n_gpu = self.n_gpu,model = self.model,logger = self.logger)
        if train_configs['resume']:
            self.logger.info(f"\nloading checkpoint: {train_configs['resume']}")
            resume_list = restore_checkpoint(resume_path= train_configs['resume'],model = self.model,optimizer = self.optimizer)
            self.model = resume_list[0]
            self.optimizer = resume_list[1]
            self.start_epoch = resume_list[2]
            best = resume_list[3]
            if self.model_checkpoint:
                self.model_checkpoint.best = best
            self.logger.info(f'\nresume checkpoint : {train_configs["resume"]}')
        self.outputs = []
        self.targets = []
    def epoch_reset(self):
        self.outputs = []
        self.targets = []
        self.is_head = []
        self.result = {}
        for metric in self.epoch_metrics:
            metric.reset()

    def batch_reset(self):
        self.info = {}
        for metric in self.batch_metrics:
            metric.reset()
class LabelTrainer(Trainer):
    def __init__(self):
        super(LabelTrainer,self).__init__()

    def _train_epoch(self,data):
        self.epoch_reset()
        self.model.train()
        for step,(input_ids,input_mask,label_ids) in enumerate(data):
            start = time.time()
            self.batch_reset()
            input_ids.input_mask,label_ids= input_ids.to(self.device),input_mask.to(self.device),label_ids.to(self.device)
            logits = self.model(input_ids,input_mask)
            #----------------------------loss----------------------------
            loss = self.criterion(logits = logits,target=label_ids) #BCEWithLogitsLoss
            if(len(self.n_gpu)>1):
                loss = loss.mean()
            if self.gradient_accumulation_steps>1:
                loss = loss/self.gradient_accumulation_steps
            loss.backward()
            if (step+1) % self.gradient_accumulation_steps==0:
                self.lr_scheduler.batch_step(training_step = self.global_step)
                self.optimizer.step()   #BertAdam
                self.optimizer.zero_grad()
                self.global_step +=1
            if self.batch_metrics:  #AccuracyThresh
                for metric in self.batch_metrics:
                    metric(logits=logits,target=label_ids)
                    self.info[metric.name()] = metric.value()
            self.info['loss'] = loss.item()
            if self.verbose>0:
                self.progressbar.step(index=step,info=self.info,use_time=time.time()-start)
            self.outputs.append(logits.cpu().detach())
            self.targets.append(label_ids.cpu().detach())
        print(f"{'-'*25} train result : {'-'*25}")
        self.outputs = torch.cat(self.outputs,dim=0).cpu().detach()
        self.targets = torch.cat(self.targets,dim=0).cpu().detach()
        loss = self.criterion(target= self.targets,logits = self.outputs)
        self.result['loss'] = loss.item()

        if self.epoch_metrics:  #F1Score
            for metric in self.epoh_metrics:
                metric(logits=self.outputs,target = self.targets)
                value = metric.value()
                if value :
                    self.result[f"{metric.name()}"] = value
        if len(self.n_gpu)>0 :
            torch.cuda.empty_cache()
        return self.result

    def _valid_epoh(self,valid_data):
        self.epoch_reset()
        self.model.eval()
        with torch.no_grad():
            for step,(input_ids,input_mask,labels_ids) in enumerate(valid_data):
                input_ids,input_mask,labels_ids = input_ids.to(self.device),input_mask.to(self.device),labels_ids.to(self.device)
                logits = self.model(input_ids= input_ids,attention_mask = input_mask,labels_ids = labels_ids)
                self.outputs.append(logits.cpu().detach())
                self.targets.append(labels_ids.cpu().detach())

            self.outputs = torch.cat(self.outputs,dim=0).cpu().detach()
            self.targets  = torch.cat(self.targets,dim=0).cpu().detach()
            loss = self.criterion(target = self.targets,logits=self.outputs)
            self.result['valid_loss'] = loss.item()
            print(f"{'-'*25} valid result {'-'*25}")
            if self.epoch_metrics:
                for metric in self.epoch_metrics:
                    metric(logits = self.outputs,target = self.targets)
                    value = metric.value()
                    if value:
                        self.result[f"valid_{metric.name()}"] = value
            if len(self.n_gpu)>0:
                torch.cuda.empty_cache()
            return self.result

    def _save_info(self,current_epoch,valid_loss):
        state = {
            'epoch': current_epoch,
            'arch': self.model_checkpoint.arch,
            'state_dict':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'valid_loss':round(valid_loss,4)
        }
        return state


    def train(self,train_data,valid_data):
        self.total  = len(train_data)
        self.progressbar = ProgressBar(total=self.total)

        print('model summary info:  ')
        input_ids,input_masks,label_ids = train_data[0]
        # np.array before
        input_ids = input_ids.to(self.device)
        input_masks = input_masks.to(self.device)
        summary_info(self.model,*(input_ids,input_masks),show_input=  True)

        for epoch in range(self.start_epoch,self.start_epoch+self.epochs):
            print(f"{'-'*25} Epoch {epoch}/{self.epochs} {'-'*25}")
            train_log = self._train_epoh(train_data)
            valid_log = self._valid_epoh(valid_data)

            logs = dict(train_log,**valid_log)
            show_info = f"\nEpoch ; {epoch} - "+' - '.join([f" {key} : {value:.4f} "for key,value in logs.items()])
            self.logger.info(show_info)
            print(f"{'-'*25}")

            if self.training_monitor:
                self.training_monitor.epoch_step(logs)
            if self.model_checkpoint:
                state = self._save_info(current_epoch=epoch,valid_loss=logs['valid_loss'])
                self.model_checkpoint.epoh_step(current = logs[self.model_checkpoint.target],state = state)
            if self.early_stopping:
                self.early_stopping.epoch_step(current_epoch = epoch,current = logs[self.early_stopping.target])
                if self.early_stopping.should_stop:
                    break;

class NerTrainer(Trainer):
    def __init__(self,train_configs):
        super(NerTrainer,self).__init__(train_configs)

    def _train_epoch(self,data):
        self.epoch_reset()
        self.model.train()
        for step,batch in enumerate(data):
            start = time.time()
            self.batch_reset()
            input_ids,token_type_ids,attention_mask,y,is_heads,sentence,ners = batch
            input_ids,token_type_ids,attention_mask,y = \
                input_ids.to(self.device),token_type_ids.to(self.device),attention_mask.to(self.device),y.to(self.device)
            logits= self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,y=y)
            y_hat = []
            for logit in logits:
                y_hat.append(logit.argmax(-1))
            #----------------------------loss----------------------------
            logits = logits.view(-1, logits.shape[-1])
            y,y_hat,is_heads = y.view(-1),y_hat.view(-1),is_heads.view(-1)
            loss = self.criterion(logits = logits,
                                  target=y) #EntryCross
            if(len(self.n_gpu)>1):
                loss = loss.mean()
            if self.gradient_accumulation_steps>1:
                loss = loss/self.gradient_accumulation_steps
            loss.backward()
            if (step+1) % self.gradient_accumulation_steps==0:
                self.lr_scheduler.batch_step(training_step = self.global_step)
                self.optimizer.step()   #BertAdam
                self.optimizer.zero_grad()
                self.global_step +=1

            if self.batch_metrics:  #AccuracyThresh
                for metric in self.batch_metrics:
                    metric(logits=y_hat,target=y)
                    self.info[metric.name()] = metric.value()
            self.info['loss'] = loss.item()
            if self.verbose>0:
                self.progressbar.step(index=step,info=self.info,use_time=time.time()-start)
            self.outputs.append(logits.cpu().detach())
            self.targets.append(y.cpu().detach())
            self.is_heads.append(is_heads.cpu().detach())

        print(f"\n{'-'*25} train result : {'-'*25}")
        self.outputs = torch.cat(self.outputs,dim=0).cpu().detach()
        self.targets = torch.cat(self.targets,dim=0).cpu().detach()
        self.is_heads = torch.cat(self.is_head,dim=0).cpu().detach()
        loss = self.criterion(target= self.targets,logits = self.outputs)
        self.result['loss'] = loss.item()

        if self.epoch_metrics:  #F1Score
            for i,metric in enumerate(self.epoch_metrics):
                metric(logits=self.outputs,target = self.targets,is_head = self.is_heads)
                value = metric.value()
                if value :
                    self.result[f"valid_{metric.name()}"+f"{i}"] = value
        if len(self.n_gpu)>0 :
            torch.cuda.empty_cache()
        return self.result

    def _valid_epoch(self,valid_data):
        self.epoch_reset()
        self.model.eval()
        with torch.no_grad():
            for step,batch in enumerate(valid_data):
                input_ids,token_type_ids,attention_mask,y,is_heads,sentence,ners = batch
                input_ids,token_type_ids,attention_mask,y = \
                    input_ids.to(self.device),token_type_ids.to(self.device),attention_mask.to(self.device),y.to(self.device)
                logits= self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
                y_hat = logits.argmax(-1)
                logits = logits.view(-1, logits.shape[-1])
                y,y_hat,is_heads = y.view(-1),y_hat.view(-1),is_heads.view(-1)
                self.outputs.append(logits.cpu().detach())
                self.targets.append(y.cpu().detach())
                self.is_heads.append(is_heads.cpu().detach())

            self.outputs = torch.cat(self.outputs,dim=0).cpu().detach()
            self.targets  = torch.cat(self.targets,dim=0).cpu().detach()
            self.is_heads  = torch.cat(self.is_heads,dim=0).cpu().detach()
            loss = self.criterion(target = self.targets,logits=self.outputs)
            self.result['valid_loss'] = loss.item()
            print(f"\n{'-'*25} valid result {'-'*25}")
            if self.epoch_metrics:
                for i,metric in enumerate(self.epoch_metrics):
                    metric(logits = self.outputs,target = self.targets,is_head = self.is_heads)
                    value = metric.value()
                    if value:
                        self.result[f"valid_{metric.name()}"+f"{i}"] = value
            if len(self.n_gpu)>0:
                torch.cuda.empty_cache()
            return self.result

    def _save_info(self,current_epoch,valid_loss):
        state = {
            'epoch': current_epoch,
            'arch': self.model_checkpoint.arch,
            'model_state':self.model.state_dict(),
            'optimizer_state':self.optimizer.state_dict(),
            'valid_loss':round(valid_loss,4)
        }
        return state


    def train(self,train_data,valid_data):
        self.total  = len(train_data)
        self.progressbar = ProgressBar(total=self.total)

        print('model summary info:  ')
        for i,batch in enumerate(train_data):
            input_ids,token_type_ids,attention_mask,y,l_is_begin,sentence,ners = batch
            input_ids,token_type_ids,attention_mask,y = \
                input_ids.to(self.device),token_type_ids.to(self.device),attention_mask.to(self.device),y.to(self.device)
            summary_info(self.model,
                         *(input_ids,token_type_ids,attention_mask),show_input=  True)
            break

        for epoch in range(self.start_epoch,self.start_epoch+self.epochs):
            print(f"{'-'*25} Epoch {epoch}/{self.epochs} {'-'*25}")

            train_log = self._train_epoch(train_data)
            valid_log = self._valid_epoch(valid_data)


            logs = dict(train_log,**valid_log)
            show_info = f"\nEpoch ; {epoch} - "+' - '.join([f" {key} : {value:.4f} "for key,value in logs.items()])
            self.logger.info(show_info)
            print(f"{'-'*25}")

            if self.training_monitor:
                self.training_monitor.epoch_step(logs)
            if self.model_checkpoint:
                state = self._save_info(current_epoch=epoch,valid_loss=logs['valid_loss'])
                self.model_checkpoint.epoch_step(current = logs[self.model_checkpoint.target],state = state)
            if self.early_stop:
                self.early_stop.epoch_step(current_epoch = epoch,current = logs[self.early_stopping.target])
                if self.early_stop.should_stop:
                    break;



class MnliTrainer(Trainer):
    def __init__(self,train_configs):
        super(MnliTrainer,self).__init__(train_configs)

    def _train_epoch(self,data):
        self.epoch_reset()
        self.model.train()
        for step,(input_ids,token_type_ids,attention_mask,y) in enumerate(data):
            start = time.time()
            self.batch_reset()
            input_ids,token_type_ids,attention_mask,y= \
                input_ids.to(self.device),token_type_ids.to(self.device),attention_mask.to(self.device),y.to(self.device)
            logits = self.model(input_ids= input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
            y_hat = logits.argmax(-1)
            #----------------------------loss----------------------------
            logits = logits.view(-1,3)
            y,y_hat = y.view(-1),y_hat.view(-1)
            loss = self.criterion(logits = logits,target=y)
            if(len(self.n_gpu)>1):
                loss = loss.mean()
            if self.gradient_accumulation_steps>1:
                loss = loss/self.gradient_accumulation_steps
            loss.backward()
            if (step+1) % self.gradient_accumulation_steps==0:
                #self.lr_scheduler.batch_step(training_step = self.global_step)
                self.optimizer.step()   #BertAdam
                self.optimizer.zero_grad()
                self.global_step +=1
            if self.batch_metrics:  #AccuracyThresh
                for metric in self.batch_metrics:
                    metric(logits=logits,target=y)
                    self.info[metric.name()] = metric.value()
            self.info['loss'] = loss.item()
            if self.verbose>0:
                self.progressbar.step(index=step,info=self.info,use_time=time.time()-start)
            self.outputs.append(logits.cpu().detach())
            self.targets.append(y.cpu().detach())
        print(f"{'-'*25} train result : {'-'*25}")
        self.outputs = torch.cat(self.outputs,dim=0).cpu().detach()
        self.targets = torch.cat(self.targets,dim=0).cpu().detach()
        loss = self.criterion(target= self.targets,logits = self.outputs)
        self.result['loss'] = loss.item()

        if self.epoch_metrics:  #F1Score
            for metric in self.epoch_metrics:
                metric(logits=self.outputs,target = self.targets)
                value = metric.value()
                if value :
                    self.result[f"{metric.name()}"] = value
        if len(self.n_gpu)>0:
            torch.cuda.empty_cache()
        return self.result

    def _valid_epoch(self,valid_data):
        self.epoch_reset()
        self.model.eval()
        with torch.no_grad():
            for step,(input_ids,token_type_ids,attention_mask,y) in enumerate(valid_data):
                input_ids,token_type_ids,attention_mask,y= \
                input_ids.to(self.device),token_type_ids.to(self.device),attention_mask.to(self.device),y.to(self.device)
                logits = self.model(input_ids= input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
                self.outputs.append(logits.cpu().detach())
                self.targets.append(y.cpu().detach())

            self.outputs = torch.cat(self.outputs,dim=0).cpu().detach()
            self.targets  = torch.cat(self.targets,dim=0).cpu().detach()
            loss = self.criterion(target = self.targets,logits=self.outputs)
            self.result['valid_loss'] = loss.item()
            print(f"{'-'*25} valid result {'-'*25}")
            if self.epoch_metrics:
                for metric in self.epoch_metrics:
                    metric(logits = self.outputs,target = self.targets)
                    value = metric.value()
                    if value:
                        self.result[f"valid_{metric.name()}"] = value
            if len(self.n_gpu)>0:
                torch.cuda.empty_cache()
            return self.result

    def _save_info(self,current_epoch,valid_loss):
        state = {
            'epoch': current_epoch,
            'arch': self.model_checkpoint.arch,
            'state_dict':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'valid_loss':round(valid_loss,4)
        }
        return state


    def train(self,train_data,valid_data):
        self.total  = len(train_data)
        self.progressbar = ProgressBar(total=self.total)

        print('model summary info:  ')
        for i,batch in enumerate(train_data):
            input_ids,token_type_ids,attention_mask,y= batch
            input_ids,token_type_ids,attention_mask,y = \
                input_ids.to(self.device),token_type_ids.to(self.device),attention_mask.to(self.device),y.to(self.device)
            summary_info(self.model,
                         *(input_ids,token_type_ids,attention_mask),show_input=  True)
            break


        for epoch in range(self.start_epoch,self.start_epoch+self.epochs):
            print(f"{'-'*25} Epoch {epoch}/{self.epochs} {'-'*25}")
            train_log = self._train_epoch(train_data)
            valid_log = self._valid_epoch(valid_data)

            logs = dict(train_log,**valid_log)
            show_info = f"\nEpoch ; {epoch} - "+' - '.join([f" {key} : {value:.4f} "for key,value in logs.items()])
            self.logger.info(show_info)
            print(f"{'-'*25}")

            if self.training_monitor:
                self.training_monitor.epoch_step(logs)
            if self.model_checkpoint:
                state = self._save_info(current_epoch=epoch,valid_loss=logs['valid_loss'])
                self.model_checkpoint.epoch_step(current = logs[self.model_checkpoint.target],state = state)
            if self.early_stop:
                self.early_stop.epoch_step(current_epoch = epoch,current = logs[self.early_stopping.target])
                if self.early_stop.should_stop:
                    break;


class BioTrainer(Trainer):
    def __init__(self,train_configs):
        super(BioTrainer,self).__init__(train_configs)
        for i in range(5):
            self.targets.append([])
            self.outputs.append([])

    def epoch_reset(self):
        self.outputs = []
        self.targets = []
        for i in range(5):
            self.targets.append([])
            self.outputs.append([])
        self.is_head = []
        self.result = {}
        for metric in self.epoch_metrics:
            metric.reset()

    def _train_epoch(self,data):
        self.epoch_reset()
        self.model.train()
        total_loss = [0]*5
        valid_loss =0
        for step,batch in tqdm.tqdm(enumerate(data)):
            start = time.time()
            self.batch_reset()
            input_ids,token_type_ids,attention_mask,y,is_heads= batch
            input_ids,token_type_ids,attention_mask,y,is_heads = \
                input_ids.to(self.device),token_type_ids.to(self.device),attention_mask.to(self.device),y.to(self.device),is_heads.to(self.device)
            logits= self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,y=y)

            #----------------------------loss----------------------------

            for i in range(5):
                logit = logits[i]
                target = y[...,i]
                if i==0:
                    loss = self.criterion(logits = logit,
                                  target=target)
                else :
                    loss += self.criterion(logits = logit,
                                  target=target)
                if step==0:
                    total_loss[i] = loss
                else:
                    total_loss[i]+= loss
            loss = loss/len(y)
            if(len(self.n_gpu)>1):
                loss = loss.mean()
            if self.gradient_accumulation_steps>1:
                loss = loss/self.gradient_accumulation_steps
            loss.backward()
            if (step+1) % self.gradient_accumulation_steps==0:
                self.lr_scheduler.batch_step(training_step = self.global_step)
                self.optimizer.step()   #BertAdam
                self.optimizer.zero_grad()
                self.global_step +=1

            self.info['loss'] = loss.item()
            if step==0:
                    valid_loss = loss
            else:
                    valid_loss += loss
            if self.verbose>0:
                self.progressbar.step(index=step,info=self.info,use_time=time.time()-start)
            #self.is_head.append(is_heads.view(-1).cpu().detach())

        print(f"\n{'-'*25} train result : {'-'*25}")



        for i in range(5):
            loss = total_loss[i]
            self.result[f'loss{i}'] = loss.item()

        valid_loss = valid_loss/(step+1)
        self.result['train_mean_loss'] = valid_loss.item()
        if self.epoch_metrics:  #F1Score
            for i,metric in enumerate(self.epoch_metrics):
                metric(logits=self.outputs,target = self.targets,is_head = self.is_head)
                value = metric.value()
                if value :
                    self.result[f"valid_{metric.name()}"+f"{i}"] = value
        if len(self.n_gpu)>0 :
            torch.cuda.empty_cache()
        return self.result

    def _valid_epoch(self,valid_data):
        self.epoch_reset()
        self.model.eval()
        total_loss = [0]*5
        with torch.no_grad():
            for step,batch in tqdm.tqdm(enumerate(valid_data)):
                start = time.time()
                self.batch_reset()
                input_ids,token_type_ids,attention_mask,y,is_heads= batch
                input_ids,token_type_ids,attention_mask,y,is_heads = \
                input_ids.to(self.device),token_type_ids.to(self.device),attention_mask.to(self.device),y.to(self.device),is_heads.to(self.device)
                logits= self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,y=y)
                #----------------------------loss----------------------------
                is_heads = is_heads.view(-1)


                for i in range(5):
                    logit = logits[i]
                    target = y[...,i]
                    loss = self.criterion(logits = logit,
                                      target=target)
                    if step==0:
                        total_loss[i] = loss
                    else:
                        total_loss[i]+= loss

            print(f"\n{'-'*25} train result : {'-'*25}")
            valid_loss =0
            for i in range(5):
                loss = total_loss[i]/(step+1)
                self.result[f'loss{i}'] = loss.item()
                if i==0 :
                    valid_loss=loss
                else :
                    valid_loss+=loss
            valid_loss/=5

            self.result['valid_loss'] = valid_loss.item()

            print(f"\n{'-'*25} valid result {'-'*25}")
            if self.epoch_metrics:
                for i,metric in enumerate(self.epoch_metrics):
                    metric(logits = self.outputs,target = self.targets,is_head = self.is_head)
                    value = metric.value()
                    if value:
                        self.result[f"valid_{metric.name()}"+f"{i}"] = value
            if len(self.n_gpu)>0:
                torch.cuda.empty_cache()
            return self.result

    def _save_info(self,current_epoch,valid_loss):
        state = {
            'epoch': current_epoch,
            'arch': self.model_checkpoint.arch,
            'model_state':self.model.state_dict(),
            'optimizer_state':self.optimizer.state_dict(),
            'valid_loss':round(valid_loss,4)
        }
        return state


    def train(self,train_data,valid_data):
        self.total  = len(train_data)
        self.progressbar = ProgressBar(total=self.total)

        print('model summary info:  ')
        for i,batch in enumerate(train_data):
            input_ids,token_type_ids,attention_mask,y,is_heads = batch
            input_ids,token_type_ids,attention_mask,y = \
                input_ids.to(self.device),token_type_ids.to(self.device),attention_mask.to(self.device),is_heads.to(self.device)
            summary_info(self.model,
                         *(input_ids,token_type_ids,attention_mask,y),show_input=  True)
            break

        for epoch in range(self.start_epoch,self.start_epoch+self.epochs):
            print(f"{'-'*25} Epoch {epoch}/{self.epochs} {'-'*25}")

            train_log = self._train_epoch(train_data)
            valid_log = self._valid_epoch(valid_data)


            logs = dict(train_log,**valid_log)
            show_info = f"\nEpoch ; {epoch} - "+' - '.join([f" {key} : {value:.4f} "for key,value in logs.items()])
            self.logger.info(show_info)
            print(f"{'-'*25}")

            if self.training_monitor:
                self.training_monitor.epoch_step(logs)
            if self.model_checkpoint:
                state = self._save_info(current_epoch=epoch,valid_loss=logs['valid_loss'])
                self.model_checkpoint.epoch_step(current = logs[self.model_checkpoint.target],state = state)
            if self.early_stop:
                self.early_stop.epoch_step(current_epoch = epoch,current = logs[self.early_stopping.target])
                if self.early_stop.should_stop:
                    break;










