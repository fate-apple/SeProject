#encoding:utf-8
import torch
import os
import warnings
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
#from torch.nn import BCEWithLogitsLoss

from Config.BasicConfig import label_configs as config
from Data.PrepareDL import PreLabelDL,PreNerDL
from Pybert.utils.logger import init_logger
from Pybert.utils.utils import seed_everything
from Pybert.model.nn.bert_finetune import Bert_Finetune
from Pybert.callback.model_checkpoint import Model_Checkpoint
from Pybert.callback.training_monitor import TrainingMonitor
from Pybert.callback.lr_scheduler import BertLR
from Pybert.train.metric import  F1Score,MultiLabelReport,AccuracyThresh
from Pybert.train.Trainer import LabelTrainer
from Pybert.train.losses import BCEWithLogitsLoss
warnings.filterwarnings("ignore")

def main():
    print(os.path.pardir)
    os.chdir(os.path.pardir)
    device = f"cuda: {config['common']['n_gpu'][0] if len(config['common']['n_gpu'])!=0 else 'cpu'}"
    #----------------logger----------------
    logger = init_logger(log_name = config['model']['arch'],log_dir = config['output']['log_dir'])
    logger.info(f'device    :   {device}')
    logger.info(f"seed is {config['common']['seed']}")
    seed_everything(seed = config['common']['seed'],device=device)
    logger.info('load data from disk')

    #----------------Prepare Data----------------
    label_train_loader ,label_valid_loader  = PreLabelDL()
    #ner_train_loader,ner_valid_loader = PreNerDL()

    #----------------Model----------------
    logger.info('Initializing Model')
    id2label = {index:label for index,label in enumerate(config['label'])}
    model = Bert_Finetune.from_pretrained(model_dir =config['model']['pretrained']['bert_model_dir'],
                                          cache_dir = config['output']['cache_dir'],
                                          num_classes = len(id2label))

    #----------------Optimizer----------------
    param_optimizer = list(model.named_parameters())#Returns an iterator over module parameters, yielding both the  name of the parameter as well as the parameter itself.
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_grouped_parameters =[
        {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.01},
        {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0}
    ]
    trainsteps_num = int(   len(label_train_loader) / config['train']['gradient_accumulation_steps']# Number of updates steps to accumulate before performing a backward/update pass.
                            * config['train']['epochs'])
    optimizer  = BertAdam(optimizer_grouped_parameters,
                          lr = config['train']['learning_rate'],
                          warmup=config['train']['warmup_proportion'],
                          t_total=trainsteps_num)

    #----------------CallBacks----------------
    logger.info('initializing callbacks')
    model_checkpoint = Model_Checkpoint(checkpoint_dir=config['output']['checkpoint_dir'],
                                        logger = logger,
                                        mode = config['model']['callback']['mode'],
                                        target =config['model']['callback']['target'],
                                        arch = config['model']['arch'],
                                        only_save_best= config['model']['callbacks']['only_save_best']
                                        )

    training_monitor = TrainingMonitor(file_dir=config['output']['figure_dir'],
                                    arch = config['model']['arch'])
    lrscheduler  = BertLR(optimizer = optimizer,
                          lr= config['train']['learning_rate'],
                          t_total= trainsteps_num,
                          warmup=config['train']['warmup_proportion'])

    #----------------Train----------------
    logger.info('training...')
    train_config = {'model':model,
                    'logger':logger,
                    'optimizer':optimizer,
                    'model_checkpoint' : model_checkpoint,
                    'training_monitor' : training_monitor,
                    'lr_scheduler' : lrscheduler,
                    'epochs':config['train']['epochs'],
                    'resume':config['train']['resume'],
                    'gradient_accumulation_steps':config['train']['gradient_accumulation_steps'],
                    'epoch_metrics' : [F1Score(average='micro',task_type='binary'),MultiLabelReport(id2label=id2label)],
                    'barch_metrics' : [AccuracyThresh(thresh=0.5)],
                    'criterion': BCEWithLogitsLoss(),
                    'early_stop' : config['train']['early_stop'],
                    'verbose' : 1      }
    label_trainer = LabelTrainer(train_config = train_config)
    label_trainer.train(trian_data = label_train_loader,valid_data= label_valid_loader)

    if device!= 'cpu':
        torch.cuda.empty_cache()




    print('debug')


if __name__ == '__main__':
    main()
