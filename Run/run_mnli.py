#encoding: utf-8
'''
@time: 2019/5/11 15:34
@desc:
'''
import torch
import os
import warnings

from pytorch_pretrained_bert.optimization import BertAdam


from Pybert.utils.utils import seed_everything
from Config.BasicConfig import Mnli_configs as config
from Data.PrepareDL import PreMnliDL,PreMnliDL_test
from Pybert.utils.logger import init_logger
from Pybert.model.nn.bert_finetune import Bert_Mnli_Finetune
from Pybert.callback.model_checkpoint import Model_Checkpoint
from Pybert.callback.training_monitor import TrainingMonitor
from Pybert.callback.lr_scheduler import BertLR
from Pybert.train.metric import  F1Score,MultiLabelReport,AccuracyThresh
from Pybert.train.Trainer import MnliTrainer
from Pybert.train.losses import *
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from Pybert.Predictor import MnliPredictor
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
    #label_train_loader ,label_valid_loader  = PreLabelDL()
    Mnli_train_loader,Mnli_valid_loader = PreMnliDL(logger,config)

    #----------------Model----------------
    logger.info('Initializing Model')
    #model = Bert_Mnli_Finetune.from_pretrained(pretrained_model_name_or_path =config['model']['pretrained']['bert_model_dir'],
    #                                     cache_dir = config['output']['cache_dir'],
    #                                      num_classes = len(config['Labels']))
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path =config['model']['pretrained']['bert_model_dir'],
              cache_dir = config['output']['cache_dir'],
              num_labels=len(config['Labels']))

    #----------------Optimizer----------------
    param_optimizer = list(model.named_parameters())#Returns an iterator over module parameters, yielding both the  name of the parameter as well as the parameter itself.
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
    trainsteps_num = int(   len(Mnli_train_loader) / config['train']['gradient_accumulation_steps']# Number of updates steps to accumulate before performing a backward/update pass.
                            * config['train']['epochs'])
    optimizer  = BertAdam(optimizer_grouped_parameters,
                          lr = config['train']['learning_rate'],
                          warmup=config['train']['warmup_proportion'],
                          t_total=trainsteps_num)

    #----------------CallBacks----------------
    logger.info('initializing callbacks')
    model_checkpoint = Model_Checkpoint(checkpoint_dir=config['output']['checkpoint_dir'],
                                        logger = logger,
                                        optimode = config['model']['callback']['mode'],
                                        target =config['model']['callback']['target'],
                                        arch = config['model']['arch'],
                                        only_save_best= config['model']['callback']['only_save_best']
                                        )

    training_monitor = TrainingMonitor(file_dir=config['output']['figure_dir'],
                                    arch = config['model']['arch'])
    lrscheduler  = BertLR(optimizer = optimizer,
                          lr= config['train']['learning_rate'],
                          t_total= trainsteps_num,
                          warmup=config['train']['warmup_proportion'])
    id2label = {value:key for key,value in config['Labels'].items()}

    #----------------Train----------------
    logger.info('training...')
    train_configs = {'model':model,
                    'logger':logger,
                    'optimizer':optimizer,
                    'model_checkpoint' : model_checkpoint,
                    'training_monitor' : training_monitor,
                    'lr_scheduler' : lrscheduler,
                    'n_gpu': config['common']['n_gpu'],
                    'epochs':config['train']['epochs'],
                    'resume':config['train']['resume'],
                    'gradient_accumulation_steps':config['train']['gradient_accumulation_steps'],
                    #'epoch_metrics' : [F1Score(average='micro',task_type='multiclass'),MultiLabelReport(id2label=id2ner)],
                     'epoch_metrics' : [F1Score(average='micro',task_type='multiclass',normalizate=False,only_head=False),
                                       ],
                    'batch_metrics':[],
                    'criterion': CrossEntropy(),
                    'early_stop' : config['train']['early_stop'],
                    'verbose' : 1      }
    Mnli_trainer = MnliTrainer(train_configs = train_configs)
    Mnli_trainer.train(train_data = Mnli_train_loader,valid_data= Mnli_valid_loader)

    if device!= 'cpu':
        torch.cuda.empty_cache()




    print('debug')

def test():

    print(os.path.pardir)
    os.chdir(os.path.pardir)
    device = f"cuda: {config['common']['n_gpu'][0] if len(config['common']['n_gpu'])!=0 else 'cpu'}"
    #----------------logger----------------
    logger = init_logger(log_name = config['model']['arch'],log_dir = config['output']['log_dir'])
    logger.info(f'device    :   {device}')
    logger.info(f"seed is {config['common']['seed']}")
    seed_everything(seed = config['common']['seed'],device=device)
    logger.info('load data from disk')

    test_loader = PreMnliDL_test(logger,config)
    model = Bert_Mnli_Finetune.from_pretrained(pretrained_model_name_or_path =config['model']['pretrained']['mnli_model_dir'],
                                         cache_dir = config['output']['cache_dir'],
                                          num_classes = len(config['Labels']))
    predicter = MnliPredictor(model = model,
                            logger = logger,
                            #model_path = config['output']['checkpoint_dir'] / f"best_{config['model']['arch']}_model.pth",
                              model_path=None,
                              config=config,
                              criterion= CrossEntropy()
                         )
    result = predicter.test(data = test_loader)


if __name__ == '__main__':
    test()
