#encoding: utf-8
'''
@time: 2019/5/9 16:10
@desc:
'''
import torch
import os
import warnings
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
import wikipedia

from Pybert.utils.utils import seed_everything
from Config.BasicConfig import ner_configs as config
from Data.PrepareDL import PreNerDL
from Pybert.utils.logger import init_logger
from Pybert.model.nn.bert_finetune import Bert_Ner_Finetune
from Pybert.callback.model_checkpoint import Model_Checkpoint
from Pybert.callback.training_monitor import TrainingMonitor
from Pybert.callback.lr_scheduler import BertLR
from Pybert.train.metric import  F1Score,MultiLabelReport,AccuracyThresh
from Pybert.train.Trainer import LabelTrainer,NerTrainer
from Pybert.train.losses import CrossEntropy
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
    #----------------Prepare Data----------------

    #----------------Model----------------
    logger.info('Initializing Model')
    id2ner = {index:label for index,label in enumerate(config['Ner'])}
    model = Bert_Ner_Finetune.from_pretrained(pretrained_model_name_or_path =config['model']['pretrained']['bert_model_dir'],
                                          cache_dir = config['output']['cache_dir'],
                                          num_classes = len(config['Ner']))
    logger.info('Initializing List')
    event_list = {}

    logger.info('Generating Data')
    change_flag = False
    while(len(event_list)>0):
        event_type,arg1,arg2,arg3 = event_list[0]
        try:
            content =wikipedia.page(arg1).content()
        except wikipedia.exceptions.DisambiguationError as e :
            content = wikipedia.page(e.options[0]).content()
        paragraphs  = content.split('\n')
        sentences = []
        for p  in paragraphs:
            sentences.extend(p.split('.'))



