#encoding:utf-8
from os import path
import multiprocessing
from pathlib import Path

Label_Dir = Path('Pybert')
Model_Dir = Path('Model')

label_configs = {
    'model':{
        'arch':'bert',
        'pretrained':{'bert_model_dir': Model_Dir / 'pretrained/pytorch_pretrain',
                      'pytorch_model_path': Model_Dir / 'pretrained/pytorch_pretrain/pytorch_model.bin',
                      'bert_vocab_path': Model_Dir / 'pretrain/uncased_L-12_H-768_A-12/vocab.txt'
                      },
        'callbacks':{'mode':'min',
                     'target': 'valid_loss',
                     'early_stop':20,
                     'only_save_best':True,
                     'save_freq':5

        }

    },
    'output':{
        'log_dir': Label_Dir / 'output/log',
        'checkpoint_dir' : Label_Dir  / "output/checkpoints",
    },
    'common':{
        'seed':1996,
        #1 for debug
        'num_workers':0,
        'n_gpu':[0],

    },
    'data':{
        'label_raw_data_path': Label_Dir / 'dataset/raw/train.txt',
        'train_path' : Label_Dir / 'dataset/processed/train.tsv',
        'valid_path' : Label_Dir / 'dataset/processed/valid.tsv',
    },
    'train':{
        'valid_size':0.15,
        'max_seq_len':256,
        'learning_rate':2e-5,
        'gradient_accumulation_steps': 1,
        'warmup_proportion':0.1,
        'resume':False,

    },
    'Ner':['<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG'],
    'Labels':('test')

}


Ner_Dir = Path('Ner')
ner_configs = {
    'model':{
        'arch':'bert_ner',
        'pretrained':{'bert_model_dir': Model_Dir / 'pretrained/pytorch_pretrain',
                      'pytorch_model_path': Model_Dir / 'pretrained/pytorch_pretrain/pytorch_model.bin',
                      'bert_vocab_path': (Model_Dir / 'pretrained/uncased_L-12_H-768_A-12/vocab.txt') ,
                      'chinese_bert_vocab_path': (Model_Dir / 'pretrained/chinese_L-12_H-768_A-12/vocab.txt') ,
                      },
        'callback':{'mode':'min',
                     'target': 'valid_loss',
                     'early_stop':20,
                     'only_save_best':True,
                     'save_freq':5

        },
        'do_lower_case':True,

    },
    'output':{
        'log_dir': Ner_Dir / 'output/log',
        'checkpoint_dir' :  Ner_Dir/ "output/checkpoints/1",
        'cache_dir':  Ner_Dir/ "output/model_cache",
        'figure_dir': Ner_Dir/ "output/fighre",
        'result': Ner_Dir / "output/result",
    },
    'common':{
        'seed':1996,
        #1 for debug
        'num_workers':0,
        'n_gpu':[0],

    },
    'data':{
        #'raw_data_path': Ner_Dir / 'dataset/raw/temp.txt',
        'train_path' : Ner_Dir / 'dataset/raw/temp.txt',
        'valid_path' : Ner_Dir / 'dataset/raw/temp2.txt',
        'test_path': Ner_Dir / 'dataset/raw/test.txt',
        'dataset':'Conll2003',

    },
    'train':{
        'valid_size':0.15,
        'max_seq_len':256,
        'learning_rate':2e-5,
        'gradient_accumulation_steps': 1,
        'warmup_proportion':0.1,
        #'resume':'Ner/output/checkpoints/best_bert_ner_model.pth',
        'resume':False,
        'epochs':6,
        'early_stop':None,
        'do_lower_case':True,
        'batch_size':1,

    },
    'Ner':['<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG'],
    'Labels':('test')

}


CnNer_Dir = Path('Ner_Cn')
chinese_ner_configs = {
    'model':{
        'arch':'bert_ner',
        'pretrained':{'bert_model_dir': Model_Dir / 'pretrained/chinese_pytorch_pretrain',
                      'pytorch_model_path': Model_Dir / 'pretrained/chinese_pytorch_pretrain/pytorch_model.bin',
                      'bert_config_file': Model_Dir / 'pretrained/chinese_L-12_H-768_A-12/bert_config.json',
                      'bert_vocab_path': Model_Dir / 'pretrained/chinese_L-12_H-768_A-12/vocab.txt' ,
                      'tf_checkpoint_path': Model_Dir / 'pretrained/chinese_L-12_H-768_A-12/bert_model.ckpt',
                      },
        'callback':{'mode':'min',
                     'target': 'valid_loss',
                     'early_stop':20,
                     'only_save_best':True,
                     'save_freq':5

        },
        'do_lower_case':True,

    },
    'output':{
        'log_dir': CnNer_Dir / 'output/log',
        'checkpoint_dir' :  CnNer_Dir/ "output/checkpoints",
        'cache_dir':  CnNer_Dir/ "output/model_cache",
        'figure_dir': CnNer_Dir/ "output/fighre",
        'result': CnNer_Dir / "output/result",
    },
    'common':{
        'seed':1996,
        #1 for debug
        'num_workers':0,
        'n_gpu':[0],

    },
    'data':{
        'source_path' : CnNer_Dir / 'dataset/raw/source_BIO_2014_cropus.txt',
        'target_path' : CnNer_Dir / 'dataset/raw/target_BIO_2014_cropus.txt',
        'train_path' :CnNer_Dir / 'dataset/processed/train_BIO_2014.txt',
        'valid_path' :CnNer_Dir / 'dataset/processed/valid_BIO_2014.txt',


    },
    'train':{
        'valid_size':0.15,
        'max_seq_len':256,
        'learning_rate':5e-5,
        'gradient_accumulation_steps': 1,
        'warmup_proportion':0.1,
        #'resume':'Ner/output/checkpoints/best_bert_ner_model.pth',
        'resume':False,
        'epochs':3,
        'early_stop':None,
        'do_lower_case':True,
        'batch_size':4,

    },
    'Ner':['<PAD>', 'O', 'I_LOC', 'B_PER', 'I_PER', 'I_ORG', 'B_LOC', 'B_ORG', 'B_T', 'I_T'],
    'Labels':('test')

}

Mnli_Dir = Path('multinli_1.0')
Mnli_configs = {
    'model':{
        'arch':'bert_ner',
        'pretrained':{'bert_model_dir': Model_Dir / 'pretrained/pytorch_pretrain',
                      'pytorch_model_path': Model_Dir / 'pretrained/pytorch_pretrain/pytorch_model.bin',
                      'bert_config_file': Model_Dir / 'pretrained/uncased_L-12_H-768_A-12/bert_config.json',
                      'bert_vocab_path': Model_Dir / 'pretrained/uncased_L-12_H-768_A-12/vocab.txt' ,
                      'tf_checkpoint_path': Model_Dir / 'pretrained/uncased_L-12_H-768_A-12/bert_model.ckpt',
                      'mnli_model_dir': Model_Dir / 'pretrained/mnli/',
                      'mnli_config_file': Model_Dir / 'pretrained/mnli/bert_config.json',
                      'mnli_vocab_path': Model_Dir / 'pretrained/mnli/vocab.txt' ,
                      },
        'callback':{'mode':'min',
                     'target': 'valid_loss',
                     'early_stop':20,
                     'only_save_best':True,
                     'save_freq':5

        },
        'do_lower_case':True,

    },
    'output':{
        'log_dir': Mnli_Dir / 'output/log',
        'checkpoint_dir' :  Mnli_Dir/ "output/checkpoints"/'1',
        'cache_dir':  Mnli_Dir/ "output/model_cache",
        'figure_dir': Mnli_Dir/ "output/fighre",
        'result': Mnli_Dir / "output/result",
    },
    'common':{
        'seed':2019,
        'num_workers':0,
        'n_gpu':[0],

    },
    'data':{
        'source_path' : Mnli_Dir / 'dataset/raw/multinli_1.0_temp.jsonl',
        #'target_path' : Mnli_Dir / 'dataset/raw/target_BIO_2014_cropus.txt',
        'train_path' :Mnli_Dir / 'dataset/processed/multinli_train.jsonl',
        'valid_path' :Mnli_Dir / 'dataset/processed/multinli_valid.jsonl',
        #'test_path' :Mnli_Dir / 'dataset/raw/multinli_1.0_dev_matched.jsonl',
        'test_path' :Mnli_Dir / 'dataset/raw/multinli_1.0_temp.jsonl',



    },
    'train':{
        'valid_size':0.15,
        'max_seq_len':256,
        'learning_rate':5e-5,
        'gradient_accumulation_steps': 1,
        'warmup_proportion':0.1,
        #'resume':'Ner/output/checkpoints/best_bert_ner_model.pth',
        'resume':False,
        'epochs':3,
        'early_stop':None,
        'do_lower_case':True,
        'batch_size':4,

    },
    'Labels' : {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
    },
    'newLabels' : {
            "contradiction": 0,
            "entailment": 1,
            "neutral": 2,
    }

}

BIO_Dir = Path('ontonotes_5.0')
BIO_configs = {
    'model':{
        'arch':'bert_ner',
        'pretrained':{'bert_model_dir': Model_Dir / 'pretrained/pytorch_pretrain',
                      'pytorch_model_path': Model_Dir / 'pretrained/pytorch_pretrain/pytorch_model.bin',
                      'bert_config_file': Model_Dir / 'pretrained/uncased_L-12_H-768_A-12/bert_config.json',
                      'bert_vocab_path': Model_Dir / 'pretrained/uncased_L-12_H-768_A-12/vocab.txt' ,
                      'tf_checkpoint_path': Model_Dir / 'pretrained/uncased_L-12_H-768_A-12/bert_model.ckpt',
                      },
        'callback':{'mode':'min',
                     'target': 'valid_loss',
                     'early_stop':20,
                     'only_save_best':True,
                     'save_freq':5
        },
        'do_lower_case':True,

    },
    'output':{
        'log_dir': BIO_Dir / 'output/log',
        'checkpoint_dir' :  BIO_Dir/ "output/checkpoints/4",
        'cache_dir':  BIO_Dir/ "output/model_cache",
        'figure_dir': BIO_Dir/ "output/fighre",
        'result': BIO_Dir / "output/result",
    },
    'common':{
        'seed':1996,
        'num_workers':0,
        'n_gpu':[0],

    },
    'data':{
        'train_path' :BIO_Dir / 'dataset/processed/BIOtrain.txt',
        'valid_path' :BIO_Dir / 'dataset/processed/BIOdevelopment.txt',
        'test_path' :BIO_Dir / 'dataset/processed/BIOtest.txt',
    },
    'train':{
        'valid_size':0.15,
        'max_seq_len':256,
        'learning_rate':5e-5,
        'gradient_accumulation_steps': 1,
        'warmup_proportion':0.1,
        'resume':False,
        'epochs':3,
        'early_stop':None,
        'do_lower_case':True,
        'batch_size':4,
    },
    'Ner':['<PAD>', 'O','V','B-ARG0','I-ARG0','B-ARG1','I-ARG1','B-ARG2','I-ARG2'],

}
