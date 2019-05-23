from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader
import numpy as np
from Data.data_transformer import *
from Data.preprocessor import EnglishPreprocessor
from Data.Dataset import *


def PreLabelDL(logger,labels,config):

    #PreProcess
    DT = LabelDT(logger,seed =config['common']['seed'] )
    targets,sentences = DT.read_data(raw_data_path = config['data']['label_raw_data_path'],
                                     preprocessor = EnglishPreprocessor(),
                                     is_train=True)
    train,valid = DT.SplitDatasetTV(x=sentences,y=targets,save=True,stratify=True,
                                     valid_size = (config['train']['valid_size'])/(1-(config['train']['valid_size'])),
                                     train_path = config['data']['train_path'],
                                     valid_path = config['data']['valid_path'])
    #def __init__(self, vocab_file, do_lower_case=True, max_len=None, do_basic_tokenize=True,
    #            never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
    tokenizer = BertTokenizer(vocab_file=config['data']['pretrained']['bert_vocab_path'],
                              do_lower_case=config['train']['do_lower_case'])

    train_dataset = LabelDataset(data = train,
                                 tokenizer = tokenizer,
                                 max_seq_len = config['train']['max_seq_len'],
                                 seed = config['common']['seed'],
                                 example_type = 'train')
    valid_dataset = LabelDataset(data = valid,
                                 tokenizer=tokenizer,
                                 max_seq_len=config['train']['max_seq_len'],
                                 seed = config['common']['seed'],
                                 example_type='valid')
    #pin_memory (bool, optional): If ``True``, the data loader will copy tensors into CUDA pinned memory before returning them.
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = config['train']['batch_size'],
                              num_workers= config['common']['num_workers'],
                              shuffle=True,
                              drop_last=False,
                              pin_memory=False)
    valid_loader = DataLoader(dataset = valid_dataset,
                              batch_size = config['train']['batch_size'],
                              num_workers= config['common']['num_workers'],
                              shuffle=False,
                              drop_last=False,
                              pin_memory=False)
    return train_loader,valid_loader

def PreNerDL(logger,config):
    ner2idx = {ner:idx for idx,ner in enumerate(config['Ner'])}
    tokenizer = BertTokenizer(vocab_file=config['model']['pretrained']['bert_vocab_path'],
                              do_lower_case=config['train']['do_lower_case'])
    DT = Conll2003DT(logger,seed=config['common']['seed'])
    #train_l_words,train_l_ners = DT.read_data(path=['data']['ner_train'])
    train_data = DT.read_data(path=config['data']['train_path'])
    valid_data = DT.read_data(path=config['data']['valid_path'])

    train_dataset = NerDataset(data = train_data,
                               tokenizer=tokenizer,
                               ner2idx = ner2idx,
                               max_seq_len = config['train']['max_seq_len'],
                               seed = config['common']['seed'],
                               example_type = 'train'
                               )
    valid_dataset = NerDataset(data = valid_data,
                               tokenizer=tokenizer,
                               ner2idx = ner2idx,
                               max_seq_len = config['train']['max_seq_len'],
                               seed = config['common']['seed'],
                               example_type = 'train'
                               )
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size = config['train']['batch_size'],
                              shuffle=True,
                              num_workers= config['common']['num_workers'],
                               pin_memory=False,
                              #TODO :just pad to max_seq_len in dataset
                              collate_fn=NerDataset.pad
                              )
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size = config['train']['batch_size'],
                              shuffle=False,
                              num_workers= config['common']['num_workers'],
                              pin_memory=False,
                              collate_fn=NerDataset.pad
                              )
    return train_loader,valid_loader

def PreNerDL_test(logger,config):
    ner2idx = {ner:idx for idx,ner in enumerate(config['Ner'])}
    tokenizer = BertTokenizer(vocab_file=config['model']['pretrained']['bert_vocab_path'],
                              do_lower_case=config['train']['do_lower_case'])
    DT = Conll2003DT(logger,seed=config['common']['seed'])

    test_data = DT.read_data(path=config['data']['test_path'])

    test_dataset = NerDataset(data = test_data,
                               tokenizer=tokenizer,
                               ner2idx = ner2idx,
                               max_seq_len = config['train']['max_seq_len'],
                               seed = config['common']['seed'],
                               example_type = 'train'
                               )

    test_loader = DataLoader(dataset=test_dataset,
                              batch_size = 1,
                              shuffle=True,
                              num_workers= config['common']['num_workers'],
                               pin_memory=False,
                              #TODO :just pad to max_seq_len in dataset
                              collate_fn=NerDataset.pad
                              )

    return test_loader


def PreCnNerDL(logger,config):
    ner2idx = {ner:idx for idx,ner in enumerate(config['Ner'])}
    tokenizer = BertTokenizer(vocab_file=config['model']['pretrained']['bert_vocab_path'],
                              do_lower_case=config['train']['do_lower_case'])
    DT = BIO2014DT(logger,seed=config['common']['seed'])
    l_words,l_ners   = DT.read_data(source_path = config['data']['source_path'],
                          target_path=config['data']['target_path'])
    train_data,valid_data = DT.SplitDatasetTV(x=l_words,y=l_ners,save=True,
                                     valid_size = (config['train']['valid_size'])/(1-(config['train']['valid_size'])),
                                     train_path = config['data']['train_path'],
                                     valid_path = config['data']['valid_path'])

    train_dataset = NerDataset(data = train_data,
                               tokenizer=tokenizer,
                               ner2idx = ner2idx,
                               max_seq_len = config['train']['max_seq_len'],
                               seed = config['common']['seed'],
                               example_type = 'train'
                               )
    valid_dataset = NerDataset(data = valid_data,
                               tokenizer=tokenizer,
                               ner2idx = ner2idx,
                               max_seq_len = config['train']['max_seq_len'],
                               seed = config['common']['seed'],
                               example_type = 'valid'
                               )
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size = config['train']['batch_size'],
                              shuffle=True,
                              num_workers= config['common']['num_workers'],
                               pin_memory=False,
                              #TODO :just pad to max_seq_len in dataset
                              collate_fn=NerDataset.pad
                              )
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size = config['train']['batch_size'],
                              shuffle=False,
                              num_workers= config['common']['num_workers'],
                              pin_memory=False,
                              collate_fn=NerDataset.pad
                              )
    return train_loader,valid_loader

def PreMnliDL(logger,config):
    label2idx = config['Labels']
    tokenizer = BertTokenizer(vocab_file=config['model']['pretrained']['bert_vocab_path'],
                              do_lower_case=config['train']['do_lower_case'])
    DT = MnliDT(logger,seed=config['common']['seed'],label = label2idx)
    data  = DT.read_data(data_path = config['data']['source_path'])
    train_data,valid_data = DT.SplitDatasetTV(data=data,save=True,stratify = True,
                                     valid_size = (config['train']['valid_size'])/(1-(config['train']['valid_size'])),
                                     train_path = config['data']['train_path'],
                                     valid_path = config['data']['valid_path'])

    train_dataset = MnliDataset(data = train_data,
                               tokenizer=tokenizer,
                               label2idx = label2idx,
                               max_seq_len = config['train']['max_seq_len'],
                               seed = config['common']['seed'],
                               example_type = 'train'
                               )

    valid_dataset = MnliDataset(data = valid_data,
                               tokenizer=tokenizer,
                               label2idx = label2idx,
                               max_seq_len = config['train']['max_seq_len'],
                               seed = config['common']['seed'],
                               example_type = 'valid'
                               )
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size = config['train']['batch_size'],
                              shuffle=True,
                              num_workers= config['common']['num_workers'],
                               pin_memory=False,
                              collate_fn=MnliDataset.pad
                              )
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size = config['train']['batch_size'],
                              shuffle=False,
                              num_workers= config['common']['num_workers'],
                              pin_memory=False,
                              collate_fn=MnliDataset.pad
                              )
    return train_loader,valid_loader

def PreMnliDL_test(logger,config):
    label2idx = config['newLabels']
    tokenizer = BertTokenizer(vocab_file=config['model']['pretrained']['mnli_vocab_path'],
                              do_lower_case=config['train']['do_lower_case'])
    DT = MnliDT(logger,seed=config['common']['seed'],label = label2idx)
    test_data  = DT.read_data(data_path = config['data']['test_path'])

    test_dataset = MnliDataset(data = test_data,
                               tokenizer=tokenizer,
                               label2idx = label2idx,
                               max_seq_len = config['train']['max_seq_len'],
                               seed = config['common']['seed'],
                               example_type = 'test'
                               )

    test_loader = DataLoader(dataset=test_dataset,
                              batch_size = config['train']['batch_size'],
                              shuffle=True,
                              num_workers= config['common']['num_workers'],
                               pin_memory=False,
                              collate_fn=MnliDataset.pad
                             )
    return test_loader

def PreBIODL(logger,config):
    ner2idx = {ner:idx for idx,ner in enumerate(config['Ner'])}
    tokenizer = BertTokenizer(vocab_file=config['model']['pretrained']['bert_vocab_path'],
                              do_lower_case=config['train']['do_lower_case'])
    DT = ontonotes5DT(logger,seed=config['common']['seed'])
    train_data = DT.read_data(path=config['data']['train_path'])
    valid_data = DT.read_data(path=config['data']['valid_path'])

    train_dataset = NerDataset(data = train_data,
                               tokenizer=tokenizer,
                               ner2idx = ner2idx,
                               max_seq_len = config['train']['max_seq_len'],
                               seed = config['common']['seed'],
                               example_type = 'train'
                               )
    valid_dataset = NerDataset(data = valid_data,
                               tokenizer=tokenizer,
                               ner2idx = ner2idx,
                               max_seq_len = config['train']['max_seq_len'],
                               seed = config['common']['seed'],
                               example_type = 'train'
                               )
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size = config['train']['batch_size'],
                              shuffle=True,
                              num_workers= config['common']['num_workers'],
                               pin_memory=False,
                              #TODO :just pad to max_seq_len in dataset
                              collate_fn=NerDataset.pad
                              )
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size = config['train']['batch_size'],
                              shuffle=False,
                              num_workers= config['common']['num_workers'],
                              pin_memory=False,
                              collate_fn=NerDataset.pad
                              )
    return train_loader,valid_loader

def PreBIODL_test(logger,config):
    ner2idx = {ner:idx for idx,ner in enumerate(config['Ner'])}
    tokenizer = BertTokenizer(vocab_file=config['model']['pretrained']['bert_vocab_path'],
                              do_lower_case=config['train']['do_lower_case'])
    DT = Conll2003DT(logger,seed=config['common']['seed'])

    test_data = DT.read_data(path=config['data']['test_path'])

    test_dataset = NerDataset(data = test_data,
                               tokenizer=tokenizer,
                               ner2idx = ner2idx,
                               max_seq_len = config['train']['max_seq_len'],
                               seed = config['common']['seed'],
                               example_type = 'train'
                               )

    test_loader = DataLoader(dataset=test_dataset,
                              #batch_size = config['train']['batch_size'],
                             batch_size = 1,
                              shuffle=True,
                              num_workers= config['common']['num_workers'],
                               pin_memory=False,
                              #TODO :just pad to max_seq_len in dataset
                              collate_fn=NerDataset.pad
                              )

    return test_loader


