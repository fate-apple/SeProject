#encoding:utf-8
import os
from Config.BasicConfig import chinese_ner_configs as config
from pytorch_pretrained_bert.convert_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

if __name__ == "__main__":
    os.system('cp {config} {save_path}'.format(config = config['model']['pretrained']['bert_config_file'],
                                               save_path =config['model']['pretrained']['bert_model_dir']))
    convert_tf_checkpoint_to_pytorch(config['model']['pretrained']['tf_checkpoint_path'],
                                     config['model']['pretrained']['bert_config_file'],
                                     config['model']['pretrained']['pytorch_model_path'])
