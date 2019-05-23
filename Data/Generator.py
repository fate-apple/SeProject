#encoding: utf-8
'''
@time: 2019/5/22 13:56
@desc:
'''
from Config.BasicConfig import ner_configs,Mnli_configs,BIO_configs
from Pybert.Predictor import MnliPredictor,NerPredictor
import pymongo

class KG_Generator(object):
    def __init__(self,logger,connection):
        self.logger = logger
        self.connection = connection

    def ProcessDoc(self,doc):
        raise  NotImplementedError
    def GenerateKG(self):
        raise  NotImplementedError

class EventCausalityKG_Generator(KG_Generator):
    def __init__(self,models,configs,logger,connection):
        super(EventCausalityKG_Generator,self).__init__(logger,connection)
        self.NerModel = models[0]
        self.NerConfig = configs[0]
        self.BIOModel = models[1]
        self.BIOConfig = configs[1]
        self.MnliModel = models[2]
        self.MnliConfig = configs[2]


