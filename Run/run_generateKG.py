#encoding: utf-8
'''
@time: 2019/5/22 14:08
@desc:
'''
from Config.BasicConfig import  ner_configs,Mnli_configs,BIO_configs
from Pybert.model.nn.bert_finetune import Bert_Mnli_Finetune,Bert_Ner_Finetune,Bert_BIO_Finetune
from Pybert.Predictor import MnliPredictor,NerPredictor,BIOPredictor
from Pybert.utils.logger import *
from Pybert.utils.utils import Causal_Cue_Words
import wikipedia
import pymongo
import argparse
import torch.nn.functional as F
import re
import tqdm


def is_causality(textA,textB,mnli_predictor):
    logits = mnli_predictor.predict(textA=textA,textB=textB).view(-1)
    logits_softmax = F.softmax(logits,0)
    if logits_softmax[1]>0.01:
        return 1
    else :
        return 0
def extract_event(textA,ner_predictor,bio_predictor):
    from nltk.stem.lancaster import LancasterStemmer
    st = LancasterStemmer()
    event = {}
    _,output = ner_predictor.predict(textA)
    event['Ner'] = output
    _,output,des   = bio_predictor.predict_2(textA)
    event['Arg'] = output
    event['Des'] =des
    event['V'] = st.stem(output['V'])
    l =event['Ner'].values()
    print(f"raw : {textA}\n Ner : {'-'.join([' '.join(i) for i in l])}   Arg0:{output['ARG0']}"
          f"    V:{output['V']}   Arg1:{output['ARG1']}")
    return event

def meet_ignore(str1,str2):
    if len(str2)<30:
        return True
    if re.match(r'\d',str1[-1]) and re.match(r'\d',str2[0]):
        return True
    if str1[-1].lower()=='s' and str1[-3:-1].lower()=='u.' and str2[0]==' ':
        return True
    return False

def main():
    os.chdir(os.path.pardir)
    print(os.path.pardir)
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data",
                        default="wiki",
                        type=str,
                        help="choose raw data: wikipedia,")
    ## Other parameters
    parser.add_argument("--startword",
                        default=None,
                        type=str,
                        help="start of the data")
    args = parser.parse_args()

    EventCollection = pymongo.MongoClient().EcProject.Event2
    CausalityCollection = pymongo.MongoClient().EcProject.Causality
    count = EventCollection.find().count()
    device = f"cuda: 0"
    #----------------logger----------------
    logger = init_logger(log_name = 'generateKG',log_dir = 'log_dir')
    logger.info(f'device    :   {device}')

    config = Mnli_configs
    model = Bert_Mnli_Finetune.from_pretrained(pretrained_model_name_or_path =config['model']['pretrained']['mnli_model_dir'],
                                         cache_dir = config['output']['cache_dir'],
                                          num_classes = len(config['Labels']))
    mnli_predicter = MnliPredictor(model = model,
                            logger = logger,
                              model_path=None,
                              config=config,
                              criterion= None
                         )

    config = ner_configs
    model = Bert_Ner_Finetune.from_pretrained(pretrained_model_name_or_path =config['model']['pretrained']['bert_model_dir'],
                                          cache_dir = config['output']['cache_dir'],
                                          num_classes = len(config['Ner']))
    ner_predicter = NerPredictor(model = model,
                              logger = logger,
                              model_path = config['output']['checkpoint_dir'] / f"best_{config['model']['arch']}_model.pth",
                              config=config,
                              criterion= None
                         )

    config = BIO_configs
    model = Bert_BIO_Finetune.from_pretrained(pretrained_model_name_or_path =config['model']['pretrained']['bert_model_dir'],
                                          cache_dir = config['output']['cache_dir'])
    bio_predicter = BIOPredictor(model = model,
                              logger = logger,
                              model_path = config['output']['checkpoint_dir'] / f"best_{config['model']['arch']}_model.pth",
                              config=config,
                              criterion= None
                         )

    try:
        page =wikipedia.page(args.startword)
    except wikipedia.exceptions.DisambiguationError as e :
        page = wikipedia.page(e.options[0])
    content = page.content
    paragraphs  = content.split('\n')
    sentences = []
    for p  in paragraphs:
        #p = preprocess(p)
        l = p.split('.')
        n = len(l)
        i=0
        if n>1:
            while i<n-1:
                if meet_ignore(l[i],l[i+1]):
                    l[i+1] = l[i]+'.'+l[i+1]
                    #print(l[i+1])
                    i+=1
                else:
                    sentences.append(l[i])
                    i+=1
            sentences.append(l[n-1])

            #sentences.extend(l)

    for i in tqdm.tqdm(range(len(sentences)-1)):
        textA = sentences[i]
        for cueword in Causal_Cue_Words:
            if cueword in textA:
                textA,textB = textA.split(cueword,1)
                EventA = extract_event(textA,ner_predicter,bio_predicter)
                EventB = extract_event(textB,ner_predicter,bio_predicter)
                _idA = EventCollection.find_one({'Ner':EventA['Ner'],'V':EventA['V']})
                _idB = EventCollection.find_one({'Ner':EventB['Ner'],'V':EventB['V']})

                if not  _idA:
                    count+=1
                    idA  = count
                    EventCollection.insert_one({'_id':idA,'Ner':EventA['Ner'],'Arg':EventA['Arg'],'Des':EventA['Des']})
                else:
                    idA  =_idA['_id']
                if not _idB:
                    count+=1
                    idB  = count
                    EventCollection.insert_one({'_id':idB,'Ner':EventB['Ner'],'Arg':EventB['Arg'],'Des':EventB['Des']})
                else:
                    idB  =_idB['_id']
                CausalityCollection.insert_one({'cause':idA,'caused':idB})
                textA = textB
        textB = sentences[i+1]

        if is_causality(textA=textA,textB=textB,mnli_predictor=mnli_predicter):
            EventA = extract_event(textA,ner_predicter,bio_predicter)
            EventB = extract_event(textB,ner_predicter,bio_predicter)
            _idA = EventCollection.find_one({'Ner':EventA['Ner'],'V':EventA['V']})
            _idB = EventCollection.find_one({'Ner':EventB['Ner'],'V':EventB['V']})

            if not  _idA:
                count+=1
                idA  = count
                EventCollection.insert_one({'_id':idA,'Ner':EventA['Ner'],'Arg':EventA['Arg'],'Des':EventA['Des']})
            else:
                idA  =_idA['_id']
            if not _idB:
                count+=1
                idB  = count
                EventCollection.insert_one({'_id':idB,'Ner':EventB['Ner'],'Arg':EventB['Arg'],'Des':EventB['Des']})
            else:
                idB  =_idB['_id']
            CausalityCollection.insert_one({'cause':idA,'caused':idB})


def test():
    os.chdir(os.path.pardir)
    print(os.path.pardir)
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data",
                        default="wiki",
                        type=str,
                        required=True,
                        help="choose raw data: wikipedia,")
    ## Other parameters
    parser.add_argument("--startword",
                        default=None,
                        type=str,
                        help="start of the data")
    args = parser.parse_args()

    EventCollection = pymongo.MongoClient().EcProject.Event
    CausalityCollection = pymongo.MongoClient().EcProject.Causality
    count = EventCollection.find().count()
    device = f"cuda: 0"
    #----------------logger----------------
    logger = init_logger(log_name = 'generateKG',log_dir = 'log_dir')
    logger.info(f'device    :   {device}')

    config = Mnli_configs
    model = Bert_Mnli_Finetune.from_pretrained(pretrained_model_name_or_path =config['model']['pretrained']['mnli_model_dir'],
                                         cache_dir = config['output']['cache_dir'],
                                          num_classes = len(config['Labels']))
    mnli_predicter = MnliPredictor(model = model,
                            logger = logger,
                              model_path=None,
                              config=config,
                              criterion= None
                         )

    config = ner_configs
    model = Bert_Ner_Finetune.from_pretrained(pretrained_model_name_or_path =config['model']['pretrained']['bert_model_dir'],
                                          cache_dir = config['output']['cache_dir'],
                                          num_classes = len(config['Ner']))
    ner_predicter = NerPredictor(model = model,
                              logger = logger,
                              model_path = config['output']['checkpoint_dir'] / f"best_{config['model']['arch']}_model.pth",
                              config=config,
                              criterion= None
                         )

    config = BIO_configs
    model = Bert_BIO_Finetune.from_pretrained(pretrained_model_name_or_path =config['model']['pretrained']['bert_model_dir'],
                                          cache_dir = config['output']['cache_dir'])
    bio_predicter = BIOPredictor(model = model,
                              logger = logger,
                              model_path = config['output']['checkpoint_dir'] / f"best_{config['model']['arch']}_model.pth",
                              config=config,
                              criterion= None
                         )
    test_str  ="The Ministry of Commerce assumed administration of the college in 1904, and in 1905 changed the college's name to Imperial Polytechnic College of the Commerce Ministry. In 1906, the college was placed under the Ministry of Posts and Telegraphs, and its name was changed to Shanghai Industrial College of the Ministry of Posts and Telegraphs. When the Republic of China was founded, the college was placed under the Ministry of Communications and its name was once again changed, this time to Government Institute of Technology of the Communications Ministry."
    for sentence in test_str.split('.'):
        EventA = extract_event(sentence,ner_predicter,bio_predicter)

if __name__ == '__main__':
    test()






