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

class CreatePage:
    def __init__(self):
        self.base = '''
    <html>
    <head>
      <script type="text/javascript" src="VIS/dist/vis.js"></script>
      <link href="VIS/dist/vis.css" rel="stylesheet" type="text/css">
      <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    </head>
    <body>

    <div id="VIS_draw"></div>
    <script type="text/javascript">
      var nodes = data_nodes;
      var edges = data_edges;

      var container = document.getElementById("VIS_draw");

      var data = {
        nodes: nodes,
        edges: edges
      };

      var options = {
          nodes: {
              shape: 'dot',
              size: 25,
              font: {
                  size: 14
              }
          },
          edges: {
              font: {
                  size: 14,
                  align: 'middle'
              },
              color: 'gray',
              arrows: {
                  to: {enabled: true, scaleFactor: 0.5}
              },
              smooth: {enabled: false}
          },
          physics: {
              enabled: true
          }
      };

      var network = new vis.Network(container, data, options);

    </script>
    </body>
    </html>
    '''

    '''生成数据'''
    def collect_data(self, nodes, edges):
        node_dict = {node:index for index, node in enumerate(nodes)}
        data_nodes= []
        data_edges = []
        for node, id in node_dict.items():
            data = {}
            data["group"] = 'Event'
            data["id"] = id
            data["label"] = node
            data_nodes.append(data)

        for edge in edges:
            data = {}
            data['from'] = node_dict.get(edge[0])
            data['label'] = edge[2]
            data['to'] = node_dict.get(edge[1])
            data_edges.append(data)
        return data_nodes, data_edges

    '''生成html文件'''
    def create_html(self, data_nodes, data_edges,output_path):
        if not output_path:
            output_path ='a.out'
        try:
            f = open(output_path, 'w+',encoding='UTF-8')
        except:
            output_path ='a.out'
            f = open(output_path, 'w+',encoding='UTF-8')
        html = self.base.replace('data_nodes', str(data_nodes)).replace('data_edges', str(data_edges))
        f.write(html)
        f.close()
        return output_path

'''顺承关系图谱'''
class EventGraph:
    def __init__(self,conn_event,conn_causality):
        self.conn_event = conn_event
        self.conn_causality  =conn_causality

    '''统计事件频次'''
    def collect_events(self):
        event_dict = {}
        node_dict = {}
        causalitys = self.conn_causality.find_many({})
        for causality in causalitys:
            cause,effect,tag = causality['des_cause'],causality['des_effect'],causality['tag']
            for event in [cause,effect]:
                if event not in event_dict:
                    event_dict[event] = 1
                else:
                    event_dict[event] += 1

        return event_dict, node_dict

    '''过滤低频事件,构建事件图谱'''
    def filter_events(self,nodes_count):
        edges = []
        causalitys = self.conn_causality.find({})
        event_dict={}
        for causality in causalitys:
            cause,effect,tag = causality['des_cause'],causality['des_effect'],causality['tag']
            for event in [cause,effect]:
                if '[SEP]' not in event:
                    if event not in event_dict:
                        event_dict[event] = 1
                    else:
                        event_dict[event] += 1
                    edges.append([cause,effect,tag])
        events = [event for event,count in sorted(event_dict.items(), key=lambda asd: asd[1], reverse=True)[:nodes_count]]
        for edge in edges:
            if edge[0] not in events and edge[1] not in events:
                    edges.remove(edge)
        for edge in edges:
            if edge[0] not in events:
                events.append(edge[0])
            if edge[1] not in events:
                events.append(edge[1])
        return events, edges

    '''调用VIS插件,进行事件图谱展示'''
    def show_graph(self, nodes,edges,output_path):
        handler = CreatePage()
        data_nodes, data_edges = handler.collect_data(nodes, edges)
        output =handler.create_html(data_nodes, data_edges,output_path)
        return output

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
    parser.add_argument("--pages",
                        default=100,
                        type=int,
                        help="how many wiki pages to process")
    parser.add_argument("--refresh",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--genData",
                        action='store_true',
                        help="whether clean collection")
    parser.add_argument("--genGraph",
                        default=None,
                        type=str,
                        help="whether clean collection")
    parser.add_argument("--nodes",
                        default=100,
                        type=int)
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
    if  args.refresh:
        logger.info('refresh!')
        EventCollection.delete_many({})
        CausalityCollection.delete_many({})
    count = EventCollection.find().count()
    if args.genData:
        wikipedia.set_lang("en")
        work_list=[args.startword]
        worked_list=[]
        while(len(work_list)>0):
            if(work_list[0] in worked_list):
                work_list = work_list[1:]
                continue
            try:
                page =wikipedia.page(work_list[0])
            except wikipedia.exceptions.DisambiguationError as e :
                page = wikipedia.page(e.options[0])
            except:
                work_list = work_list[1:]
                continue
            work_list = work_list[1:]
            if(page.title in worked_list):
                continue
            logger.info(f'processing {page.title}')
            content = page.content

            paragraphs  = content.split('\n')
            sentences = []
            for p  in paragraphs:
                l = p.split('.')
                n = len(l)
                i=0
                if n>1:
                    while i<n-1:
                        if meet_ignore(l[i],l[i+1]):
                            l[i+1] = l[i]+'.'+l[i+1]
                            i+=1
                        else:
                            sentences.append(l[i])
                            i+=1
                    sentences.append(l[n-1])

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
                        CausalityCollection.insert_one({'cause':idA,'caused':idB,
                                                        'tag':cueword,
                                                        'des_cause':EventA['Des'],'des_effect':EventB['Des']})
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
                    CausalityCollection.insert_one({'cause':idA,'caused':idB,
                                                        'tag':'mnli-judge',
                                                        'des_cause':EventA['Des'],'des_effect':EventB['Des']})
            logger.info(f'processed {page.title}')
            if len(work_list)<args.pages:
                work_list.extend(page.links)
            worked_list.append(page.title)
            if len(worked_list)>args.pages:
                break
            logger.info(f"total process {len(worked_list)} page")
        logger.info(f"total process {len(worked_list)} page:\n{' '.join(worked_list)}")
    if args.genGraph:
        if args.genGraph=='default':
            if args.data=='wiki':
                output_path = '_'.join([args.data,str(args.pages),args.startword,
                                        str(args.nodes)])+'.html'
            if args.data=='PeopleDaily':
                output_path = '_'.join([args.data,
                                        str(args.nodes)])+'.html'
        else:
            output_path = args.genGraph
        logger.info('genGraph!')
        handler = EventGraph(EventCollection,CausalityCollection)
        nodes,edges = handler.filter_events(args.nodes)
        output = handler.show_graph(nodes,edges, output_path)
        logger.info(f"generate graph output in path : {output}")



if __name__ == '__main__':
    main()



