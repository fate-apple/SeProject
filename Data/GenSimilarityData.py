#encoding: utf-8
'''
@time: 2019/5/26 19:27
@desc:
'''

import requests
import json
import time
from googletrans import  Translator
import  argparse
import pymongo
from Pybert.utils.logger import *
import  wikipedia
import tqdm
from tqdm import trange
import re
import jieba.posseg as pseg
import random
from pyltp import *
class similarityData_generator(object):
    def __init__(self,fw):
        self.translator  =  Translator()
        self.fw = fw


    def process_content(self, content):
        return [sentence for sentence in SentenceSplitter.split(content) if sentence]

    def fined_sentence(self, sentence):
        return re.split(r'[？！，；]', sentence)


    def process(self,content):
        paragraphs = content.split('\n')
        sentences = []

        for p  in paragraphs:
                if len(p.split('。'))<3:
                    continue
                else :
                    sentences.extend(self.process_content(p))

        subsents = []
        for sentence in sentences:
            subsents += self.fined_sentence(sentence)
        sent = subsents[0].strip()
        sent2 = subsents[1].strip()
        for j in trange(len(subsents)-2):
            try:
                time.sleep(random.uniform(0,2))
                temp =self.translator.translate(sent,src='zh-cn',dest='en')
                time.sleep(random.uniform(0,2))
                similar_data = self.translator.translate(temp.text,src='en',dest='zh-cn')
                self.fw.write('\t'.join([sent,similar_data.text,str(1)]))
                self.fw.write('\t'.join([sent,sent2,str(0)]))
                sent = sent2
                sent2 = ' '.join([word.word + '/' + word.flag for word in pseg.cut(subsents[j+2])]).strip()
            except:
                print('network error')
                continue
    def process2(self,content):
        paragraphs = content.split('\n')
        sentences = []

        for p  in paragraphs:
                if len(p.split('。'))<3:
                    continue
                else :
                    sentences.extend(self.process_content(p))

        subsents = []
        for sentence in sentences:
            subsents += self.fined_sentence(sentence)
        for subsent in subsents:
            self.fw.write(subsent+'\n')
        self.fw.write('\n')

    def process3(self,content):
        paragraphs = content.split('\n')
        sentences = []

        for p  in paragraphs:
                if len(p.split('。'))<3:
                    continue
                else :
                    sentences.extend(self.process_content(p))

        subsents = []
        for sentence in sentences:
            subsents += self.fined_sentence(sentence)
        try:
            temp =self.translator.translate(subsents,src='zh-cn',dest='en')
            time.sleep(random.uniform(0,2))
            similar_data = self.translator.translate([t.text for t in temp],src='en',dest='zh-cn')
            time.sleep(random.uniform(0,2))
        except:
            print('network error')

    def process4(self,content):
        try:
            paragraphs = content.split('\n')
            sentences = []

            for p  in paragraphs:
                    if len(p.split('。'))<3:
                        continue
                    else :
                        sentences.extend(self.process_content(p))

            subsents = []
            for sentence in sentences:
                subsents += [s for s in self.fined_sentence(sentence) if len(s)>3]
            new_subsents=[]
            for j in range(int((len(subsents)+31)/32)):

                s  = '\n'.join(subsents[j*32:j*32+32])
                time.sleep(random.uniform(0,1))
                temp = self.translator.translate(s,src='zh-cn',dest='en')
                time.sleep(random.uniform(0,1))
                similar_data = self.translator.translate(temp.text,src='en',dest='zh-cn')
                new_subsents.extend(similar_data.text.split('\n'))
            assert  len(subsents)== len(new_subsents)
            for i in range(len(subsents)-1):
                self.fw.write('\t'.join([subsents[i],subsents[i+1],str(0)])+'\n')
                self.fw.write('\t'.join([subsents[i],new_subsents[i],str(1)])+'\n')
        except:
            print('network error')


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--refresh",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--genData",
                        action='store_true',
                        help="whether clean collection")
    parser.add_argument("--fwpath",
                        default=None,
                        type=str,
                        help="whether clean collection")
    parser.add_argument("--pages",
                        default=100,
                        type=int,
                        help="how many wiki pages to process")
    args = parser.parse_args()

    device = f"cuda: 0"
    #----------------logger----------------
    logger = init_logger(log_name = 'generateKG',log_dir = 'log_dir')
    logger.info(f'device    :   {device}')

    EventCollection = pymongo.MongoClient().EcProject_cn.Event2
    CausalityCollection = pymongo.MongoClient().EcProject_cn.Causality

    if args.fwpath=='default':
        if args.data=='wiki':
                output_path = '_'.join([args.data,str(args.pages),args.startword,
                                        ])+'_similarity.txt'
        elif args.data=='PeopleDaily':
                output_path = '_'.join([args.data,str(args.pages),
                                        ])+'_similarity.txt'
    else:
        output_path = args.fwpath
    if  args.refresh:
        fw = open(output_path,'w',encoding='utf-8')
    else:
        fw = open(output_path,'a',encoding='utf-8')
    print(f"fw : {output_path}")

    handler = similarityData_generator(fw)
    if args.genData:
        logger.info('genData!')
        if args.data =='wiki':
            logger.info('genData!')
            wikipedia.set_lang("zh")
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
                handler.process(content)
                logger.info(f'processed {page.title}')
                if len(work_list)<args.pages:
                    work_list.extend(page.links)
                worked_list.append(page.title)
                if len(worked_list)>args.pages:
                    break
            logger.info(f"total process {len(worked_list)} page:\n{' '.join(worked_list)}")
        elif args.data=='PeopleDaily':
            fr = open('../PeopleDaily/news2016zh_train.json','r',encoding='utf-8')

            for i in tqdm.trange(args.pages):
                line = fr.readline()
                if len(line)>0:
                    example =  json.loads(line)
                    content = example['content']
                    handler.process4(content)
                else :
                    break

if __name__ == '__main__':
    main()
