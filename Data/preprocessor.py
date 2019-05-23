#encoding:utf-8

class EnglishPreprocessor(object):
    def __init__(self,min_len=2,stopwords_path=None):
        '''

        :param min_len: 视为句子的最小长度
        :param stopwords_path: 用于初始化禁用词的文件路径
        '''
        self.min_len = min_len
        self.stopwords = set()
        if stopwords_path:
            with open(stopwords_path,'r',encoding='UTF-8') as f:
                for line in f:
                    word = line.strip('\n').strip(' ')
                    self.stopwords.add(word)


