import numpy as np
import tqdm
from sklearn.metrics import roc_auc_score,f1_score,classification_report
class Metric:
    def __init__(self):
        pass

    def __call__(self,ouputs,target):
        raise  NotImplementedError

    def reset(self):
        raise  NotImplementedError
    def value(self):
        raise  NotImplementedError
    def name(self):
        raise NotImplementedError

class F1Score(Metric):
    def __init__(self,thresh=0.5,normalizate=True,task_type='binary',average='binary',search_thresh=False,only_head=False):
        super(F1Score).__init__()
        assert  task_type in ['binary','multiclass']

        self.thresh = thresh
        self.task_type = task_type
        self.average = average
        self.search_thresh = search_thresh
        self.normalizate = normalizate
        self.only_head =  only_head
    def reset(self):
        '''
        重设目标真实类别和分类器预测得的类别
        :return:
        '''
        self.y_pred = []
        self.y_true = []
        #self.is_head =[]
    def __call__(self,logits,target,is_head=None):
        self.y_true = target.cpu().numpy()

        if self.normalizate:
            if self.task_type =='binary':
                y_prob = logits.sigmoid().data.cpu().numpy()
            elif self.task_type=='multiclass':
                y_prob = logits.softmax(-1).data.cpu().numpy()
        else:
            y_prob = logits.cpu().detach().numpy()

        if self.task_type=='binary':
            if self.thresh and self.search_thresh==False:
                self.y_pred = (y_prob>self.thresh).astype(int)
                self.value()
            else:
                thresh,f1 = self.thresh_search(y_prob=y_prob)
                print(f"Best thresh : {thresh:.4f}  -   F1 Score : {f1:.4f}")
        if self.task_type == 'multiclass':
            self.y_pred = np.argmax(y_prob,-1)
        if self.only_head and is_head is not None:
            is_head = is_head.cpu().numpy()
            self.y_true = self.y_true[np.where(is_head!=0)]
            self.y_pred = self.y_pred[np.where(is_head!=0)]
        assert  len(self.y_true)==len(self.y_pred)

    def thresh_search(self,y_prob):
        best_thresh = 0
        best_score = 0
        for thresh_old in tqdm([i*0.01 for i in range(100)],disable = True):
            self.y_pred = y_prob> thresh_old
            score = self.value()
            if score > best_score:
                best_thresh = thresh_old
                best_score = best_score
        return best_thresh,best_score
    def value(self):
        #if self.task_type =='binary':
            f1 = f1_score(y_true = self.y_true,y_pred=self.y_pred,average=self.average)
            return f1
        #elif self.task_type=='multicalss':
    def name(self):
        return 'f1_score'

class  mnli_simple_accuracy(Metric):
    def __init__(self):
        super(mnli_simple_accuracy).__init__()

    def reset(self):
        self.y_prob = 0
        self.y_true = 0
    def __call__(self,logits,target):
        self.y_prob = logits.argmax(-1).cpu().detach().numpy()
        self.y_true  = target.cpu().numpy()
    def value(self):
        return (self.y_prob==self.y_true).mean()
    def name(self):
        return "simple_accuracy"






class MultiLabelReport(Metric):
    def __init__(self,id2label=None):
        super(MultiLabelReport).__init__()
        self.id2label  = id2label

    def reset(self):
        self.y_prob = 0
        self.y_true = 0
    def __call__(self,logtis,target):
        self.y_prob = logtis.sigmoid().data.cpu().detach().numpy()
        self.y_true  = target.cpu().numpy()
    def value(self):
        for i,label in self.id2label.items():
            #AUC值越大，当前分类算法越有可能将正样本排在负样本前面，从而能够更好地分类。
            auc = roc_auc_score(y_score=self.y_prob[:,i],y_true=self.y_true[:,i])
            print(f"label ; {label} - auc : {auc:.4f}")
    def name(self):
        return "multilabel_report"

class AccuracyThresh(Metric):
    def __init__(self,thresh = 0.5):
        super(AccuracyThresh).__init__()
        self.thresh = thresh
        self.reset()
    def reset(self):
        self.correct_k = 0
        self.total = 0
    def __call__(self,logits,target):
        self.y_pred = logits.sigmoid()
        self.y_true = target
    def value(self):
        data_size =  self.y_pred.size(0)
        debug = np.mean( ((self.y_pred>self.thresh)==self.y_true.byte()).float().cpu().numpy(), axis=1)
        acc = debug.sum()
        return acc/data_size
    def name(self):
        return "AccuracyThresh"


