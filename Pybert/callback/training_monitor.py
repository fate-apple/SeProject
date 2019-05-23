#encoding:utf-8
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class TrainingMonitor():
    def __init__(self,file_dir,arch,begin=0):
        if not isinstance(file_dir,Path):
            file_dir = Path(file_dir)
        #create any parentdir if needed
        file_dir.mkdir(parents=True,exist_ok=True)
        self.file_dir = file_dir
        self.arch = arch
        self.begin = begin
        self.file_path = file_dir / (arch+'_training_monitor.json')
        self.reset()
        self.history = {}

    def reset(self):
        if self.begin>0:
            if self.file_path is not None:
                self.history = json.loads(open(str(self.file_path,encoding='utf-8')).read())
                for k in self.history.keys():
                    self.history[k] = self.history[k][:self.begin]

    def epoch_step(self,logs={}):
        for k,v in logs.items():
            l_value = self.history.get(k,[])
            if not isinstance(v,np.float):
                v = round(float(v),4)
            l_value.append(v)
            self.history[k]=l_value


