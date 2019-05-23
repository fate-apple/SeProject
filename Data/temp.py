#encoding: utf-8
'''
@time: 2019/5/14 18:19
@desc:
'''
import json
import random
def read_data(self,data_path):
        data = []
        with open(data_path) as f:
            for line in f:
                loaded_example = json.loads(line)
                if loaded_example["gold_label"] not in self.LABEL_MAP:
                    continue
                loaded_example["label"] = self.LABEL_MAP[loaded_example["gold_label"]]
                data.append(loaded_example)
            random.seed(1)
            random.shuffle(data)
        return data
