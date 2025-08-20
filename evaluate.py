import json
import re
import string
import numpy as np
import evaluate
from tqdm import *
import pandas as pd

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    def split_ans(text):
        if '<answer>' in text:
            text = text.split('<answer>')[1]
        if '</answer>' in text:
            text = text.split('</answer>')[0]
        return text
    return white_space_fix(remove_articles(remove_punc(split_ans(lower(s)))))

class Evaluator:
    def __init__(self, datapath):
        self.ans, self.predict = self.read_data(datapath)

    def read_data(self, datapath):
        with open(datapath, 'r') as f:
            data = json.load(f)
        ans = [[normalize_answer(i) for i in p['answer']] for p in data]
        predict = [normalize_answer(p['predict']) for p in data]
        return ans, predict

    def cal_acc(self, ans, predict):
        acc = 0
        for ref, pred in zip(ans, predict):
            flag = False
            for r in ref:
                if r in pred:
                    flag = True
                    break
            if flag:
                acc += 1
        return np.round(acc / len(ans), 3)

    def run(self):
        return self.cal_acc(self.ans, self.predict)
