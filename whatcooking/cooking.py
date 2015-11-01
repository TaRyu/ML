# coding: utf-8
import sys
import numpy as np
import pandas as pd

reload(sys)
sys.setdefaultencoding('utf-8')
FTRAIN = '/home/chlau/data/whatcooking/train.json'
FTEST = ''


def load():
    data = pd.read_json(FTRAIN)
    tl = []
    tl2 = []
    for i in data.ingredients:
        tl = list(set(tl).union(set(i)))
    for x in data.ingredients:
        ta = np.zeros(6714)
        for i in x:
            ta[tl.index(i)] = 1
        tl2.append(ta)
    data['ingredients_number'] = tl2
    return data, tl
