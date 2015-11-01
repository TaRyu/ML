# congding: utf-8
# from numba import jit
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

FTRAIN_LINUX = ''
FTRAIN_WIN = 'C:\\Users\\ChLiu\\Documents\\GitHub\\data\\crime\\train.csv'
FTEST_LINUX = ''
FTEST_WIN = 'C:\\Users\\ChLiu\\Documents\\GitHub\\data\\crime\\test.csv'
FPATH = ''


def load(test=False):
    try:
        data = pd.read_csv(FTEST_LINUX) if test else pd.read_csv(FTRAIN_LINUX)
    except IOError:
        data = pd.read_csv(FTEST_WIN) if test else pd.read_csv(FTRAIN_WIN)
    # data = pd.read_csv(FTRAIN_WIN)
    scaler = preprocessing.StandardScaler()
    # 正则化经纬度
    data['X_re'] = scaler.fit_transform(data.X)
    data['Y_re'] = scaler.fit_transform(data.Y)
    # 数字化警区
    dummies = pd.get_dummies(data.PdDistrict)
    dummies = dummies.rename(columns=lambda x: 'PdDistrict_' + str(x))
    data = pd.concat([data, dummies], axis=1)
    # 数字化星期
    dummies = pd.get_dummies(data.DayOfWeek)
    dummies = dummies.rename(columns=lambda x: 'DayOfWeek_' + str(x))
    data = pd.concat([data, dummies], axis=1)
    # 日期处理，一天内时间
    data.Dates = pd.to_datetime(data.Dates)
    data['Dates_D'] = data.Dates.map(lambda x: x.to_period('D').to_timestamp())
    data['Timedelate'] = scaler.fit_transform(
        (data.Dates - data.Dates_D).values)
    # 日期处理，载入测试集标准化
    try:
        data2 = pd.read_csv(FTRAIN_LINUX) if test else pd.read_csv(FTEST_LINUX)
    except IOError:
        data2 = pd.read_csv(FTRAIN_WIN) if test else pd.read_csv(FTEST_WIN)
    data2 = pd.to_datetime(data2.Dates)
    data2 = data.Dates.append(data2)
    data2 = scaler.fit_transform(data2 - pd.Timestamp('2015-06-01'))
    data.Dates_D = data2[:data.shape[0]]
    # 返回数据（训练时返回目标）
    if not test:
        target = data.Category
        data = data.drop(
            ['Dates', 'Category', 'Descript', 'DayOfWeek',
             'PdDistrict', 'Resolution', 'Address',
             'X', 'Y'], axis=1)
        data, target = shuffle(data, target, random_state=33)
    else:
        data = data.drop(
            ['Dates', 'DayOfWeek',
             'PdDistrict', 'Address',
             'X', 'Y'], axis=1)
        target = None

    return data, target


def run_fit(data_train, data_trian_label):
    clf = RandomForestClassifier(verbose=1)
    t0 = time.time()
    clf.fit(data_train, data_trian_label)
    t1 = time.time() - t0
    joblib.dump(
        clf, '%ssave_model/%s.pkl' % (FPATH))
    print('%s costs %.3f' % (t1))

if __name__ == '__main__':
    run_fit(load())
