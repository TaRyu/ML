# coding: utf-8
# from numba import jit
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

FTRAIN_LINUX = '/home/chlau/data/crime/train.csv'
FTRAIN_WIN = 'C:\\Users\\ChLiu\\Documents\\GitHub\\data\\crime\\train.csv'
FTEST_LINUX = '/home/chlau/data/crime/test.csv'
FTEST_WIN = 'C:\\Users\\ChLiu\\Documents\\GitHub\\data\\crime\\test.csv'
FPATH = '/home/chlau/data/crime/'


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


def run_fit(data_train, data_trian_label, key):
    clf = RandomForestClassifier(verbose=1)
    t0 = time.time()
    clf.fit(data_train, data_trian_label)
    t1 = time.time() - t0
    joblib.dump(
        clf, '%ssave_model/%s.pkl' % (FPATH, key))
    print('%s costs %.3f' % (key, t1,))


def predict(data_test, key):
    t0 = time.time()
    clf = joblib.load('%ssave_model/%s.pkl' % (FPATH, key))
    resualt = clf.predict(data_test)
    t1 = time.time() - t0
    print('prediction costs %.3f' % t1)
    resualt = pd.DataFrame(data={'Category': resualt})
    dummies = pd.get_dummies(data['Category'])
    dummies = dummies.rename(columns=lambda x: str(x))
    resualt = pd.concat([resualt, dummies], axis=1).drop('Category', axis=1)
    resualt.index.name = 'Id'
    resualt.to_csv('%s_resualt.csv' % key)

if __name__ == '__main__':
    data, _ = load(True)
    predict(data, 'rf')
