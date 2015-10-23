from sklearn import neighbors, cross_validation, svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import pandas as pd
import time
import re
import numpy as np


def getdata():
    data = pd.read_csv('data/train.csv')
    data_f = pd.DataFrame(data, columns=[
        'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'])
    data_t = data.Survived
    return data_f, data_t

data_train, data_train_label = getdata()
n_samples = data_train.shape[0]
cv = cross_validation.ShuffleSplit(n_samples,
                                   n_iter=3,
                                   test_size=0.3,
                                   random_state=0)


class DataProcess:

    """docstring for data_process"""

    def __init__(self, data=data_train):
        self.data = data
        self.scaler = preprocessing.StandardScaler()

    def fill_age(self, data_predict_age=[]):
        clf = RandomForestClassifier()
        clf.fit(data_predict_age[0], data_predict_age[1])
        return clf.predict(data_predict_age[2])

    def get_cabin(self, cabin):
        match = re.compile("([0-9])+").search(cabin)
        if match:
            return match.group()
        else:
            return 0

    def process_age(self):
        data_age = self.data.drop(['Name', 'Cabin', 'Embarked'], axis=1)
        data_age_train = data_age[data_age.Age.notnull()]
        data_age_train_label = data_age_train.Age
        data_age_train = data_age_train.drop('Age', axis=1)
        data_age_test = data_age[data_age.Age.isnull()].drop('Age', axis=1)
        self.data.loc[self.data.Age.isnull(), 'Age'] = self.fill_age(
            [data_age_train, data_age_train_label, data_age_test])
        bins = [0, 15, 45, 100]
        cats = pd.cut(self.data.Age, bins)
        self.data.Age = cats.labels
        self.data.Age = self.scaler.fit_transform(self.data.Age)

    def process_pclass(self):
        self.data.Pclass = self.scaler.fit_transform(self.data.Pclass)

    def process_sib_par(self):
        self.data.SibSp = self.scaler.fit_transform(self.data.SibSp)
        self.data.Parch = self.scaler.fit_transform(self.data.Parch)

    def process_fare(self):
        bins = [0, 15, 40, 100, 200, 600]
        cats = pd.cut(self.data.Fare, bins)
        self.data.Fare = cats.labels
        self.data.Fare = self.scaler.fit_transform(self.data.Fare)

    def process_cabin(self):
        self.data.Cabin[self.data.Cabin.isnull()] = 'U0'
        self.data['CabinLetter'] = self.data.Cabin.map(
            lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        self.data['CabinLetter'] = pd.factorize(self.data.CabinLetter)[0]
        self.data['CabinNumber'] = self.data.Cabin.map(
            lambda x: self.get_cabin(x)).astype(int) + 1
        # Normalize
        self.data.CabinLetter = self.scaler.fit_transform(
            self.data.CabinLetter)
        self.data.CabinNumber = self.scaler.fit_transform(
            self.data.CabinNumber)

    def process_emb(self):
        self.data.Embarked[
            self.data.Embarked.isnull()] = self.data.Embarked.dropna().mode().values
        dummies = pd.get_dummies(self.data.Embarked)
        dummies = dummies.rename(columns=lambda x: 'Embarked_' + str(x))
        self.data = pd.concat([self.data, dummies], axis=1)

    def process_name(self):
        self.data['Title'] = self.data.Name.map(
            lambda x: re.compile(",(.*?)\.").findall(x)[0])
        self.data['Title'][self.data.Title == 'Mme'] = 'Mrs'
        self.data['Title'][self.data.Title.isin(['Ms', 'Mlle'])] = 'Miss'
        self.data['Title'][
            self.data.Title.isin(['Capt', 'Don', 'Major', 'Col'])] = 'Sir'
        self.data['Title'][
            self.data.Title.isin(['Dona', 'the Countess'])] = 'Lady'
        self.data['Title'] = pd.factorize(self.data.Title)[0]
        self.data.Title = self.scaler.fit_transform(self.data.Title)

    def process_sex(self):
        self.data.Sex = pd.factorize(self.data.Sex)[0]

    def process_b4_age(self):
        self.process_cabin()
        self.process_emb()
        self.process_fare()
        self.process_name()
        self.process_pclass()
        self.process_sex()
        self.process_sib_par()

    def new_data(self):
        return self.data.drop(['Name', 'Cabin', 'Embarked'], axis=1)


def test_data():
    return pd.read_csv('data/test.csv')
data_test = test_data()


class TestDataProcess(DataProcess):

    """docstring for TestDataProcess"""

    def __init__(self):
        super(TestDataProcess, self).__init__()
        self.data = data_test

    def process_age(self):
        bins = [0, 15, 45, 100]
        cats = pd.cut(self.data.Age, bins)
        self.data.Age = cats.labels
        self.data.Age = self.scaler.fit_transform(self.data.Age)


def knn():
    return neighbors.KNeighborsClassifier(10, weights='distance')


def rf():
    return RandomForestClassifier()


def svc_poly(d=3):
    return svm.SVC(kernel='poly', gamma=0.7, degree=d)


def svc_rbf():
    return svm.SVC(kernel='rbf', gamma=0.7)


def svc_linear():
    return svm.SVC(kernel='linear', gamma=0.7)


def scores(clf, data_train, data_train_label):
    scores = cross_validation.cross_val_score(
        clf, data_train, data_train_label, cv=cv, scoring='f1')
    print (scores)
    return scores


def run_fit(data_train, data_trian_label,
            clfs={'knn': knn(),
                  'rf': rf(),
                  'svc_poly': svc_poly(),
                  'svc_linear': svc_linear(),
                  'svc_rbf': svc_rbf()}):
    for key in clfs:
        t0 = time.time()
        clfs[key].fit(data_train, data_trian_label)
        t1 = time.time() - t0
        joblib.dump(clfs[key], 'save_model/%s.pkl' % key)
        print('%s costs %.3f' % (key, t1))
    print('over')


def predict(path):
    t0 = time.time()
    clf = joblib.load(path)
    resualt = clf.predict(data_test)
    t1 = time.time() - t0
    print('prediction costs %.3f' % t1)
    resualt = pd.DataFrame(index=[x + 1 for x in range(28000)],
                           data={'Label': resualt})
    resualt.index.name = 'ImageId'
    resualt.to_csv('%s_resualt.csv' % path)
