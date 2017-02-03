# coding: utf-8
import pandas as pd
import numpy as np
import csv as csv

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score


def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    return (accuracy_score(y, yPred),
            precision_score(y, yPred, labels=['1'],average='macro'),
            recall_score(y, yPred, labels=['1'], average='macro'))

def my_scorer(estimator, x, y):
    a, p, r = getScores(estimator, x, y)
    print u'sklearn准确率:%s,准确率:%s,召回率:%s\n'% (a, p, r)
    return a

def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df

def change_data(train_df):
    # female = 0, Male = 1
    # map遍历某列的每一个数据,使用一个函数或者一个字典映射对象
    train_df['Gender'] = train_df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    dummies_Pclass = pd.get_dummies(train_df['Pclass'], prefix='Pclass')
    dummies_Embarked = pd.get_dummies(train_df['Embarked'], prefix='Embarked')

    # 使用中位数填充age
    # train_df.Age = train_df.Age.fillna(train_df.Age.dropna().median())

    train_df.loc[(train_df.Age.isnull()) & (train_df.Gender == 0), 'Age'] = train_df.Age.dropna().loc[
        (train_df.Gender == 0)].median()
    train_df.loc[(train_df.Age.isnull()) & (train_df.Gender == 1), 'Age'] = train_df.Age.dropna().loc[
        (train_df.Gender == 1)].median()

    train_df.loc[(train_df.Fare.isnull()), 'Fare'] = train_df.Fare.dropna().median()
    #
    # median_age = train_df['Age'].dropna().median()
    # if len(train_df.Age[train_df.Age.isnull()]) > 0:
    #     train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

    # train_df = set_missing_ages(train_df)

    train_df.loc[(train_df.Cabin.notnull()), 'Cabin'] = 1
    train_df.loc[(train_df.Cabin.isnull()), 'Cabin'] = 0

    train_df.Cabin = train_df.Cabin.astype(int)

    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(train_df['Age'])
    train_df['Age_scaled'] = scaler.fit_transform(train_df['Age'], age_scale_param)

    fare_scale_param = scaler.fit(train_df['Fare'])
    train_df['Fare_scaled'] = scaler.fit_transform(train_df['Fare'], fare_scale_param)

    ids = train_df.PassengerId.values

    train_df = pd.concat([train_df, dummies_Pclass, dummies_Embarked], axis=1)

    # 删除暂时不需要的列

    train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'PassengerId', 'Embarked', 'Age', 'Fare', 'Pclass'], axis=1)
    return train_df,ids

def run():
    train_df = pd.read_csv('train.csv', header=0)

    train_df,ids = change_data(train_df)

    test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

    test_df,ids = change_data(test_df)

    train_data = train_df.values
    test_data = test_df.values

    print train_df.head()
    print train_df.dtypes

    print 'Training...'
    # forest = RandomForestClassifier(n_estimators=100)
    forest = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    forest.fit( train_data[0::,1::], train_data[0::,0] )

    if hasattr(forest, 'feature_importances_'):
        print 'feature_importances_:%s' % (['%0.8f' % i for i in forest.feature_importances_])

    fit_x, predict_x, fit_y, predict_y = train_test_split(train_data[0::,1::], train_data[0::,0], test_size=0.2)
    print confusion_matrix(predict_y, forest.predict(predict_x))
    print classification_report(predict_y, forest.predict(predict_x))

    print 'cross_val_score...'
    scores = cross_val_score(forest, train_data[0::,1::], train_data[0::,0],scoring=my_scorer)

    print 'Predicting...'
    output = forest.predict(test_data).astype(int)


    predictions_file = open("my_rf.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    print 'Done.'

if __name__=="__main__":
    run()