# coding: utf-8
import pandas as pd
import numpy as np
import csv as csv

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
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


def names(train, test):
    for i in [train, test]:
        i['Name_Len'] = i['Name'].apply(lambda x: len(x))
        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        i['Name_Title'] = np.where((i['Name_Title']).isin(['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.']),
                                   i['Name_Title'], 'other')
    good_cols = ['Name_Title' + '_' + i for i in train['Name_Title'].unique() if i in test['Name_Title'].unique()]
    train = pd.concat((train, pd.get_dummies(train['Name_Title'], prefix='Name_Title')[good_cols]), axis=1)
    test = pd.concat((test, pd.get_dummies(test['Name_Title'], prefix='Name_Title')[good_cols]), axis=1)
    return train,test

def age_impute(train, test):
    for i in [train, test]:
        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
        data = train.groupby(['Name_Title', 'Pclass'])['Age']
        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))
    return train, test

def cabin(train, test):
    for i in [train, test]:
        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
        i['Cabin_num1'].replace('an', np.NaN, inplace=True)
        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x <> '' else np.NaN)
        i['Cabin_num'] = pd.qcut(train['Cabin_num1'], 3)
    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix='Cabin_num')), axis=1)
    test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix='Cabin_num')), axis=1)
    train = train.drop(['Cabin_num1','Cabin_num'], axis=1)
    test = test.drop(['Cabin_num1','Cabin_num'], axis=1)
    return train, test

def fam_size(train, test):
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',
                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))
    good_cols = ['Fam_Size' + '_' + i for i in train['Fam_Size'].unique() if i in test['Fam_Size'].unique()]
    train = pd.concat((train, pd.get_dummies(train['Fam_Size'], prefix='Fam_Size')[good_cols]), axis=1)
    test = pd.concat((test, pd.get_dummies(test['Fam_Size'], prefix='Fam_Size')[good_cols]), axis=1)
    return train, test


def fare(train, test):
    for i in [train, test]:
        i.loc[(i.Fare.isnull()), 'Fare'] = i.Fare.dropna().median()
    return train, test


def sex(train, test):
    train = pd.concat([train, pd.get_dummies(train['Sex'], prefix='Sex')], axis=1)
    test = pd.concat([test, pd.get_dummies(test['Sex'], prefix='Sex')], axis=1)
    return train, test


def pclass(train, test):
    train = pd.concat([train, pd.get_dummies(train['Pclass'], prefix='Pclass'), pd.get_dummies(train['Embarked'], prefix='Embarked')], axis=1)
    test = pd.concat([test, pd.get_dummies(test['Pclass'], prefix='Pclass'), pd.get_dummies(test['Embarked'], prefix='Embarked')], axis=1)
    return train, test

def feature_process(train_df, test_df):
    train_df, test_df = names(train_df,test_df)
    train_df, test_df = age_impute(train_df,test_df)
    train_df, test_df = cabin(train_df,test_df)
    train_df, test_df = fam_size(train_df,test_df)
    train_df, test_df = fare(train_df,test_df)
    train_df, test_df = pclass(train_df,test_df)
    train_df, test_df = sex(train_df,test_df)

    return train_df,test_df

def get_best_param(train_data):
    rf14 = RandomForestClassifier(max_features='auto',
                                  oob_score=True,
                                  random_state=1,
                                  n_jobs=-1)

    param_grid = {"criterion": ["gini", "entropy"],
                  "min_samples_leaf": [1, 5, 10],
                  "min_samples_split": [2, 4, 10, 12, 16],
                  "n_estimators": [50, 100, 400, 700, 1000]}

    gs = GridSearchCV(estimator=rf14,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=3,
                      n_jobs=-1)

    gs = gs.fit(train_data[0::,1::], train_data[0::,0])

    print(gs.best_score_)
    print(gs.best_params_)

def run():
    train_df = pd.read_csv('train.csv', header=0)
    test_df = pd.read_csv('test.csv', header=0)  # Load the test file into a dataframe

    train_df,test_df = feature_process(train_df, test_df)

    train_df = train_df.drop(['Name','Name_Title','Sex', 'Ticket', 'PassengerId', 'Embarked', 'Pclass','Fam_Size','Cabin'], axis=1)

    ids = test_df['PassengerId']

    test_df = test_df.drop(['Name','Name_Title','Sex', 'Ticket', 'PassengerId', 'Embarked', 'Pclass','Fam_Size','Cabin'], axis=1)

    train_data = train_df.values


    print train_df.head()
    print train_df.dtypes
    print test_df.dtypes

    print 'Training...'
    # forest = RandomForestClassifier()
    forest = RandomForestClassifier(criterion='entropy',
                                  n_estimators=400,
                                  min_samples_split=16,
                                  min_samples_leaf=1,
                                  max_features='auto',
                                  oob_score=True,
                                  random_state=1,
                                  n_jobs=-1)
    # forest = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    forest.fit( train_data[0::,1::], train_data[0::,0] )

    fit_x, predict_x, fit_y, predict_y = train_test_split(train_data[0::,1::], train_data[0::,0], test_size=0.2)
    print confusion_matrix(predict_y, forest.predict(predict_x))
    print classification_report(predict_y, forest.predict(predict_x))

    print 'cross_val_score...'
    scores = cross_val_score(forest, train_data[0::,1::], train_data[0::,0],scoring=my_scorer,cv=5)

    # get_best_param(train_data)

    print 'Predicting...'



    test_data = test_df.values

    output = forest.predict(test_data).astype(int)

    predictions_file = open("my_rf.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()

    print pd.concat((pd.DataFrame(train_df.columns, columns=['variable']),
               pd.DataFrame(forest.feature_importances_, columns=['importance'])),
              axis=1).sort_values(by='importance', ascending=False)[:20]

    print 'Done.'

if __name__=="__main__":
    run()
    # *参考解决思路 87%* https://www.kaggle.com/jasonm/titanic/large-families-not-good-for-survival
    # *参考解决思路 88%* https://www.kaggle.com/scirpus/titanic/genetic-programming-lb-0-88/code