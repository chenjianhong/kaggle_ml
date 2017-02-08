# coding: utf-8
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import feature_process
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score

# # RF Model 14

# #### Load data & transform variables


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train, test = feature_process.names(train, test)
train, test = feature_process.age_impute(train, test)
train, test = feature_process.cabin_num(train, test)
train, test = feature_process.cabin(train, test)
train, test = feature_process.embarked_impute(train, test)
train, test = feature_process.fam_size(train, test)
train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))
test['Ticket_Len'] = test['Ticket'].apply(lambda x: len(x))
train, test = feature_process.ticket_grouped(train, test)
train, test = feature_process.dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',
                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
train, test = feature_process.drop(train, test, bye = ['Ticket', 'SibSp', 'Parch'])


# #### Tune hyper-parameters


# rf14 = RandomForestClassifier(max_features='auto',
#                                 oob_score=True,
#                                 random_state=1,
#                                 n_jobs=-1)
#
# param_grid = { "criterion"   : ["gini", "entropy"],
#              "min_samples_leaf" : [1,5,10],
#              "min_samples_split" : [2, 4, 10, 12, 16],
#              "n_estimators": [50, 100, 400, 700, 1000]}
#
# gs = GridSearchCV(estimator=rf14,
#                   param_grid=param_grid,
#                   scoring='accuracy',
#                   cv=3,
#                   n_jobs=-1)
#
# gs = gs.fit(train.iloc[:, 2:], train.iloc[:, 1])
#
#
#
# print(gs.best_score_)
# print(gs.best_params_)
# #print(gs.cv_results_)


def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    return (accuracy_score(y, yPred),
            precision_score(y, yPred, labels=['1'],average='macro'),
            recall_score(y, yPred, labels=['1'], average='macro'))

def my_scorer(estimator, x, y):
    a, p, r = getScores(estimator, x, y)
    print u'sklearn准确率:%s,准确率:%s,召回率:%s\n'% (a, p, r)
    return a



# #### Fit model

rf14 = RandomForestClassifier(criterion='gini',
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf14.fit(train.iloc[:, 2:], train.iloc[:, 1])
print "%.4f" % rf14.oob_score_


# #### Obtain cross-validation score with optimal hyperparameters

scores1 = cross_val_score(rf14, train.iloc[:, 2:], train.iloc[:, 1], n_jobs=-1)
scores1.mean()


# #### Inspect feature ranking


pd.concat((pd.DataFrame(train.iloc[:, 2:].columns, columns = ['variable']),
           pd.DataFrame(rf14.feature_importances_, columns = ['importance'])),
          axis = 1).sort_values(by='importance', ascending = False)[:20]


# #### Generate submission file


test['Fare'].fillna(train['Fare'].mean(), inplace = True)
predictions = rf14.predict(test.iloc[:, 1:])
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv('y_test14.csv', sep=",", index = False)

print 'cross_val_score...'
scores = cross_val_score(rf14, train.iloc[:, 2:], train.iloc[:, 1],scoring=my_scorer,cv=5)

# Leaderboard score: 0.82775
# 150 out of 6048