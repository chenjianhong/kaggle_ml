# coding: utf-8
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('train.csv', header=0)

# female = 0, Male = 1
# map遍历某列的每一个数据,使用一个函数或者一个字典映射对象
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# 使用众数填补缺失项
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # 为每个舱室分配id
Ports_dict = { name : i for i, name in Ports }
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

# 使用中位数填充age
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

# 删除暂时不需要的列
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)


test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe


test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values

test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

ids = test_df['PassengerId'].values

test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)


train_data = train_df.values
test_data = test_df.values


print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)


predictions_file = open("my_rf.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'