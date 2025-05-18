# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 03:09:13 2025

@author: User
"""

import pandas as pd


data = pd.read_csv('titanic-train(1).csv')
data.drop(['PassengerId'],axis=1,inplace=True)

import numpy as np
import statistics
data['Cabin'] = np.where(data['Cabin'].isnull(),statistics.mode(data['Cabin']),data['Cabin'])
data['Age'] = np.where(data['Age'].isnull(),np.nanmean(data['Age']),data['Age'])
#data.info()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Sex = le.fit_transform(data['Sex'])
Cabin = le.fit_transform(data['Cabin'])
Embarked = le.fit_transform(data['Embarked'])

X = pd.DataFrame([data['Pclass'],Sex,data['Age'],data['SibSp'],data['Parch'],data['Ticket'],data['Fare'],Cabin,Embarked]).T
#記得
X.columns=["Pclass","Sex","Age","SibSp", "Parch",
           "Ticket","Fare","Cabin","Embarked"]
y = data['Survived']

from sklearn.feature_selection import SelectKBest, chi2

sk = SelectKBest(chi2, k=5)
sk.fit(X,y)
print("卡方前五重要",sk.get_feature_names_out())
X_new1 = sk.transform(X)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=0.05,min_samples_leaf=2,random_state=20250408)

clf.fit(X_new1,y)
print("卡方正確率",clf.score(X_new1, y))
print("卡方重要程度",clf.feature_importances_)

clf.fit(X,y)
feature = pd.DataFrame([X.columns,clf.feature_importances_]).T
X_new2 = pd.DataFrame([X["Sex"],X["Pclass"],X["Ticket"],X["Fare"],X["Cabin"]]).T

clf.fit(X_new2,y)
print('全變數正確率',clf.score(X_new2,y))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_new1, y, cv=10, scoring='accuracy')
print(scores.mean())

scores2 = cross_val_score(clf, X_new2, y, cv=10, scoring='accuracy')
print(scores2.mean())
