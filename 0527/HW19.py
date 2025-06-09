# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 16:43:36 2025

@author: 411422259 HW19
"""

import pandas as pd
titanic = pd.read_csv('titanic-train(1).csv')

import numpy as np
titanic["Age"] = np.where(titanic["Age"].isnull(),
                       np.nanmedian(titanic["Age"]),titanic["Age"])

import statistics
titanic["Cabin"] = np.where(titanic["Cabin"].isnull(),
                       statistics.mode(titanic["Cabin"]),titanic["Cabin"])

#不需要標準化
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

Cabin = ohe.fit_transform(titanic[["Cabin"]])
Cabin = pd.DataFrame(Cabin)
Cabin.columns = ohe.categories_[0]

Embarked = ohe.fit_transform(titanic[["Embarked"]])
Embarked = pd.DataFrame(Embarked)
Embarked.columns = ohe.categories_[0]

#
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Sex=le.fit_transform(titanic["Sex"])
Pclass = le.fit_transform(titanic['Pclass'])
SibSp = le.fit_transform(titanic['SibSp'])
Parch = le.fit_transform(titanic['Parch'])

#多個資料表，縱向合併axis=1
X = pd.DataFrame([Pclass, Sex, titanic["Age"], SibSp, Parch, titanic["Ticket"], titanic["Fare"]]).T
X.columns = ["Pclass","Sex","Age","SibSp","Parch","Ticket","Fare"]
X_new = pd.concat([X, Cabin, Embarked],axis = 1)

y = titanic['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new,y,test_size=0.2,random_state=20250527)


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=150,max_depth=8,random_state=20250527)

RF.fit(X_train, y_train)
print("訓練的正確率 =",RF.score(X_train, y_train))
print("測試的正確率 =",RF.score(X_test, y_test))
