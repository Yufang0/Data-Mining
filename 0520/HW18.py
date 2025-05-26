# -*- coding: utf-8 -*-
"""
Created on Tue May 27 02:22:43 2025

@author: 411422259 HW18
"""

import pandas as pd
titanic = pd.read_csv('titanic-train(2).csv')

import numpy as np
titanic["Age"] = np.where(titanic["Age"].isnull(),
                       np.nanmedian(titanic["Age"]),titanic["Age"])

import statistics
titanic["Cabin"] = np.where(titanic["Cabin"].isnull(),
                       statistics.mode(titanic["Cabin"]),titanic["Cabin"])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

Sex = ohe.fit_transform(titanic[["Sex"]])
Sex = pd.DataFrame(Sex)
Sex.columns = ohe.categories_[0]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

Age = pd.DataFrame(ss.fit_transform(titanic[["Age"]]))
Age.columns = ['Age']

Fare = pd.DataFrame(ss.fit_transform(titanic[["Fare"]]))
Fare.columns = ['Fare']

X = pd.concat([Sex,Age,Fare],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,titanic["Survived"],test_size=0.2,random_state=20250520)

from sklearn.svm import LinearSVC
m = LinearSVC(C=0.8, dual=False, class_weight="balanced")
m.fit(X_train, y_train)

print("訓練資料集正確率=", m.score(X_train, y_train))

y_pred = m.predict(X_train)
print("分錯的有幾筆 =",(y_train != y_pred).sum())

#!超重要 測試正確率的算法，不能ss.fit(測試資料集) => 因為要套用訓練資料集的平均數和標準差
#下面的寫法才是正確的 直接transform 不用fit
print("訓練資料集正確率 =",m.score(X_test, y_test))