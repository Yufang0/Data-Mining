# -*- coding: utf-8 -*-
"""
Created on Tue May 20 11:40:36 2025

@author: 411422259
Topic: SVM
"""

#適合連續型資料


import pandas as pd

bank = pd.read_csv("bank-data(3).csv")


subdata = bank[(bank["region"] == "INNER_CITY")|(bank["region"] == "TOWN") | (bank["region"] == "RURAL")]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(subdata[["age","income"]],subdata["region"],test_size=0.2,random_state=20250520)

#SVM連目標變數都要連續型編碼(python套件關係，其實原理上可以不用)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

X_train_std = ss.fit_transform(X_train)

from sklearn.svm import LinearSVC

m = LinearSVC(C=0.1, dual=False, class_weight="balanced")
m.fit(X_train_std, y_train)

print("訓練資料集正確率=", m.score(X_train_std, y_train))

y_pred = m.predict(X_train_std)
print("分錯的有幾筆 =",(y_train != y_pred).sum())

#!超重要 測試正確率的算法，不能ss.fit(測試資料集) => 因為要套用訓練資料集的平均數和標準差
#下面的寫法才是正確的 直接transform 不用fit
X_test_std = ss.transform(X_test)

print("訓練資料集正確率 =",m.score(X_test_std, y_test))