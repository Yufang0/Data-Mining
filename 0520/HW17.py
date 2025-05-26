# -*- coding: utf-8 -*-
"""
Created on Tue May 27 01:55:09 2025

@author: 411422259 HW17
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

Cabin = ohe.fit_transform(titanic[["Cabin"]])
Cabin = pd.DataFrame(Cabin)
Cabin.columns = ohe.categories_[0]

Embarked = ohe.fit_transform(titanic[["Embarked"]])
Embarked = pd.DataFrame(Embarked)
Embarked.columns = ohe.categories_[0]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

Pclass = le.fit_transform(titanic['Pclass'])
SibSp = le.fit_transform(titanic['SibSp'])
Parch = le.fit_transform(titanic['Parch'])


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

Age = pd.DataFrame(ss.fit_transform(titanic[["Age"]]))
Age.columns = ['Age']

Ticket = pd.DataFrame(ss.fit_transform(titanic[["Ticket"]]))
Ticket.columns = ['Ticket']

Fare = pd.DataFrame(ss.fit_transform(titanic[["Fare"]]))
Fare.columns = ['Fare']

Pclass = pd.DataFrame(ss.fit_transform(titanic[["Pclass"]]))
Pclass.columns = ['Pclass']

SibSp = pd.DataFrame(ss.fit_transform(titanic[["SibSp"]]))
SibSp.columns = ['SibSp']

Parch = pd.DataFrame(ss.fit_transform(titanic[["Parch"]]))
Parch.columns = ['Parch']

#多個資料表，縱向合併axis=1
X = pd.concat([Pclass,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked],axis=1)

y = titanic['Survived']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=20250520)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)
print("訓練資料集的正確率 =",knn.score(X_train, y_train))

#k=幾最好
from sklearn.metrics import accuracy_score

acc = []

for i in range(1,481):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    #print("測試資料集 k =", i, " 的正確率 =", accuracy_score(y_test, y_pred))
    acc.append(accuracy_score(y_test, y_pred))
    
print("測試資料集最高的正確率 =", max(acc))

bestK = 0
for i in range(0,480):
    if acc[i] == max(acc):
        bestK = i+1

    
print("最佳的 k =", bestK, "測試資料集正確率 =",max(acc))