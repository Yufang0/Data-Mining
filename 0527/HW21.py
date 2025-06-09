# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 23:02:59 2025

@author: User
"""

import pandas as pd
titanic = pd.read_csv('2019-資料採礦-#中11-titanic-train.csv')

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

Cabin = pd.DataFrame(ohe.fit_transform(titanic[["Cabin"]]))
Cabin.columns = ohe.categories_[0]

Embarked = pd.DataFrame(ohe.fit_transform(titanic[["Embarked"]]))
Embarked.columns = ohe.categories_[0]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

Sex = le.fit_transform(titanic['Sex'])
Pclass = le.fit_transform(titanic['Pclass'])
SibSp = le.fit_transform(titanic['SibSp'])
Parch = le.fit_transform(titanic['Parch'])

from sklearn.preprocessing import scale

#多個資料表，縱向合併axis=1
X = pd.DataFrame([scale(titanic["Pclass"]),Sex,scale(titanic["Age"]),SibSp,Parch,scale(titanic["Ticket"]),scale(titanic["Fare"])]).T
X.columns = ["Pclass","Sex","Age","SibSp","Parch","Ticket","Fare"]
X_new = pd.concat([X, Cabin, Embarked],axis = 1)

y = titanic['Survived']

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, n_init=10, init="k-means++", random_state=20241228)
kmeans.fit(X_new)

centrid = pd.DataFrame(kmeans.cluster_centers_, columns=X_new.columns)
X_pred = kmeans.predict(X_new)

print(pd.crosstab(y, X_pred))
print("分群結果正確率:", (127+188)/470)

SSE = []
for i in range(10):
    kmeans = KMeans(n_clusters=i+1,n_init=10, init="k-means++",random_state=20241228)
    kmeans.fit(X_new)
    SSE.append(kmeans.inertia_)

print(SSE)

import matplotlib.pyplot as plt
plt.plot(range(1,11), SSE, marker = "o")
plt.xlabel("Number of clusters")
plt.ylabel("SSE")

kmeans1 = KMeans(n_clusters=4, n_init=10, init="k-means++", random_state=20241228)
kmeans1.fit(X_new)
print(kmeans1.inertia_)
print(kmeans1.cluster_centers_)
centrid1 = pd.DataFrame(kmeans1.cluster_centers_, columns=X_new.columns)
X_pred1 = kmeans1.predict(X_new)

print(pd.crosstab(y, X_pred1))
print("分群結果正確率:", (103+15+24+173)/470)