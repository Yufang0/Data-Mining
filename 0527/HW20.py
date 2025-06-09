# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 17:19:02 2025

@author: 411422259 HW20
"""

import pandas as pd
train = pd.read_csv('titanic-train(1)_20.csv')
test = pd.read_csv('#5-titanic-test.csv')

import numpy as np
train["Age"] = np.where(train["Age"].isnull(),
                       np.nanmedian(train["Age"]),train["Age"])

test["Age"] = np.where(test["Age"].isnull(),
                       np.nanmedian(test["Age"]),test["Age"])

#多個資料表，縱向合併axis=1
X_train = pd.DataFrame([train["Age"], train["Fare"]]).T
y_train = train['Survived']

X_test = pd.DataFrame([test["Age"], test["Fare"]]).T
y_test = test['Survived']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
std_X_train = ss.fit_transform(X_train)
std_X_test = ss.transform(X_test)

#決策樹和隨機森林都不需要標準化，目標變數也可以名目尺度
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100,random_state=20250527, max_depth=5)

RF.fit(X_train, y_train)
print("訓練的正確率 =",RF.score(X_train, y_train))
print("測試的正確率 =",RF.score(X_test, y_test))

#KNN因為是算相似度，所以變數間要標準化，差異才不會過大
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(std_X_train, y_train)
print("訓練的正確率 =",knn.score(std_X_train, y_train))
print("測試的正確率 =",knn.score(std_X_test, y_test))

#SVM系列X,y都必須是數值型，所以連Y都要轉編碼，要標準化
from sklearn.svm import SVC #非線性
svm = SVC(gamma=0.1,kernel="rbf",probability=True)

svm.fit(std_X_train, y_train)
print("訓練的正確率 =",svm.score(std_X_train, y_train))
print("測試的正確率 =",svm.score(std_X_test, y_test))

#Voting 是大家一起比，所以要配合所有人的規格
from sklearn.ensemble import VotingClassifier
vot = VotingClassifier(estimators=[("RF",RF),("KNN",knn),("SVM",svm)],voting="soft",n_jobs=-1)

vot.fit(std_X_train, y_train)
print("訓練的正確率 =",vot.score(std_X_train, y_train))
print("測試的正確率 =",vot.score(std_X_test, y_test))
