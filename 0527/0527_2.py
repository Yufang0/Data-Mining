# -*- coding: utf-8 -*-
"""
Created on Tue May 27 10:23:05 2025

@author: 411422259
@topic: Voting
"""

import pandas as pd
bank=pd.read_csv("bank-data(3).csv")
#X = bank[["age","income"]]
#y = bank[["pep"]]

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
sex=le.fit_transform(bank["sex"])
married=le.fit_transform(bank["married"])
car=le.fit_transform(bank["car"])
children=le.fit_transform(bank["children"])
save=le.fit_transform(bank["save_act"])
current=le.fit_transform(bank["current_act"])
mortgage=le.fit_transform(bank["mortgage"])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
region = pd.DataFrame(ohe.fit_transform(bank[["region"]]))
region.columns = ohe.categories_[0]

X = pd.DataFrame([bank['age'], sex, bank["income"], married, children, car, save, current, mortgage]).T
X.columns = ["age","sex","income","married","children","car","save","current","mortgage"]

X = pd.concat([X,region],axis = 1)
y = bank["pep"]


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=20250527)

#如果有標準化的話，一定要先切資料集，模擬測試是外來的資料集，必須要配合訓練資料transform
#所以X_test，不可以自己fit標準化，只能transorm X_train的平均數和標準差

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
