# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 00:00:48 2025

@author: 411422259 蕭羽芳
Topic: HW10
"""

import pandas as pd
data = pd.read_csv("titanic-train(1).csv")

#刪掉欄位
data.drop(["PassengerId"],axis=1,inplace=True)

#轉換型態
import numpy as np
import statistics
data["Cabin"] = np.where(data["Cabin"].isnull(),
                       statistics.mode(data["Cabin"]),data["Cabin"])
data["Age"] = np.where(data["Age"].isnull(),
                       np.nanmean(data["Age"]),data["Age"])



"""
Created on Tue Mar  4 11:46:53 2025

@author: 411422259 蕭羽芳
Topic: titanic
"""

import pandas as pd
data = pd.read_csv("titanic-train(1).csv")

#刪掉欄位
data.drop(["PassengerId"],axis=1,inplace=True)

#轉換型態
import numpy as np
import statistics
data["Cabin"] = np.where(data["Cabin"].isnull(),
                       statistics.mode(data["Cabin"]),data["Cabin"])
data["Age"] = np.where(data["Age"].isnull(),
                       np.nanmean(data["Age"]),data["Age"])

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Sex=le.fit_transform(data["Sex"])
Cabin=le.fit_transform(data["Cabin"])
Embarked=le.fit_transform(data["Embarked"])

X=pd.DataFrame([data["Pclass"],Sex,data["Age"],
                data["SibSp"],data["Parch"],data["Ticket"],
                data["Fare"],Cabin,Embarked]).T

X.columns=["Pclass","Sex","Age","SibSp", "Parch",
           "Ticket","Fare","Cabin","Embarked"]

y = data["Survived"]


#Use chi-square test to choose the top 5 best features
from sklearn.feature_selection import SelectKBest, chi2

sk = SelectKBest(chi2,k=5)
sk.fit(X,y)
print("The feature selection by Chi2=",sk.get_feature_names_out())

X_new1 = sk.transform(X)

from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier(criterion="entropy",min_samples_split=0.05,
                              min_samples_leaf=2,random_state=20250408)

clf1.fit(X_new1,y)
print("卡方分配法挑選的變數正確率",clf1.score(X_new1,y))
print("卡方分配法挑選的變數重要性",clf1.feature_importances_)


#Use feature importance to choose the top 5 best features 
clf1.fit(X,y)
print("全變數正確率",clf1.score(X,y))
print("全變數中的模型挑選的變數重要性",clf1.feature_importances_)
#建構變數挑選重要性的資料框feature

feature = pd.DataFrame([X.columns,clf1.feature_importances_]).T
X_new2 = pd.DataFrame([X["Sex"],X["Pclass"],X["Ticket"],X["Fare"],X["Cabin"]]).T
#建構變數挑選重要性top5資料框feature
                       
clf1.fit(X_new2, y)
print("模型挑選變數top5的正確率= ",clf1.score(X_new2, y))


#使用交叉驗證:多次平均評估兩種變數挑選方法優劣
from sklearn.model_selection import cross_val_score
cross = cross_val_score(clf1, X_new1, y,cv=10,scoring="accuracy")
print("卡方分配法挑出的變數cv=10的模型正確率",cross.mean())
cross2 = cross_val_score(clf1, X_new2, y,cv=10,scoring="accuracy")
print("卡方分配法挑出的變數cv=10的模型正確率",cross2.mean())


'''
#離散化
print(np.min(data["Age"])," , ",np.max(data["Age"]))
#print(pd.cut(data["Age"],bins=5).value_counts())
data["Age"] = pd.cut(data["Age"],bins=5,labels=["0-16","17-32","33-48","49-64","64-80"])

#print(pd.qcut(data["Fare"],q=3,labels=["L","M","H"]).value_counts())
data["Fare"] = pd.qcut(data["Fare"],q=3,labels=["L","M","H"])

#轉換資料型態
data["SibSp"] = data["SibSp"].astype(str)

data.info()

data.to_csv("411422259.csv",index=False)'''