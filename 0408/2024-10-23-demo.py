# -*- coding: utf-8 -*-
"""
Spyder Editor

author:蕭羽芳
Date:2025-04-08
Topic: details of Decision Tree
"""


import pandas as pd
bank=pd.read_csv("#5-bank-data.csv")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
sex=le.fit_transform(bank["sex"])
region=le.fit_transform(bank["region"])
married=le.fit_transform(bank["married"])
children=le.fit_transform(bank["children"])
car=le.fit_transform(bank["car"])
save=le.fit_transform(bank["save_act"])
current=le.fit_transform(bank["current_act"])
mortgage=le.fit_transform(bank["mortgage"])

X=pd.DataFrame([bank["age"],sex,region,bank["income"],
                married,children,car,save,current,mortgage]).T
X.columns=["age","sex","region","income", "married","children"
           ,"car","save","current","mortgage"]

y=bank["pep"]

from sklearn.feature_selection import SelectKBest, chi2

sk = SelectKBest(chi2,k=5)
sk.fit(X,y)
print("The feature selection by Chi2=",sk.get_feature_names_out())

X_new1 = sk.transform(X)

from sklearn.tree import DecisionTreeClassifier
clf1 = DecisionTreeClassifier(criterion="gini",min_samples_split=0.05,
                              min_samples_leaf=2,random_state=20250408)

clf1.fit(X_new1,y)
print("卡方分配法挑選的變數正確率",clf1.score(X_new1,y))
print("卡方分配法挑選的變數重要性",clf1.feature_importances_)

clf2 = DecisionTreeClassifier(criterion="gini",min_samples_split=0.05,
                              min_samples_leaf=2,random_state=20250408)
clf2.fit(X,y)
print("全變數正確率",clf2.score(X,y))
print("全變數中的模型挑選的變數重要性",clf2.feature_importances_)
#建構變數挑選重要性的資料框feature

feature = pd.DataFrame([X.columns,clf2.feature_importances_]).T
X_new2 = pd.DataFrame([X["income"],X["children"],X["mortgage"],X["save"],X["age"]]).T
#建構變數挑選重要性top5資料框feature
                       
clf2.fit(X_new2, y)
print("模型挑選變數top5的正確率= ",clf2.score(X_new2, y))
#期中考必考

#使用交叉驗證:多次平均評估兩種變數挑選方法優劣
from sklearn.model_selection import cross_val_score
cross = cross_val_score(clf2, X_new1, y,cv=10,scoring="accuracy")
print("卡方分配法挑出的變數cv=10的模型正確率",cross.mean())
cross2 = cross_val_score(clf2, X_new2, y,cv=10,scoring="accuracy")
print("卡方分配法挑出的變數cv=10的模型正確率",cross2.mean())


"""
from sklearn.feature_selection import SelectFromModel
sm = SelectFromModel(clf2,max_features=5)
sm.fit(X, y)
print("全變數中的模型挑選的前四強變數",sm.get_feature_names_out())
因為只能挑出錢四強所以不公平，無法直接比較
"""

"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,
                                                  random_state=20240408)

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion="entropy",min_samples_split=0.25,
                           min_samples_leaf=2)

clf.fit(X_train,y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))
"""











