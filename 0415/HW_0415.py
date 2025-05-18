# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 01:45:02 2025

@author: 411422259 蕭羽芳
"""

import pandas as pd
zoo = pd.read_csv("zoo.csv")
zoo = zoo.drop(columns=["animal name"])
X = zoo.drop(columns=['type','legs','eggs'])
y = zoo['type']

#樂觀法則 (正確率 M1
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion="entropy",
                               min_samples_split=0.1, #事前修剪法C,M
                               min_samples_leaf=2,
                               random_state=20250415)
clf.fit(X,y)
print("Optimistic accuracy=" ,clf.score(X,y))

#悲觀法則(需要葉子數 程式碼要背 M2
print("leaves=",clf.get_n_leaves())
#需要知道分錯幾個
y_pred=clf.predict(X)
#y predict 跟y比對 要記期中考必考 含混矩陣
cm=pd.crosstab(y,y_pred)
print(cm)
#發現少了第五類 且不是對稱矩陣 這樣不行 
print("正確個數(矩陣主對角相加)=",41+20+5+13+8+8)
print("錯誤個數=",101-95)
print("pessimistic accuracy=",1-(6+7*0.5)/101)
#錯誤率=(錯誤+葉子個數*0.5)/總數
#正確率1-錯誤率

#hold-out 比例分割 M3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,
                                                  random_state=20250415)

clf.fit(X_train,y_train)
print("Hold-out 8:2 =",clf.score(X_test, y_test))


#交叉驗證 資料一定是全部
from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,X,y,cv=8, scoring="accuracy")#一定適用全部資料
print("5-cross validation=", scores)
print("5-cross validation=", scores.mean()) #要取平均





















