# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
411422170吳沂珊
20250415
topic:Model evaluation HW
"""
import pandas as pd
zoo=pd.read_csv("zoo.csv")
zoo=zoo.drop(columns=["animal name"]) #drop多行的話可以逗點
X=zoo.drop(columns=["type","legs","eggs"])          #把目標變數拿掉命名為X
y=zoo["type"]                         # type 命為y

#樂觀法則 (正確率
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion="entropy",
                               min_samples_split=0.25,
                               min_samples_leaf=2,
                               random_state=20250415)
clf.fit(X,y)
print("Optimistic accuracy=" ,clf.score(X,y))


#悲觀法則(需要葉子數
print("leaves=",clf.get_n_leaves())
#需要知道分錯幾個
y_pred=clf.predict(X)
#y predict 跟y比對 要記期中考必考
cm=pd.crosstab(y,y_pred)
print(cm)
#發現少了第五類 且不是對稱矩陣 這樣不行 
print("正確個數(矩陣主對角相加)=",41+20+13+10)
print("錯誤個數=",101-84)
print("pessimistic accuracy=",1-(17+4*0.5)/101)
#(正確+個數*0.5)/總數

#hold-out 比例分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,
                                                  random_state=20250415)

clf.fit(X_train,y_train)
print("Hold-out 8:2 =",clf.score(X_test, y_test))


#交叉驗證
from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,X,y,cv=8, scoring="accuracy")#一定適用全部資料
print("8-cross validation=", scores.mean()) #要取平均




















