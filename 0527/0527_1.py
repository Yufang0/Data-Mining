# -*- coding: utf-8 -*-
"""
Created on Tue May 27 09:36:05 2025

@author: 411422259
@Topic: Random Forest
"""

import pandas as pd
bank=pd.read_csv("bank-data(3).csv")
#bank.info()
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

X_new = pd.concat([X,region],axis = 1)
y = bank["pep"]

#DT、RF、KNN 不用編碼
#SVM 要編碼

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=200,max_depth=8,random_state=20250527)

RF.fit(X_new, y)
print("隨機森林的正確率 =",RF.score(X_new, y))