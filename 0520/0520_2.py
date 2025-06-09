# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:15:13 2025

@author: 411422259
Topic: KNN
"""

import pandas as pd

bank = pd.read_csv("bank-data(3).csv")
bank.info()

#實際三種不同類型以上才要用one hot encoder

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

region = ohe.fit_transform(bank[["region"]])
region = pd.DataFrame(region)
region.columns = ohe.categories_[0]

sex = ohe.fit_transform(bank[["sex"]])
sex = pd.DataFrame(sex)
sex.columns = ohe.categories_[0]

married = ohe.fit_transform(bank[["married"]])
married = pd.DataFrame(married)
married.columns = ["married_" + x for x in ohe.categories_[0]]

children = ohe.fit_transform(bank[["children"]])
children = pd.DataFrame(children)
children.columns = ["children_" + x for x in ohe.categories_[0]]

car = ohe.fit_transform(bank[["car"]])
car = pd.DataFrame(car)
car.columns = ["car_" + x for x in ohe.categories_[0]]

current_act = ohe.fit_transform(bank[["current_act"]])
current_act = pd.DataFrame(current_act)
current_act.columns = ["current_act_" + x for x in ohe.categories_[0]]

save_act = ohe.fit_transform(bank[["save_act"]])
save_act = pd.DataFrame(save_act)
save_act.columns = ["save_act_" + x for x in ohe.categories_[0]]

mortgage = ohe.fit_transform(bank[["mortgage"]])
mortgage = pd.DataFrame(mortgage)
mortgage.columns = ["mortgage_" + x for x in ohe.categories_[0]]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
#age = ss.fit_transform(bank[["age"]])
ss.fit(bank[["age"]])
age = pd.DataFrame(ss.transform(bank[["age"]]))
age.columns = ['age']

income = pd.DataFrame(ss.fit_transform(bank[["income"]]))
income.columns = ['income']

#多個資料表，縱向合併axis=1
X = pd.concat([age,sex,region,income,married,children,car,save_act,current_act,mortgage],axis=1)

y = bank['pep']

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