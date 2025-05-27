# -*- coding: utf-8 -*-
"""
Created on Tue May 27 11:38:58 2025

@author: 411422259
@topic: Kmeans
"""

import pandas as pd
bank=pd.read_csv("bank-data-clustering.csv")
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

from sklearn.preprocessing import scale

X = pd.DataFrame([scale(bank['age']), sex, scale(bank["income"]), married, children, car, save, current, mortgage]).T
X.columns = ["age","sex","income","married","children","car","save","current","mortgage"]

X_new = pd.concat([X,region],axis = 1)
y = bank["pep"]


from sklearn.cluster import KMeans
SSE = []

for i in range(10):
    kmeans = KMeans(n_clusters=i+1, init="k-means++",random_state=20250527)
    kmeans.fit(X_new)
    SSE.append(kmeans.inertia_)

print(SSE)

import matplotlib.pyplot as plt
plt.plot(range(1,11), SSE, marker = "o")
plt.xlabel("Number of clusters")
plt.ylabel("SSE")

kmeans = KMeans(n_clusters=4, init="k-means++",random_state=20250527)
kmeans.fit(X_new)
print(kmeans.inertia_)
print(kmeans.cluster_centers_)

#還沒debug
#考試SSE Kmeas++ or 不給