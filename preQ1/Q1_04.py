# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 02:50:09 2025

@author: User
"""

import pandas as pd
import numpy as np
import statistics
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("NBApoints.csv")
#inplace
data.drop(["Player"],axis=1,inplace=True)
data.drop(["Tm"],axis=1,inplace=True)

data["Pos"] = np.where(data["Pos"].isnull(),statistics.mode(data["Pos"]),data["Pos"])
data["Age"] = np.where(data["Age"].isnull(),np.nanmedian(data["Age"]),data["Age"])
data["STL"] = np.where(data["STL"].isnull(),np.nanmean(data["STL"]),data["STL"])



data["3P"] = pd.cut(data["3P"],bins=3,labels=["L","M","H"])
print(data["3P"].value_counts())
data["2P"] = pd.qcut(data["2P"],q=3,labels=["L","M","H"])
print(data["2P"].value_counts())

data["G"] = scale(data["G"])
print(np.mean(data["G"]))
print(np.std(data["G"]))

le = LabelEncoder()

data["Pos"] = le.fit_transform(data["Pos"])

data.to_csv("411422259.csv",index=False)

#X = data[""]
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=20240318)


