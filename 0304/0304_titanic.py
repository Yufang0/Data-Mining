# -*- coding: utf-8 -*-
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

#離散化
print(np.min(data["Age"])," , ",np.max(data["Age"]))
#print(pd.cut(data["Age"],bins=5).value_counts())
#data["Age"] = pd.cut(data["Age"],bins=3,labels=["0-16","17-32","33-48"])

data.info()