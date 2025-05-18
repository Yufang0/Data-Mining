"""
# -*- coding: utf-8 -*

Created on Tue Mar  4 09:55:53 2025

@author: 411422259 蕭羽芳
Topic: data preprocessing
"""

import pandas as pd

bank = pd.read_csv("#2-bank-data(1).csv")

#有遺失值，直接不要
bank1 = bank.dropna()

#刪掉id欄位
bank.drop(["id"],axis=1,inplace=True)

#查欄位
#bank.info()

import numpy as np

#中位數補缺失值(中位數不包含遺失值
print("median of age",np.nanmedian(bank["age"]))
bank["age"] = np.where(bank["age"].isnull(),
                       np.nanmedian(bank["age"]),bank["age"])

#平均值
print("mean of income",np.nanmean(bank["income"]))
bank["income"] = np.where(bank["income"].isnull(),
                       np.nanmean(bank["income"]),bank["income"])

#眾數
import statistics
print("mode of maried",statistics.mode(bank["married"]))
bank["married"] = np.where(bank["married"].isnull(),
                       statistics.mode(bank["married"]),bank["married"])


#離散化，值平均分
#print(pd.cut(bank["age"],bins=3).value_counts())

#最小值，最大值
print(np.min(bank["age"])," , ",np.max(bank["age"]))
print(pd.cut(bank["age"],bins=3).value_counts())
#bank["age"] = pd.cut(bank["age"],bins=3,labels=["18-34","35-50","51-67"])

#等次數切
print(pd.qcut(bank["income"],q=3,labels=["L","M","H"]).value_counts())
bank["income"] = pd.qcut(bank["income"],q=3,labels=["L","M","H"])

#轉換資料型態
bank["children"] = bank["children"].astype(str)


bank.to_csv("411422259_pre.csv",index=False)