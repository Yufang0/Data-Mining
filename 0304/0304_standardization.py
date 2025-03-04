# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:14:38 2025

@author: 411422259 蕭羽芳
Topic: standardization
"""

import numpy as np
from sklearn.preprocessing import scale

data = [1,2,3,4,5,6,7,8,9,10]
average = np.mean(data)
print(average)

std = np.std(data)
print(std)

print(data-average)
print(scale(data,with_std=False))

print((data-average)/std)
print(scale(data))

print("===========以下是二維資料==========")
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

data2 = [[1,2,3,4,5],[6,7,8,9,10]]

#要先丟套件裡，才能用所有相關的值，跟一維的套件比，他可以記錄值一步就出來了
ss.fit(data2)
print("平均數",ss.mean_)
print("標準差",ss.var_)
print(ss.transform(data2))
print((data2-ss.mean_)/np.sqrt(ss.var_))