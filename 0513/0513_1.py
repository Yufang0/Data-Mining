# -*- coding: utf-8 -*-
"""
Created on Tue May 13 10:51:01 2025

@author: 411422259
topic: Reatil.csv -basket analysis
"""

import pandas as pd

df = pd.read_csv('Retail.csv')

df['Description'] = df['Description'].str.strip()

df.dropna(axis=0,subset=["InvoiceNo"],inplace=True)

#發票本來都是數字，但發現退貨包含C，所以要轉成文字處理
df["InvoiceNo"] = df["InvoiceNo"].astype("str")

#發票裡面包含C這個字母的是退貨
df = df[~df["InvoiceNo"].str.contains("C")]


#可以取出某一個欄位的值合併 Ex:Country欄位=France
#groupby()使用了發票編號和產品名稱合併，並且把數量加總
#unstack()可以把發票編號變成row, 產品名稱變成column
#reset_index().fillna(0) 是指把資料原本是nan的改成0填入
#set_index("InvoiceNo) 是指原本有多一個欄位index, 這個指令可以用發票編號取代為index
basket = (df[df["Country"]=="France"].groupby(["InvoiceNo","Description"]))["Quantity"].sum().unstack().reset_index().fillna(0).set_index("InvoiceNo")

#數值矩陣改成布林矩陣
def encode_unit(x):
    if x<=0:
        return 0
    else:
        return 1

#把資料表basket 套用(applymap)在函式(encode_unit)的方式，改成Boolean值矩陣表示
basket_set = basket.applymap(encode_unit)

#把郵資刪除
basket_set.drop("POSTAGE", inplace=True, axis=1)

#以下開始關聯法則
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

frequent_itemset = apriori(basket_set, min_support = 0.07, use_colnames=True)
rules = association_rules(frequent_itemset, metric='lift',min_threshold=1.1)
