# -*- coding: utf-8 -*-
"""
Created on Tue May 20 09:31:27 2025

@author: 411422259
Topic: Association Rules -data structure2 
"""

import pandas as pd


dataset = [["Milk","Onion","Nutmeg","Kidney Beans","Eggs","Yogurt"],
           ["Dill","Onion","Nutmeg","Kidney Beans","Eggs","Yogurt"],
           ["Milk","Apple","Kidney Beans","Eggs"],
           ["Milk","Unicorn","Corn","Kidney Beans","Yogurt"],
           ["Corn","Onion","Onion","Kidney Beans","Ice cream","Eggs"]]

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)
df2 = pd.DataFrame(te_array,columns=te.columns_)

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def encode_units(x):
    if x <= 0:
        return 0
    else:
        return 1
    
basket_set3 = df2.applymap(encode_units)

#超品品項集
frequent_items3 = apriori(basket_set3,min_support=0.6,use_colnames=True)
rules3 = association_rules(frequent_items3,metric="lift",min_threshold=1.1)