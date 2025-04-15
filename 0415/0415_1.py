# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 10:27:16 2025

@author: 411422259
"""

import pandas as pd
zoo = pd.read_csv("zoo.csv")
zoo = zoo.drop(columns=["animal name"])
X = zoo.drop(columns=['type'])
y = zoo['type']

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=0.1,min_samples_leaf=2,random_state=20250415)

clf.fit(X,y)
print("Optimastic acc= ",clf.score(X, y))

print("leaves= ",clf.get_n_leaves())

y_pred = clf.predict(X)

cm = pd.crosstab(y, y_pred)
print(cm)
print("正確個數= ",41+20+5+13+8+8)
print("錯誤個數= ",101-95)
print("Pessimistic acc= ",1-(6+7*0.5))