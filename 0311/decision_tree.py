# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:48:44 2025

@author: 蕭羽芳
@topic: Decision Tree
"""
#路徑老師會給
import os
os.environ["PATH"]+=os.pathsep+"C:/Program Files/Graphviz/bin"

import pandas as pd

bank = pd.read_csv("#2-bank-data(1).csv")

bank.info()

#age, married有遺失值

import numpy as np

bank["age"] = np.where(bank["age"].isnull(),
                       np.nanmedian(bank["age"]),
                       bank["age"])

import statistics

bank["married"] = np.where(bank["married"].isnull(),
                           statistics.mode(bank["married"]),
                           bank["married"])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
sex = le.fit_transform(bank["sex"])
region = le.fit_transform(bank["region"])
married = le.fit_transform(bank["married"])
car = le.fit_transform(bank["car"])
save_act = le.fit_transform(bank["save_act"])
current_act = le.fit_transform(bank["current_act"])
mortgage = le.fit_transform(bank["mortgage"])

X = pd.DataFrame([bank["age"],sex,region,
                   bank["income"],married,
                   bank["children"],car,save_act,
                   current_act,mortgage]).T

X.columns = ["age","sex","region","income","married",
             "children","car","save_act","current_act","mortgage"]

y = bank["pep"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=20250311)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="gini",min_samples_leaf=2,min_samples_split=0.2)
clf.fit(X_train, y_train)

print("training Acc = ",format(clf.score(X_train, y_train),'.4f'))
print("testing Acc = ",format(clf.score(X_test, y_test),'.4f'))
print("Leaves of the tree = ",clf.get_n_leaves())
print("Depth of the tree = ",clf.get_depth())

print("==============================")

clf2 = DecisionTreeClassifier(criterion="gini",min_samples_leaf=2,min_samples_split=0.05)
clf2.fit(X_train, y_train)

print("training Acc = ",format(clf2.score(X_train, y_train),'.4f'))
print("testing Acc = ",format(clf2.score(X_test, y_test),'.4f'))
print("Leaves of the tree = ",clf2.get_n_leaves())
print("Depth of the tree = ",clf2.get_depth())


from sklearn import tree
import graphviz #考試前會灌好

tree_data = tree.export_graphviz(clf2,out_file=None,feature_names=X.columns,
                                 class_names=clf2.classes_,proportion=True,
                                 rounded=True)
graph = graphviz.Source(tree_data)
graph.format = "png"
graph.render("tree.gv",view=True)
