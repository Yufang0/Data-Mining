# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 03:53:03 2025

@author: User
"""
import pandas as pd

data = pd.read_csv('zoo.csv')
data.drop(['animal name'],axis=1,inplace=True)

X = data.drop(['type'],axis=1)
y = data['type']

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='gini',min_samples_leaf=2,min_samples_split=0.1,random_state=20250415)

clf.fit(X,y)
print("Optimistic Acc",clf.score(X,y))

print('leaf',clf.get_n_leaves())
y_pred = clf.predict(X)
cm = pd.crosstab(y,y_pred)
print(cm)

print("Pessimstic Acc",1-(6+7*0.5)/101)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=20250415)

clf.fit(X_train,y_train)
print("Hold out 7:3",clf.score(X_test, y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=5, scoring=('accuracy'))
print(scores.mean())