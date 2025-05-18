# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
Name: 蕭羽芳 411422259
Topic: Resampling 3 methods
"""

import pandas as pd
titanic = pd.read_csv('titanic-train(2).csv')

X = titanic.iloc[:,:-1]
y = titanic['Survived']

print(y.value_counts())

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=20250429)
X_resample, y_resample = rus.fit_resample(X, y)
print(y_resample.value_counts())

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=20250429)

X_resample2, y_resample2 = ros.fit_resample(X, y)
print(y_resample2.value_counts())

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=20250429)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

Sex = le.fit_transform(titanic['Sex'])
Cabin = le.fit_transform(titanic['Cabin'])
Embarked = le.fit_transform(titanic['Embarked'])

X2 = pd.DataFrame([titanic['PassengerId'],titanic['Pclass'],Sex,titanic['SibSp'],titanic['Parch'],titanic['Ticket'],titanic['Fare'],Cabin,Embarked]).T
X_resample3, y_resample3 = sm.fit_resample(X2, y)
print(y_resample3.value_counts())