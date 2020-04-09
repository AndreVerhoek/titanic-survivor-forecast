# -*- coding: utf-8 -*-
# Common imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('Data/train.csv',index_col = 'PassengerId')
print(list(df.columns.values))

df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
df = df.dropna()

X = df[['Sex','Age']]
y = df[['Survived']]

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_)
print(lin_reg.coef_)