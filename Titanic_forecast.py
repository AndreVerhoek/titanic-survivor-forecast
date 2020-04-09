# -*- coding: utf-8 -*-
# Common imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('Data/train.csv',index_col = 'PassengerId')
print(list(df.columns.values))

X = df[['Sex','Age','SibSp','Parch','Embarked']]
y = df[['Survived']]

tree_clf_entropy = DecisionTreeClassifier(max_depth=2, random_state=42, criterion='entropy')
tree_clf_entropy.fit(X,y)