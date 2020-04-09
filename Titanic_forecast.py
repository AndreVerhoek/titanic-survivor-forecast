# -*- coding: utf-8 -*-
# Common imports
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm
from scipy import stats
import math

df = pd.read_csv('Data/train.csv',index_col = 'PassengerId')
print(list(df.columns.values))

df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
df = df[['Survived','Sex','Age','Pclass','SibSp','Parch','Fare']]
df = df.dropna()

df['AgeRange'] = df['Age'].apply(lambda element: math.floor((element/10)))
print(pd.get_dummies(df['Sex']))


df2 = pd.get_dummies(df,columns = ['AgeRange'])
print(df2.head())

df3 = pd.get_dummies(df2,columns = ['Pclass'])
print(df3.head())

df3['Fmem'] = df3['SibSp'] + df3['Parch']
print(df3['Fmem'])

print(max(df3['Fmem']))

print(list(df3.columns.values))

X = df3[['Sex', 'Fare', 'AgeRange_0', 'AgeRange_1', 'AgeRange_2', 'AgeRange_3', 'AgeRange_4', 'AgeRange_5', 'AgeRange_6', 'AgeRange_7', 'AgeRange_8', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Fmem']]
y = df3[['Survived']]

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

df3['YoungTarts'] = df3['AgeRange_0']
print(df3['YoungTarts'])

df3['OldFarts'] = df3['AgeRange_4'] + df3['AgeRange_5'] + df3['AgeRange_6'] + df3['AgeRange_7']
print(df3['OldFarts'])

X = df3[['Sex', 'Fare', 'YoungTarts', 'OldFarts', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Fmem']]
y = df3[['Survived']]

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

dftest = pd.read_csv('Data/test.csv',index_col = 'PassengerId')
dftest['Sex'] = dftest['Sex'].map({'female': 1, 'male': 0})
dftest = dftest[['Sex','Age','Pclass','SibSp','Parch','Fare']]
dftest = dftest.dropna()
dftest['AgeRange'] = dftest['Age'].apply(lambda element: math.floor((element/10)))
dftest2 = pd.get_dummies(dftest,columns = ['AgeRange'])
dftest3 = pd.get_dummies(dftest2,columns = ['Pclass'])
dftest3['Fmem'] = dftest3['SibSp'] + dftest3['Parch']
dftest3['YoungTarts'] = dftest3['AgeRange_0']
dftest3['OldFarts'] = dftest3['AgeRange_4'] + dftest3['AgeRange_5'] + dftest3['AgeRange_6'] + dftest3['AgeRange_7']
X = dftest3[['Sex', 'Fare', 'YoungTarts', 'OldFarts', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Fmem']]
print(list(dftest3.columns.values))
print(est2.predict(X))
