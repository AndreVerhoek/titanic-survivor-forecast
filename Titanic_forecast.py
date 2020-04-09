# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_csv('Data/train.csv',index_col = 'PassengerId')
print(df)