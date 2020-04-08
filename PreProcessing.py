# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:42:33 2020

@author: SHWETA KUMARI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
X=train.iloc[ : ,[0,2,4,5,6,7]]
Y=train.iloc[ : , 1:2]

#Missing Values
X=X.fillna(X.mean())

#Dummy Variables
X=pd.get_dummies(X)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
col_names=['Pclass','Age','SibSp','Parch']
features = X[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
X[col_names]= features
