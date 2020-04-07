# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:42:33 2020

@author: SHWETA KUMARI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
X=train.iloc[ : ,[0,2,4,5,6,7]]
Y=train.iloc[ : , 1:2]



#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)
test=sc_X.fit_transform(test)

#Dummy Variables

from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[: ,2]=labelencoder_X.fit_transform(X[: ,2])
