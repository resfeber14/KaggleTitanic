# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:10:03 2020

@author: SHWETA KUMARI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
train=pd.read_csv(r"C:\Users\SHWETA KUMARI\Desktop\aa\Titanic\train.csv")
#test=pd.read_csv(r"C:\Users\SHWETA KUMARI\Desktop\aa\Titanic\test.csv")
X=train.iloc[ : ,[0,2,4,5,6,7]]
Y=train.iloc[ : , 1:2]

#Missing Values
X=X.fillna(X.mean())

#Dummy Variables
X=pd.get_dummies(X)

#Splitting the Dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#Fitting the dataset
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train.values.ravel())

#Predicting the Results
Y_pred=classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=Y_train,cv=10)
accuracies.mean()
