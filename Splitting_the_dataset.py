# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:42:33 2020

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

#Fitting the Model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,Y_train.values.ravel())

#Predicting the test results
Y_pred=classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=Y_train,cv=10)
accuracies.mean()

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
col_names=['Pclass','Age','SibSp','Parch']
features = X[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
X[col_names]= features

#Fitting the Model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X,Y.values.ravel())
 
#Converting test
test_X=test.iloc[ : ,[0,1,3,4,5,6]]
test_X=pd.get_dummies(test_X)
test_X=test_X.fillna(test_X.mean())

#Predicting the Results
Y_pred=classifier.predict(test_X)"""






