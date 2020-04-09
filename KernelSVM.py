# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:43:50 2020

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


#Fitting the Dataset
from sklearn.svm import SVC
classifier=SVC(kernel='linear')
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

"""from sklearn.model_selection import GridSearchCV
parameters=[{'penalty':[ 'l1','l2'],'tol':[1e-4,1e-3,1e-2,1e-1,1.0,10,100,1000],'C':[0.5,0.1,0.01,
             0.001,0.0001]}]
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
grid_search=grid_search.fit(X_train,Y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_

#Converting test
test_X=test.iloc[ : ,[0,1,3,4,5,6]]
test_X=pd.get_dummies(test_X)
test_X=test_X.fillna(test_X.mean())"""
