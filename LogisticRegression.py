import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
train=pd.read_csv(r"C:\Users\SHWETA KUMARI\Desktop\aa\Titanic\train.csv")
test=pd.read_csv(r"C:\Users\SHWETA KUMARI\Desktop\aa\Titanic\test.csv")
X=train.iloc[ : ,[0,2,4,5,6,7]]
Y=train.iloc[ : , 1:2]

#Missing Values
X=X.fillna(X.mean())

#Dummy Variables
X=pd.get_dummies(X)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
col_names=['Pclass','Age','SibSp','Parch']
features = X[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
X[col_names]= features"""


#Fitting the Model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X,Y.values.ravel())
 
#Converting test
test_X=test.iloc[ : ,[0,1,3,4,5,6]]
test_X=pd.get_dummies(test_X)
test_X=test_X.fillna(test_X.mean())

#Predicting the Results
Y_pred=classifier.predict(test_X)
