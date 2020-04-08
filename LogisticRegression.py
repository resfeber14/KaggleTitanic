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
