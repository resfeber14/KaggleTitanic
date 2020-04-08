#Fitting the Model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X,Y)

#Predicting the Results
Y_pred=classifier.predict(test)
