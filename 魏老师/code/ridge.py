from sklearn import datasets, linear_model
import numpy as np
from sklearn.metrics import mean_squared_error

#diabetes = datasets.load_diabetes()
diabetes = datasets.load_boston()
X=diabetes.data
Y=diabetes.target
print(X.shape)
print(Y.shape)

X=np.hstack((np.ones((X.shape[0], 1)), X))

X_train = X[:300, :]
X_test = X[300:,:]
Y_train = Y[:300]
Y_test = Y[300:]
clf = linear_model.Ridge(alpha=0.1)
#clf = linear_model.Lasso(alpha=0.05)
#clf = linear_model.LinearRegression()

clf.fit(X_train, Y_train)
print(clf.coef_)
print(clf.intercept_)

beta = clf.coef_
beta[0] = clf.intercept_
prediction = X_test.dot(beta)
print("MSE on test set:", mean_squared_error(Y_test, prediction))