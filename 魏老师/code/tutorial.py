from sklearn import datasets, linear_model
import numpy as np
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
X=diabetes.data
Y=diabetes.target

X=np.hstack((np.ones((X.shape[0], 1)), X))
X_train = X[:300, :]
X_test = X[300:,:]
Y_train = Y[:300]
Y_test = Y[300:]

def compute_beta(X, Y):
    Xt = X.transpose()
    P = Xt.dot(X)
    Pinv = np.linalg.inv(P)
    beta = Pinv.dot(Xt).dot(Y)
    return beta

def compute_error(X, Y, beta):
    pred = X.dot(beta)
    diff = Y - pred
    norm = np.linalg.norm(diff)
    MSE = norm**2/Y.shape[0]
    return MSE

beta = compute_beta(X_train, Y_train)
print(beta)
error = compute_error(X_test, Y_test, beta)
print(error)

# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
print(regr.coef_)
print(regr.intercept_)
diabetes_y_pred = regr.predict(X_test)
print(mean_squared_error(Y_test, diabetes_y_pred))

class Linalg:
    def __init__(self):
        pass

    def fit(self, X, Y):
        beta = compute_beta(X, Y)
        self.beta=beta
        return beta


    def predict(self, X):
        return X.dot(self.beta)


    def error(self, X, Y):
        return compute_error(X, Y, self.beta)


reg = Linalg()
print(reg.fit(X_train, Y_train))
print(reg.error(X_test, Y_test))