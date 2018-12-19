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

def tune():
    multiple = np.arange(0, 12)
    alpha = np.power(2, multiple) * 0.01
    print(alpha)
    mse = np.zeros(12)
    for i in range(12):
        a = alpha[i]
        lr = linear_model.Lasso(a)
        lr.fit(X_train, Y_train)
        diabetes_y_pred = lr.predict(X_test)
        mse[i] = mean_squared_error(Y_test, diabetes_y_pred)
        print("Alpha: {0:.3f}   MSE: {1:.3f}".format(a, mse[i]))

tune()