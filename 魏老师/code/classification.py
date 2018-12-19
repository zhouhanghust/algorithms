from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

iris = datasets.load_iris()

X=iris.data
Y=iris.target
print(X)
print(Y)

Y=np.expand_dims(Y, 1)  # original Y has shape (150,) now it becomes (150, 1)
print(X.shape, Y.shape)
data = np.hstack((Y, X))  # merge X and Y

np.random.seed(123)
np.random.shuffle(data)
print(data)  # now data has been shuffled

X=data[:, 1:]
Y=data[:, 0].astype(int)
Y=np.squeeze(Y)  # squeeze dimension of Y

X_train = X[0:120, :]
X_test = X[120:, :]
Y_train = Y[0:120]
Y_test = Y[120:]

# classifier = KNeighborsClassifier(n_neighbors=1)
#classifier = LogisticRegression()
classifier = LinearDiscriminantAnalysis()

classifier.fit(X_train, Y_train)
print(classifier.predict(X_test))
print(Y_test)
print(classifier.predict_proba(X_test))