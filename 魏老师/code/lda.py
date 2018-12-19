import numpy as np
from scipy.stats import multivariate_normal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Linda:
    def __init__(self):
        pass

    def fit(self, X, Y):
        self.sample_size = X.shape[0]
        self.feature_dim = X.shape[1]
        self.num_classes = np.max(Y) + 1

        # compute the feature mean for each class
        mu = [None] * self.num_classes
        for i in range(self.num_classes):
            mu[i] = np.mean(X[Y==i, :], axis=0)
        self.mu = mu

        # compute the covariance matrix of the training sample
        var = np.zeros((self.feature_dim, self.feature_dim))
        for i in range(self.num_classes):
            diff = X[Y==i, :] - mu[i]
            mat = diff.transpose().dot(diff)
            var += mat
        self.var =var/(self.sample_size - self.num_classes)

        # compute prior probability for each label
        y_prob = [None] * self.num_classes
        for i in range(self.num_classes):
            y_prob[i] = sum(Y==i)/self.sample_size
        self.y_prob = y_prob

    def predict(self, X):
        proba = self.predict_proba(X)
        labels = np.argmax(proba, axis=1)
        print(labels)
        return labels

    def predict_proba(self, X):
        def posterior(x, k):
            numerator = self.y_prob[k] * multivariate_normal.pdf(x, mean=self.mu[k], cov=self.var)
            denominator = 0
            for i in range(self.num_classes):
                denominator += self.y_prob[i] * multivariate_normal.pdf(x, mean=self.mu[i], cov=self.var)
            return numerator/denominator
        proba = np.zeros((X.shape[0], self.num_classes))
        for i in range(X.shape[0]):
            for j in range(self.num_classes):
                proba[i, j] = posterior(X[i, :], j)
        print(proba)
        return proba



from sklearn import datasets

iris = datasets.load_iris()
X=iris.data
Y=iris.target
Y=np.expand_dims(Y, 1)  # original Y has shape (150,) now it becomes (150, 1)
data = np.hstack((Y, X))  # merge X and Y
np.random.seed(123)
np.random.shuffle(data)
X=data[:, 1:]
Y=data[:, 0].astype(int)
Y=np.squeeze(Y)
X_train = X[0:120, :]
X_test = X[120:, :]
Y_train = Y[0:120]
Y_test = Y[120:]

lda = Linda()
lda.fit(X_train, Y_train)
lda.predict(X_test)

classifier = LinearDiscriminantAnalysis()
classifier.fit(X_train, Y_train)
#print(classifier.predict(X_test))
#print(Y_test)
print(classifier.predict_proba(X_test))