from sklearn.tree import DecisionTreeClassifier
import numpy as np

from sklearn.datasets import load_digits

digits = load_digits()

X=digits.data
Y=digits.target
X_train = X[0:1200, :]
X_test = X[1200:, :]
Y_train = Y[0:1200]
Y_test = Y[1200:]

classifier = DecisionTreeClassifier(max_depth=20,
                                    min_samples_split=5,
                                    #random_state=123,
                                    max_features=None)

classifier.fit(X_train, Y_train)
prediction = classifier.predict(X_test)
match = sum(prediction == Y_test)
print("Classification rate: {0}/{1} ({2:.1f}%)".format(match, Y_test.shape[0], match/Y_test.shape[0]*100))
