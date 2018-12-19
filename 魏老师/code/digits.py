from sklearn.datasets import load_digits
from time import time

digits = load_digits()

print(digits.target[0:100])
print(digits.target.shape)
print(digits.data.shape)
print(digits.data[0,:])

X=digits.data
Y=digits.target
X_train = X[0:1200, :]
print(X_train.shape)
X_test = X[1200:, :]
print(X_test.shape)
Y_train = Y[0:1200]
Y_test = Y[1200:]

import numpy as np
mnist=np.load('mnist_images.npy')
import matplotlib.pyplot as plt
plt.gray()
for i in range(8):
    for j in range(8):
        plt.subplot(8, 8, 8*i+j+1)
        plt.imshow(mnist[8*i + j,:,:])
plt.show()