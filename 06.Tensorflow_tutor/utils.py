import numpy as np


def load_data():
    images = np.load('mnist_images.npy')
    labels = np.load('mnist_labels.npy')
    sample_size = labels.shape[0]
    data = np.hstack((images.reshape((sample_size, 784)), labels.reshape((sample_size, 1))))
    np.random.shuffle(data)
    X = data[:, :784].astype(float)
    X_train = X[:50000, :]
    X_test = X[50000:, :]

    labels=data[:, -1]
    Y = np.zeros(shape=(sample_size, 10), dtype=float)
    I = np.eye(10, 10, dtype=float)
    for i in range(sample_size):
        Y[i, :] = I[labels[i], :]
    Y_train = Y[:50000, :]
    Y_test = Y[50000:, :]

    return X_train, X_test, Y_train, Y_test


def load_data4D():
    images = np.load('mnist_images.npy')
    labels = np.load('mnist_labels.npy')
    sample_size = labels.shape[0]
    X_train = images[:50000, :, :].reshape((50000, 28, 28, 1)).astype(np.float32)
    X_test = images[50000:, :, :].reshape((10000, 28, 28, 1)).astype(np.float32)


    Y = np.zeros(shape=(sample_size, 10), dtype=np.float32)
    I = np.eye(10, 10)
    for i in range(sample_size):
        Y[i, :] = I[labels[i], :]
    Y_train = Y[:50000, :]
    Y_test = Y[50000:, :]

    return X_train, X_test, Y_train, Y_test


#
# import matplotlib.pyplot as plt
# def test():
#     X_train, X_test, Y_train, Y_test=load_data4D()
#     for i in range(4):
#         plt.subplot(2, 2, i+1)
#         plt.imshow(X_train[i, :, :, 0], cmap='gray')
#     plt.show()
#
# test()




