import numpy as np
import time


def sigmoid(x):
    temp = np.exp(x)
    return temp/(1+temp)


def sigmoid_1(x):
    temp = np.exp(x)
    return temp/np.power(1+temp, 2)


def load_data():
    images = np.load('mnist_images.npy')
    labels = np.load('mnist_labels.npy')
    sample_size = labels.shape[0]
    data = np.hstack((images.reshape((sample_size, 784)), labels.reshape((sample_size, 1))))
    np.random.shuffle(data)
    X = data[:, :784].transpose().astype(float)
    X_train = X[:, :50000]
    X_test = X[:, 50000:]

    labels=data[:, -1]
    Y = np.zeros(shape=(10, sample_size), dtype=float)
    I = np.eye(10, 10, dtype=float)
    for i in range(sample_size):
        Y[:, i] = I[:, labels[i]]
    Y_train = Y[:, :50000]
    Y_test = Y[:, 50000:]
    return X_train, X_test, Y_train, Y_test


class Network:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.num_layers = len(num_nodes) - 1
        self.W = [None] * self.num_layers
        self.b = [None] * self.num_layers
        for i in range(self.num_layers):
            self.W[i] = np.random.randn(num_nodes[i+1], num_nodes[i]) /np.sqrt(num_nodes[i])
            self.b[i] = np.random.randn(num_nodes[i + 1], 1) *0.1


    def feed_forward(self, x, y):
        # x has shape (input_size, batch_size)
        # y has shape (output_size, batch_size)
        self.x = (x - 127.5)/np.std(x) # normalization
        self.y = y
        self.batch_size = x.shape[1]
        self.a = [None] * self.num_layers
        self.z = [None] * self.num_layers
        for i in range(self.num_layers):
            if i == 0:
                self.z[i] = np.dot(self.W[i], self.x) + self.b[i]
            else:
                self.z[i] = np.dot(self.W[i], self.a[i - 1]) + self.b[i]
            self.a[i] = sigmoid(self.z[i])
        self.loss = np.mean(np.linalg.norm(self.a[-1] - self.y, axis=0))
        self.accuracy = np.sum(np.argmax(self.a[-1], axis=0) == np.argmax(self.y, axis=0))/self.batch_size


    def back_propagate(self):
        self.dW = [None] * self.num_layers
        self.dz = [None] * self.num_layers

        for i in range(self.num_layers):
            self.dW[i] = np.zeros((self.num_nodes[i+1], self.num_nodes[i], self.batch_size))
            self.dz[i] = np.random.randn(self.num_nodes[i + 1], self.batch_size)

        for k in range(self.batch_size):
            for i in range(self.num_layers):
                if i == 0:
                    self.dz[-1][:, k] = 2*(self.a[-1][:, k] - self.y[:, k]) * sigmoid_1(self.z[-1][:, k])
                else:
                    self.dz[-1-i][:, k] = (self.W[-i].transpose()@self.dz[-i][:, k]) * sigmoid_1(self.z[-1-i][:, k])

                if i == (self.num_layers-1):
                    self.dW[-1 - i][:, :, k] = self.dz[-1 - i][:, k].reshape((self.dz[-1 - i].shape[0], 1)) \
                                               @ self.x[:, k].reshape((1, self.x.shape[0]))
                else:
                    self.dW[-1-i][:, :, k] = self.dz[-1-i][:, k].reshape((self.dz[-1-i].shape[0], 1)) \
                                             @ self.a[-2-i][:, k].reshape((1, self.a[-2-i].shape[0]))


    def update_parameters(self, learning_rate):
        for i in range(self.num_layers):
            avg_dW = np.mean(self.dW[i], axis=2)
            avg_db = np.mean(self.dz[i], axis=1)
            self.W[i] -= learning_rate * avg_dW
            self.b[i] -= learning_rate * avg_db.reshape((self.num_nodes[i + 1], 1))


    def train(self, X, Y, learning_rate, batch_size, current_epoch):
        num_batches = Y.shape[1] // batch_size
        for j in range(num_batches):
            x = X[:, j*batch_size: (j+1)*batch_size]
            y = Y[:, j*batch_size: (j+1)*batch_size]
            self.feed_forward(x, y)
            self.back_propagate()
            self.update_parameters(learning_rate)
            if j%100==0:
                print('Epoch {0} [{1}/{2}]\t Loss {3:.3f}'.format(current_epoch, j, num_batches, self.loss))


    def test(self, X, Y, current_epoch):
        self.feed_forward(X, Y)
        print("=============================================")
        print('Epoch {0}  Accuracy on test set: {1:.2f}%'.format(current_epoch, self.accuracy*100))
        print("=============================================")



nn = Network([784, 30, 10])
for i in range(10):
    last_time =time.time()
    X_train, X_test, Y_train, Y_test = load_data()
    nn.train(X_train, Y_train, learning_rate=0.1, batch_size=10, current_epoch=i+1)
    print('Epoch {0} completed! ({1:.1f} seconds)'.format(i+1, time.time()-last_time))
    nn.test(X_test, Y_test, current_epoch=i+1)