import numpy as np
import os
import time
from sklearn.ensemble import RandomForestClassifier

"""
1, learn how to implement K-fold cross validation
2, learn how to format strings
3, learn how to write text to files
4, learn how to use pyplot
"""


def load_mnist():
    images = np.load('mnist_images.npy')
    sample_size = images.shape[0] # 60000
    feature_size = images.shape[1]*images.shape[2] # 28*28
    images = images.reshape((sample_size, feature_size))
    labels = np.load('mnist_labels.npy')
    return images, labels

# image_blocks = np.split(images, num_blocks)
# label_blocks = np.split(labels, num_blocks)

def create_blocks(images, labels, num_blocks):
    block_size = images.shape[0]//num_blocks
    image_blocks = []
    label_blocks = []
    for i in range(num_blocks):
        start_index = block_size*i
        end_index = block_size*(i+1)
        _image_block = images[start_index :end_index,:]
        _label_block = labels[start_index :end_index]
        image_blocks.append(_image_block)
        label_blocks.append(_label_block)
        print('The {0}th block has components of size {1} and {2}, indices range from {3} to {4}'
              .format(i, _image_block.shape, _label_block.shape, start_index, end_index))
    return image_blocks, label_blocks


def create_split(image_blocks, label_blocks, index):
    num_blocks = len(image_blocks)
    X_val = image_blocks[index]
    Y_val = label_blocks[index]
    if index == 0:
        X_train = np.vstack(image_blocks[1:])
        Y_train = np.hstack(label_blocks[1:])
    elif index == num_blocks-1:
        X_train = np.vstack(image_blocks[:num_blocks-1])
        Y_train = np.hstack(label_blocks[:num_blocks-1])
    else:
        X_train = np.vstack(image_blocks[:index] + image_blocks[index + 1:])
        Y_train = np.hstack(label_blocks[:index] + label_blocks[index + 1:])
    # print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)
    return X_train, Y_train, X_val, Y_val


def main(num_split, max_depth):
    images, labels = load_mnist()
    image_blocks, label_blocks = create_blocks(images, labels, num_blocks=num_split)

    filename='log_10.txt'
    if os.path.isfile(filename):
        print('File {0} already exists. Deleting file...'.format(filename))
        os.remove(filename)
    f=open(filename, mode='x')

    avg_accuracy = np.zeros(10)
    std_accuracy = np.zeros(10)
    num_estimators = np.zeros(10, dtype=int)
    for i in range(10):
        n = (i + 1) * 5
        num_estimators[i]=n
        accuracy = np.zeros(num_split)
        last_time = time.time()
        for j in range(num_split):
            X_train, Y_train, X_val, Y_val = create_split(image_blocks, label_blocks, j)
            classifier = RandomForestClassifier(n_estimators=n, max_depth=max_depth, n_jobs=-1)
            classifier.fit(X_train, Y_train)
            prediction = classifier.predict(X_val)
            match = sum(prediction == Y_val)
            accuracy[j] = match / Y_val.shape[0]
            string = "Classification rate: {0}/{1} ({2:.1f}%) with n_estimators={3} and split_index={4}"\
                .format(match, Y_val.shape[0], accuracy[j] * 100, n, j)
            print(string)
            f.write(string+'\n')
        avg_accuracy[i] = np.mean(accuracy)
        std_accuracy[i] = np.std(accuracy)
        dashes = "---------------------------------------------------------------------------------------------"
        result = "n_estimators={2}: average classification rate: {0:.1f}% (std={1:.4f}), elapsed time {3:.1f} sec."\
              .format(avg_accuracy[i] * 100, std_accuracy[i], n, time.time() - last_time)
        print(dashes)
        print(result)
        print(dashes)
        f.write(dashes+'\n')
        f.write(result+'\n')
        f.write(dashes+'\n')

    f.close()
    np.save('log_avg_{0}.npy'.format(max_depth), avg_accuracy)
    np.save('log_std_{0}.npy'.format(max_depth), std_accuracy)
    np.save('log_num_{0}.npy'.format(max_depth), num_estimators)



main(num_split=5, max_depth=10)