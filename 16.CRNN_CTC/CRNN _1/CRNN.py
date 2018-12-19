# coding: utf-8

from keras.layers import *
import keras.backend as K
from data_manager import DataManager
import config
import tensorflow as tf
from utils import ground_truth_to_word
import os


class CRNN():
    def __init__(self,model_path,restore,pic_width,batch_size):

        self.step = 0
        self.__model_path = model_path
        self.__save_path = os.path.join(model_path,'ckp')
        self.__restore = restore
        self.__batch_size = batch_size
        self.__pic_width = pic_width
        self.__session = tf.Session()
        self.__build_network(batch_size,pic_width)
        K.set_session(self.__session)

        with self.__session.as_default():
            (
                self.__inputs,
                self.__targets,
                self.__seq_len,
                self.__logits,
                self.__decoded,
                self.__optimizer,
                self.__acc,
                self.__cost,
                self.__conv_shape,
                self.__init
            ) = self.__build_network(batch_size,pic_width)
            self.__init.run()

        with self.__session.as_default():
            self.__saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            # Loading last save if needed
            if self.__restore:
                print('Restoring')
                ckpt = tf.train.latest_checkpoint(self.__model_path)
                if ckpt:
                    print('Checkpoint is valid')
                    self.step = int(ckpt.split('-')[1])
                    self.__saver.restore(self.__session, ckpt)

    def __build_network(self,batch_size,pic_width):

        inputs = tf.placeholder(tf.float32, [batch_size, pic_width, 35, 1])

        # Our target output
        targets = tf.sparse_placeholder(tf.int32, name='targets')

        # The length of the sequence
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        crnn_output, conv_shape = self.__crnn(inputs)

        logits = tf.reshape(crnn_output, [-1, 512])

        W = tf.Variable(tf.truncated_normal([512, config.NUM_CLASSES], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0., shape=[config.NUM_CLASSES]), name="b")

        logits = tf.matmul(logits, W) + b

        logits = tf.reshape(logits, [batch_size, -1, config.NUM_CLASSES])

        # Final layer
        logits = tf.transpose(logits, (1, 0, 2))

        # Loss and cost calculation
        loss = tf.nn.ctc_loss(targets, logits, seq_len)

        cost = tf.reduce_mean(loss)

        # Training step
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

        # The decoded answer
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)

        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)

        # The error rate
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

        init = tf.global_variables_initializer()

        return inputs, targets, seq_len, logits, dense_decoded, optimizer, acc, cost, conv_shape, init

    def __crnn(self, inputs):

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='same')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 2), padding='same')(x)
        x = Conv2D(512, (2, 2), padding='valid', activation='relu')(x)
        cnn_shape = x.get_shape()
        x = Reshape(target_shape=(int(cnn_shape[1]), int(cnn_shape[2] * cnn_shape[3])))(x)

        x = Bidirectional(GRU(256, activation='relu', return_sequences=True), merge_mode='concat')(x)
        x = Bidirectional(GRU(256, activation='relu', return_sequences=True), merge_mode='concat')(x)

        return x,cnn_shape

    def train(self,iteration_count,pic_path):
        data_manager_train = DataManager(self.__batch_size,pic_path,self.__pic_width,True)

        with self.__session.as_default():
            print('Training')
            for i in range(self.step, iteration_count + self.step):
                iter_loss = 0
                for batch_y, batch_dt, batch_x in data_manager_train.all_data:
                    op, decoded, loss_value = self.__session.run(
                        [self.__optimizer, self.__decoded, self.__cost],
                        feed_dict={
                            self.__inputs: batch_x,
                            self.__seq_len: [self.__conv_shape[1]] * self.__batch_size,
                            self.__targets: batch_dt
                        }
                    )

                    if i % 10 == 0:
                        for j in range(len(batch_y)):
                            print('original:',batch_y[j])
                            print(ground_truth_to_word(decoded[j]))

                    iter_loss += loss_value

                self.__saver.save(
                    self.__session,
                    self.__save_path,
                    global_step=self.step
                )

                print('[{}] Iteration loss: {}'.format(self.step, iter_loss))

                self.step += 1
        return None

    def test(self,pic_path):
        data_manager_test = DataManager(self.__batch_size, pic_path, self.__pic_width, True)

        y_test, dt_test, x_test = data_manager_test.all_data[0]
        with self.__session.as_default():
            print('Testing')
            op, decoded, loss_value = self.__session.run(
                [self.__optimizer, self.__decoded, self.__cost],
                feed_dict={
                    self.__inputs: x_test,
                    self.__seq_len: [self.__conv_shape[1]] * self.__batch_size,
                    self.__targets: dt_test
                })
            for j in range(len(y_test)):
                print('original:', y_test[j])
                print(ground_truth_to_word(decoded[j]))
            ACC = self.__session.run(self.__acc,feed_dict={
                self.__inputs: x_test,
                self.__seq_len: [self.__conv_shape[1]] * self.__batch_size,
                self.__targets: dt_test
            })
            print('ACC:',ACC)

    def predict(self,pic_path):
        data_manager_predict= DataManager(self.__batch_size, pic_path, self.__pic_width, False)
        y_pd, dt_pd, x_pd = data_manager_predict.all_data

        with self.__session.as_default():
            print('Predicting')
            decoded = self.__session.run(self.__decoded,feed_dict={
                self.__inputs:x_pd,
                self.__seq_len:[self.__conv_shape[1]]*self.__batch_size
            })

            for j in range(len(y_pd)):
                print('original:',y_pd[j])
                print(ground_truth_to_word(decoded[j]))

if __name__ == "__main__":
    # crnn = CRNN('./save/',False,80,20)
    #
    # crnn.train(150,'./samples/')

    #===================================
    #Test
    # crnn = CRNN('./save/',True,80,42)
    # crnn.test('./validation/')

    #===================================
    #Predict
    crnn = CRNN('./save/',True,80,4)
    crnn.predict('./predict/')












