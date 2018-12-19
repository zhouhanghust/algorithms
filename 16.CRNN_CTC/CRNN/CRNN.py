# coding: utf-8

from keras.layers import *
from keras.models import *
import keras.backend as K
from data_manager import DataManager
import config
import numpy as np


class CRNN():
    def __init__(self,pic_width):
        self.pic_width = pic_width

        self.__build_network()

    def __build_network(self):
        input_tensor = Input((self.pic_width, 35, 1))
        crnn_output,conv_shape = self.__crnn(input_tensor)

        self.conv_shape = conv_shape

        x = Dropout(0.5)(crnn_output)
        x = Dense(config.NUM_CLASSES,activation='softmax')(x)

        self.base_model = Model(input=input_tensor,outputs=x)

        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            y_pred = y_pred[:, 2:, :]
            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

        labels = Input(name='the_labels', shape=[config.SEQ_LEN], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,),\
                          name='ctc')([x, labels, input_length, label_length])

        self.model = Model(input=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')

    def __crnn(self, inputs):

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(1, 2), padding='same')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(1, 2), padding='same')(x)
        x = Conv2D(512, (2, 2), padding='valid', activation='relu')(x)
        cnn_shape = x.get_shape()
        x = Reshape(target_shape=(int(cnn_shape[1]), int(cnn_shape[2] * cnn_shape[3])))(x)

        x = Bidirectional(GRU(256, activation='relu', return_sequences=True), merge_mode='concat')(x)
        x = Bidirectional(GRU(256, activation='relu', return_sequences=True), merge_mode='concat')(x)

        return x,cnn_shape

    def train(self,epochs,batch_size,pic_path,vali_path,cb):
        data_manager = DataManager(pic_path=pic_path, max_image_width=self.pic_width)
        x,label,_ = data_manager.all_data
        input_length = np.ones(len(x))*(int(self.conv_shape[1]-2))
        label_length = np.ones(len(x))*config.SEQ_LEN
        y_placeholder = np.ones(len(x))

        vali_data_manager = DataManager(pic_path=vali_path,max_image_width=self.pic_width)

        v_x,v_label,_ = vali_data_manager.all_data
        v_input_length = np.ones(len(v_x))*(int(self.conv_shape[1]-2))
        v_label_length = np.ones(len(v_x))*config.SEQ_LEN
        v_y_placeholder = np.ones(len(v_x))

        self.model.fit([x,label,input_length,label_length],
                       y_placeholder,epochs=epochs,batch_size=batch_size,callbacks=cb,\
                       validation_data=([v_x,v_label,v_input_length,v_label_length],v_y_placeholder))

    def predict(self,file_path):
        data_manager = DataManager(pic_path=file_path, max_image_width=self.pic_width)

        x,label,file_name = data_manager.all_data
        y_pred = self.base_model.predict(x)
        shape = y_pred[:, 2:, :].shape
        ctc_decode = K.ctc_decode(y_pred[:, 2:, :],
                                  input_length=np.ones(shape[0]) * shape[1])[0][0]
        out = K.get_value(ctc_decode)

        length = len(out)
        acc_count = 0
        for i in range(length):
            str1 = file_name[i].split('_')[0]
            str2 = ''.join([config.CHAR_VECTOR[j] for j in out[i] if j != -1])
            if str1 == str2:
                acc_count += 1

            print('ori', file_name[i])
            print(str2)
        print('ACC: {0:.2f}%'.format(100*acc_count/length))




