# -*- coding:utf-8 -*-

import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn

df = pd.read_csv("elec_load.csv",error_bad_lines=False)

# fig,ax = plt.subplots()
# ax.plot(df.values[:1500],label='Load')
# ax.legend()
# plt.show()
# print(df.describe())

TimeSteps = 5
RNN_Layers = [{'steps':TimeSteps}]
Dense_layers = None
Training_Steps = 10000
Batch_Size = 100
Print_Steps = Training_Steps / 100


def lstm_model(time_steps,rnn_layers,dense_layers=None):
    def lstm_cells(layers):
        return [rnn.BasicLSTMCell(layer['steps']
                                  ,state_is_tuple=True,forget_bias=1.0) for layer in layers]
    def dnn_layers(input_layers,layers):
        return input_layers
    def _lstm_model(X,y):

















