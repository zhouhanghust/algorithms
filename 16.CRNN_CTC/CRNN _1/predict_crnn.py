# coding: utf-8

from CRNN import CRNN
import keras
import os
from keras.utils.vis_utils import plot_model

crnn = CRNN(80)
crnn.base_model.load_weights('./checkpoints/weights.13-3.99.hdf5')
crnn.predict('./predict/')
plot_model(crnn.base_model,to_file='./crnn_base_model.png',show_shapes=True)


