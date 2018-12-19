# coding: utf-8

from CRNN import CRNN
import keras
import os
from keras.utils.vis_utils import plot_model

crnn = CRNN(120)
crnn.base_model.load_weights('./checkpoints/weights.62-0.11.hdf5')
crnn.predict('./predict/')






