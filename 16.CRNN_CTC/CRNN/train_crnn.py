# coding: utf-8

from CRNN import CRNN
import keras
import os
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model

cp_folder = 'checkpoints'
cp_file = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'


if not os.path.isdir(cp_folder):
    os.mkdir(cp_folder)

crnn = CRNN(120)
batch_size = 10

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

cb = [keras.callbacks.ModelCheckpoint(
    os.path.join(cp_folder, cp_file),
    save_best_only=True,
    save_weights_only=True), tensorboard]

plot_model(crnn.model,to_file='./new.png',show_shapes=True)
crnn.train(epochs=80,batch_size=batch_size,pic_path='./samples/',vali_path='./samples/',cb=cb)

K.clear_session()







