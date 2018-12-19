# coding: utf-8

from CRNN import CRNN
import keras
import os
import keras.backend as K

cp_folder = 'checkpoints'
cp_file = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'


if not os.path.isdir(cp_folder):
    os.mkdir(cp_folder)

crnn = CRNN(120)

cb = [keras.callbacks.ModelCheckpoint(
    os.path.join(cp_folder, cp_file),
    save_best_only=True,
    save_weights_only=True)]
crnn.train(epochs=20,batch_size=10,pic_path='./samples/',vali_path='./validation',cb=cb)

K.clear_session()







