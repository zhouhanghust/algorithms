
import os
import numpy as np


from utils import resize_image, label_to_array, sparse_tuple_from


class DataManager(object):
    def __init__(self, batch_size,pic_path, max_image_width, istrain=False):

        self.batch_size = batch_size
        self.max_image_width = max_image_width
        self.pic_path = pic_path
        self.istrain = istrain
        self.data, self.data_len = self.__load_data()
        if istrain:
            self.all_data = self.__generate_train_data()
        else :
            self.all_data = self.__generate_test_data()

    def __load_data(self):
        """
            Load all the images in the folder
        """

        print('Loading data')
        examples = []

        for f in os.listdir(self.pic_path):

            arr, initial_len = resize_image(
                os.path.join(self.pic_path, f),
                self.max_image_width
            )
            examples.append(
                (
                    arr,
                    f.split('_')[0],
                    label_to_array(f.split('_')[0])
                )
            )

        return examples, len(examples)

    def __generate_train_data(self):
        begin = 0
        end = begin + self.batch_size
        train_batches = []
        while not end > len(self.data):

            raw_batch_x, raw_batch_y, raw_batch_la = zip(*self.data[begin:end])
            begin = end
            end = end+self.batch_size

            batch_y = np.reshape(
                np.array(raw_batch_y),
                (-1)
            )

            batch_dt = sparse_tuple_from(
                    np.array(raw_batch_la)
            )

            batch_x = np.reshape(
                np.array(raw_batch_x),
                (-1, self.max_image_width, 35, 1)
            )

            train_batches.append((batch_y, batch_dt, batch_x))
        return train_batches

    def __generate_test_data(self):
            raw_x, raw_y, raw_la = zip(*self.data)

            y = np.array(raw_y)

            dt = np.array(raw_la)

            x = np.reshape(
                np.array(raw_x),
                (-1, self.max_image_width, 35, 1)
            )

            return y, dt, x



