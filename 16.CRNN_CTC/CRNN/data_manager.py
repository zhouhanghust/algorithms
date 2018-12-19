
import os
import numpy as np


from utils import resize_image, label_to_array


class DataManager(object):
    def __init__(self, pic_path, max_image_width):

        self.max_image_width = max_image_width
        self.pic_path = pic_path

        self.data, self.data_len = self.__load_data()
        self.all_data = self.__generate_all_data()

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
            file_name = f
            examples.append(
                (
                    arr,
                    label_to_array(f.split('_')[0]),
                    file_name
                )
            )

        return examples, len(examples)

    def __generate_all_data(self):
            raw_x, raw_y, file_name = zip(*self.data)

            y = np.array(raw_y)
            x = np.reshape(
                np.array(raw_x),
                (-1, self.max_image_width, 35, 1)
            )

            return [x,y,file_name]




