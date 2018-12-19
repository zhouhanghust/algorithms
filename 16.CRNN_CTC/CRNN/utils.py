import numpy as np
from scipy.misc import imread, imresize
import config


def resize_image(image, input_width):
    """
        Resize an image to the "good" input size
    """

    im_arr = imread(image, mode='L').astype(np.float32)
    r, c = np.shape(im_arr)
    if c > input_width:
        c = input_width
        ratio = float(input_width / c)
        final_arr = imresize(im_arr, (int(35 * ratio), input_width))
    else:
        final_arr = np.zeros((35, input_width))
        ratio = float(35 / r)
        im_arr_resized = imresize(im_arr, (35, int(c * ratio)))
        final_arr[:, 0:np.shape(im_arr_resized)[1]] = im_arr_resized
    final_arr = (final_arr / 127.5) -1
    final_arr = final_arr.transpose((1,0))
    return final_arr, c


def label_to_array(label):
    lst = [config.CHAR_VECTOR.find(x) for x in label]
    if len(lst) < config.SEQ_LEN:
        for i in range(config.SEQ_LEN-len(lst)):
            lst.append(config.NUM_CLASSES)
    return np.array(lst)


def ground_truth_to_word(ground_truth):
    """
        Return the word string based on the input ground_truth
    """

    try:
        return ''.join([config.CHAR_VECTOR[i] for i in ground_truth if i != -1])
    except Exception as ex:
        print(ground_truth)
        print(ex)
        input()


