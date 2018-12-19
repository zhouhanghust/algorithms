import numpy as np
from scipy.misc import imread, imresize
import config


def sparse_tuple_from(sequences, dtype=np.int32):
    """
        Inspired (copied) from https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
    """

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), [i for i in range(len(seq))]))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape


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
    try:
        return [config.CHAR_VECTOR.index(x) for x in label]
    except Exception as ex:
        print(label)
        raise ex


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


