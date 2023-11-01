import numpy as np


def one_hot_encode_custom(number):
    arr = np.zeros(10)
    arr[number] = 1
    return arr
