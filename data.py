import os
import numpy as np
from keras import backend as K
# from lxml import objectify
import cv2
from dataset import data_writer,data_reader
# from progress.bar import Bar
# import progressbar
import matplotlib.pyplot as plt
import random
# from progress import bar
def preprocess_input(x, dim_ordering='default',is_list=False):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}
    
    x=x.astype('float')
    if is_list:
        if dim_ordering == 'th':
            x[:,0, :, :] -= 103.939
            x[:,1, :, :] -= 116.779
            x[:,2, :, :] -= 123.68
            # 'RGB'->'BGR'
            x = x[::-1, :, :]
            assert x.shape[1]==3
        else:
            x[:,:, :, 0] -= 103.939
            x[:,:, :, 1] -= 116.779
            x[:,:, :, 2] -= 123.68
            # 'RGB'->'BGR'
            x = x[:, :, ::-1]
            assert x.shape[3]==3
    else:
        if dim_ordering == 'th':
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
            # 'RGB'->'BGR'
            x = x[::-1, :, :]
            assert x.shape[0]==3
        else:
            x[:, :, 0] -= 103.939
            x[:, :, 1] -= 116.779
            x[:, :, 2] -= 123.68
            # 'RGB'->'BGR'
            x = x[:, :, ::-1]
            assert x.shape[2]==3
    return x
def inverse_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}
    x=np.copy(x)
    if dim_ordering == 'th':
        # 'BGR'->'RGB'
        x = x[::-1, :, :]
        x[0, :, :] += 103.939
        x[1, :, :] += 116.779
        x[2, :, :] += 123.68
        assert x.shape[0]==3
    else:
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        assert x.shape[2]==3
    x=x.astype('uint8')
    return x