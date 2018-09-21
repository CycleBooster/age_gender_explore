import os
import sys
import numpy as np
from keras import backend as K
from keras.applications.resnet50 import preprocess_input
# from lxml import objectify
import cv2
from dataset import data_writer,data_reader
# from progress.bar import Bar
# import progressbar
import matplotlib.pyplot as plt
import random
# from progress import bar
def resnet50_preprocess_input(x,is_cv_load=True):
    if is_cv_load:#cv load in BGR inverse back channel
        x=x[...,::-1]
    x = preprocess_input(x)#used for PIL, from RGB to BGR
    return x
def show_image(img):
    img=img.astype("uint8")
    cv2.imshow("test",img)
    cv2.waitKey()
    # img=img[...,::-1]#cv load in BGR, but plt want RGB
    # plt.imshow(img)
    # plt.show()