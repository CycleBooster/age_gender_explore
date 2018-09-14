import os
import numpy as np
import h5py
import scipy.io
# from lxml import objectify
import cv2
from dataset import data_writer,data_reader
# from progress.bar import Bar
import random
from multiprocessing import Process,Queue
from data import *
from time import sleep
class data_generator():
    def __init__(self,data_name,batch_size=8
    ,random_crop=False,random_mirror=False,random_width=False,random_rotate=False,random_gray=False,shuffle=False):
        self.output_size=(64,64)
        self.data_name=data_name
        self.batch_size=batch_size
        self.random_crop=random_crop
        self.random_mirror=random_mirror
        self.random_width=random_width
        self.random_rotate=random_rotate
        self.random_gray=random_gray
        self.shuffle=shuffle
        self.__build_data()
    def __build_data(self):
        label_path="./data/imdb/imdb.mat"
        # label_data=h5py.File(label_path,'r')
        # for key in label_data:
        #     print(key)
        label_data=scipy.io.loadmat(label_path)
        # for key,test_list in label_data["imdb"][0][0].items():
        #     print(key)
        print(len(label_data["imdb"][0][0]["full_path"][0]))
        # print(len(label_data["imdb"][0][0]))

        