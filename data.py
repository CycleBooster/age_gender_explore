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
def show_image(img,name="test"):
    img=img.astype("uint8")
    cv2.imshow(name,img)
    cv2.waitKey()
    # img=img[...,::-1]#cv load in BGR, but plt want RGB
    # plt.imshow(img)
    # plt.show()
def show_result(imgs,labels,save=False,show=True,name=None,out_size=(320,320)):
    save_path="./test_photo/temp_save/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for index,img in enumerate(imgs):
        img=cv2.resize(img,out_size)
        gender=labels[0][index][0]
        age=(int)(labels[1][index][0]+0.5)
        if gender>0.5:
            gender="M"
        else:
            gender="F"
        cv2.rectangle(img,(0,0),(100,40),(0,0,0),-1)
        cv2.putText(img,gender+" "+str(age),(0,30),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1)
        if save:
            if name==None:
                cv2.imwrite(save_path+str(index)+".jpg",img)
            else:
                cv2.imwrite(save_path+name+".jpg",img)
        if show:
            show_image(img)
