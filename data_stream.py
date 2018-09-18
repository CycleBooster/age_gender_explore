import os
import numpy as np
import scipy.io
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
from progress.bar import Bar
import random
from time import sleep
import csv
from multiprocessing import Process,Queue
from dataset import data_writer,data_reader
from data import *
class data_generator():
    def __init__(self,data_name,batch_size=8
    ,random_crop=False,random_mirror=False,random_width=False,random_rotate=False,random_gray=False,shuffle=False):
        self.output_size=(64,64)
        self.data_size=(128,128)#width and height must be the same to avoid resize and dataset get error
        self.crop_size_list=[]
        self.data_name=data_name
        self.batch_size=batch_size
        self.random_crop=random_crop
        self.random_mirror=random_mirror
        self.random_width=random_width
        self.random_rotate=random_rotate
        self.random_gray=random_gray
        self.shuffle=shuffle
        self.__build_data()
    def __get_imdb_year(self,dob):
        birth=datetime.fromordinal(max(int(dob) - 366, 1))
        if birth.month<7:
            return birth.year
        else:
            return birth.year+1
    def __load_imdb_wiki_data(self,type_name):
        target_path="./data/"+type_name+"/"
        label_path=target_path+type_name+".mat"
        label_data=scipy.io.loadmat(label_path)
        data_path_list=label_data[type_name][0][0]["full_path"][0]
        gender_list=label_data[type_name][0][0]["gender"][0]
        dob_list=label_data[type_name][0][0]["dob"][0]
        year_list=[self.__get_imdb_year(dob) for dob in dob_list]
        photo_taken_list=label_data[type_name][0][0]["photo_taken"][0]
        age_list=[photo_taken_list[index]-year for index,year in enumerate(year_list)]
        name_list=label_data[type_name][0][0]["celeb_names"][0]
        for i in range(10):
            print(data_path_list[i])
            print("dob=",dob_list[i],"  taken=",photo_taken_list[i])
            print("age=",age_list[i],"  name=",name_list[i],"   gender=",gender_list[i])
            img_path=target_path+data_path_list[i][0]
            img=cv2.imread(img_path,1)#BGR
            show_image(img)
    def __load_UTK_data(self):
        input_name="UTK"
        label_path=self.hdf5_path+"UTK_label.npy"
        if os.path.isfile(self.hdf5_path+input_name+".hdf5") and os.path.isfile(label_path):
            return 
        data_path="./data/UTKFace/crop/"
        img_name_list=os.listdir(data_path)
        data_len=len(img_name_list)
        print("data length is "+str(data_len))
        labels=[]
        writer=data_writer(input_name,self.hdf5_path)
        writer.build_dataset("input",self.data_size+(3,))
        bar = Bar('Processing', max=data_len,fill='-')
        for img_name in img_name_list:
            img_path=data_path+img_name
            try:
                age,gender,race,_=img_name.split("_")
            except:#some image get error in name
                bar.next()
                continue
            age=int(age)
            gender=int(gender)
            race=int(race)
            labels.append([age,gender,race])
            img=cv2.imread(img_path,1)
            img=cv2.resize(img,self.data_size)
            writer.write("input",img)
            bar.next()
        bar.finish()
        labels=np.array(labels)
        np.save(label_path,labels)
    def __load_appa_data(self):
        input_name="appa"
        label_path=self.hdf5_path+"appa_label.npy"
        if os.path.isfile(self.hdf5_path+input_name+".hdf5") and os.path.isfile(label_path):
            return 
        shared_name="gt_avg_"
        data_name_list=[]
        data_name_list.append("train")
        data_name_list.append("valid")
        data_name_list.append("test")
        data_list=[]
        for data_name in data_name_list:
            print(data_name)
            with open("./data/appa-real/"+shared_name+data_name+".csv", 'r') as csvfile:
                row_data=csv.DictReader(csvfile)
                data_list.extend(row_data)
        print(len(data_list))
        #     img_name_list=os.listdir(data_path)
        #     data_len=data_len+len(img_name_list)
        # print("data length is "+str(data_len))
        # labels=[]
        # writer=data_writer(input_name,self.hdf5_path)
        # writer.build_dataset("input",self.data_size+(3,))
        # bar = Bar('Processing', max=data_len,fill='-')
        # for data_path in data_path_list:
        #     img_name_list=os.listdir(data_path)
        # for img_name in img_name_list:
        #     img_path=data_path+img_name
        #     try:
        #         age,gender,race,_=img_name.split("_")
        #     except:#some image get error in name
        #         bar.next()
        #         continue
        #     age=int(age)
        #     gender=int(gender)
        #     race=int(race)
        #     labels.append([age,gender,race])
        #     img=cv2.imread(img_path,1)
        #     img=cv2.resize(img,self.data_size)
        #     writer.write("input",img)
        #     bar.next()
        # bar.finish()
        # labels=np.array(labels)
        # np.save(label_path,labels)
    def __build_data(self):
        self.hdf5_path="./data/hdf5/"
        if not os.path.isdir(self.hdf5_path):
            os.mkdir(self.hdf5_path)
        # if self.data_name=="age_gender_imdb":
        #     meta_data=self.__load_imdb_wiki_data("imdb")
        # elif self.data_name=="age_gender_wiki":
        #     meta_data=self.__load_imdb_wiki_data("wiki")
        if self.data_name=="age_gender_UTK":
            self.__load_UTK_data()
        elif self.data_name=="age_gender_appa":
            self.__load_appa_data()
        else:
            print("error in data name")
            exit()
    def show_data(self):
        if self.data_name=="age_gender_UTK":
            label_path=self.hdf5_path+"UTK_label.npy"
            input_name="UTK"
        labels=np.load(label_path)
        reader=data_reader(input_name,self.hdf5_path)
        imgs=reader.get_data("input")
        for i in range(10):
            print(labels[i])
            show_image(imgs[i])

        