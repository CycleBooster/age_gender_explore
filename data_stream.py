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
    def __init__(self,data_name,batch_size=32,age_name="one",one_hot_gender=True
    ,random_shift=False,random_mirror=False,random_rotate=False,random_scale=False,shuffle=False):
        self.output_size=(64,64)
        self.data_size=(128,128)#width and height must be the same to avoid resize and dataset get error
        # self.crop_size_list=[]#set after get data name
        self.data_name=data_name
        self.batch_size=batch_size
        self.age_name=age_name
        self.one_hot_gender=one_hot_gender
        self.random_shift=random_shift
        self.random_mirror=random_mirror
        self.random_rotate=random_rotate
        self.random_scale=random_scale
        self.shuffle=shuffle
        self.queue= Queue()
        self.__build_data()
        self.__get_data_len()
        self.scale_list=[1,0.9,0.8]
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
        np.random.shuffle(img_name_list)
        for img_name in img_name_list:
            img_path=data_path+img_name
            try:
                age,gender,race,_=img_name.split("_")
            except:#some image get error in name
                bar.next()
                continue
            try:
                age=int(age)
                gender=int(gender)
                race=int(race)
            except:
                print(img_name)
                exit()
            labels.append([age,gender,race])
            img=cv2.imread(img_path,1)
            img=cv2.resize(img,self.data_size)
            writer.write("input",img)
            bar.next()
        bar.finish()
        labels=np.array(labels)
        np.save(label_path,labels)
    def __load_appa_data(self,type):
        input_name="appa_"+type
        label_path=self.hdf5_path+"appa_"+type+"_label.npy"
        if os.path.isfile(self.hdf5_path+input_name+".hdf5") and os.path.isfile(label_path):
            return 
        shared_name="gt_avg_"
        extend_name="allcategories_"
        data_name_list=[]
        if type=="val":
            data_name_list.append("train")
            data_name_list.append("valid")
        elif type=="test":
            data_name_list.append("test")
        else:
            print("error in type of appa data")
            exit()
        data_list=[]
        extend_data_list=[]
        for data_name in data_name_list:
            with open("./data/appa-real/"+shared_name+data_name+".csv", 'r') as csvfile:
                row_data=csv.DictReader(csvfile)
                data_list.append(list(row_data))
            with open("./data/appa-real/"+extend_name+data_name+".csv", 'r') as csvfile:
                row_data=csv.DictReader(csvfile)
                extend_data_list.append(list(row_data))
        data_len=0
        for sub_data_list in data_list:
            data_len=data_len+len(sub_data_list)
        print("data length is "+str(data_len))
        writer=data_writer(input_name,self.hdf5_path)
        writer.build_dataset("input",self.data_size+(3,))
        labels=[]
        bar = Bar('Processing', max=data_len,fill='-')
        for folder_index,sub_data_list in enumerate(data_list):
            shuffle_index=[i for i in range(len(sub_data_list))]
            np.random.shuffle(shuffle_index)
            for i in range(len(sub_data_list)):
                index=shuffle_index[i]
                data=sub_data_list[index]
            # for index,data in enumerate(sub_data_list):
                bar.next()
                img_name=data['file_name']
                # img_number,img_form=img_name.split(".")
                # img_path="./data/appa-real/"+data_name_list[folder_index]+"/"+img_number+".jpg_face."+img_form
                img_path="./data/appa-real/crop_"+data_name_list[folder_index]+"/"+img_name
                if not os.path.isfile(img_path):
                    continue
                real_age=data['real_age']
                gender=extend_data_list[folder_index][index]['gender']
                race=extend_data_list[folder_index][index]['race']
                labels.append([real_age,gender,race])
                img=cv2.imread(img_path,1)
                img=cv2.resize(img,self.data_size)
                writer.write("input",img)
        bar.finish()
        labels=np.array(labels)
        np.save(label_path,labels)
    def __build_data(self):
        self.hdf5_path="./data/hdf5/"
        if not os.path.isdir(self.hdf5_path):
            os.mkdir(self.hdf5_path)
        # if self.data_name=="age_gender_imdb":
        #     self.__load_imdb_wiki_data("imdb")
        # elif self.data_name=="age_gender_wiki":
        #     self.__load_imdb_wiki_data("wiki")
        if self.data_name=="age_gender_UTK":#used in training
            self.__load_UTK_data()
        elif self.data_name=="age_gender_appa_val":#used in validate
            self.__load_appa_data("val")
        elif self.data_name=="age_gender_appa_test":#used in validate
            self.__load_appa_data("test")
        else:
            print("error in data name")
            exit()
    def __get_data_len(self):
        self.data_len=self.__data_read(label_only=True).shape[0]
    def __data_read(self,label_only=False):
        if self.data_name=="age_gender_UTK":
            label_path=self.hdf5_path+"UTK_label.npy"
            input_name="UTK"
        elif self.data_name=="age_gender_appa_val":
            label_path=self.hdf5_path+"appa_val_label.npy"
            input_name="appa_val"
        elif self.data_name=="age_gender_appa_test":
            label_path=self.hdf5_path+"appa_test_label.npy"
            input_name="appa_test"
        labels=np.load(label_path)
        if not label_only:
            reader=data_reader(input_name,self.hdf5_path)
            imgs=reader.get_data("input")
            return labels,imgs
        else:
            return labels

    def __set_input(self,img):
        random_num=random.randint(0,5)
        temp_img=np.array(img,copy=True)
        (h,w)=temp_img.shape[:2]
        if self.random_mirror and (random_num%2)==0:
            temp_img=cv2.flip(temp_img,1)
        if self.random_shift:
            crop_random_num=random.randint(0,8)
            offset_para=10
            w_offset=((int)(crop_random_num/3)-1)*offset_para
            h_offset=(crop_random_num%3-1)*offset_para
            shift_M=np.float32([[1, 0, w_offset], [0, 1, h_offset]])
            temp_img = cv2.warpAffine(temp_img, shift_M, (w, h))
        if self.random_rotate:#3
            angle=(random_num%3-1)*10
        else:
            angle=0
        if self.random_scale:
            random_scale_num=random.randint(0,2)
            scale=self.scale_list[random_scale_num]
        else:
            scale=self.scale_list[0]
        M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
        temp_img = cv2.warpAffine(temp_img, M, (w, h))
        out_img=cv2.resize(temp_img,self.output_size)
        return out_img
    def __set_label(self,meta_label):#decode label
        meta_age=meta_label[0]
        meta_age=(int)(meta_age)
        if meta_age>100:
            meta_age=100
        if self.one_hot_gender:
            age=np.zeros(101)
            age[meta_age]=1
        else:
            age=np.zeros(1)
            age[0]=meta_age
        
        meta_gender=meta_label[1]#gender=1 for male
        gender=np.zeros(1)
        if self.data_name=="age_gender_UTK":
            gender[0]=1-(int)(meta_gender)
        elif self.data_name=="age_gender_appa_val" or self.data_name=="age_gender_appa_test":
            if meta_gender=="male":
                gender[0]=1
            elif meta_gender=="female":
                gender[0]=0
            else:
                print("??")
        else:
            print("error in data name")
            exit()
        return gender,age
    def __get_data(self,img_buffer,label_buffer,index_buffer,buffer_start_index):
        _inputs=np.zeros((self.batch_size,)+self.output_size+(3,))
        _out_gender=np.zeros((self.batch_size,1))
        if self.one_hot_gender:
            _out_age=np.zeros((self.batch_size,101))#age classification
        else:
            _out_age=np.zeros((self.batch_size,1))#age regression
        for i in range(self.batch_size):
            _inputs[i]=self.__set_input(img_buffer[index_buffer[i+buffer_start_index]])
            _out_gender[i],_out_age[i]=self.__set_label(label_buffer[index_buffer[i+buffer_start_index]])
        # return (_inputs,{"gender_y":_out_gender,"one_age_y":_out_age})
        return (_inputs,{"gender_y":_out_gender,self.age_name+"_age_y":_out_age})
        # return (_inputs,{"gender_y":_out_gender})
        # return (_inputs,{"gender_y":_out_gender,"med_gen_y":_out_gender,"one_age_y":_out_age,"med_age_y":_out_age})
        # return (_inputs,{"med_gen_y":_out_gender,"med_age_y":_out_age})
        _virtual_conf=np.zeros((self.batch_size,1))
        _out_age=np.concatenate([_virtual_conf,_out_age],axis=-1)
        _out_gender=np.concatenate([_virtual_conf,_out_gender],axis=-1)
        return (_inputs,{"gender_y":_out_gender,"one_age_y":_out_age})
    def show_data(self):
        labels,imgs=self.__data_read()
        for i in range(10):
            print(self.__set_label(labels[i]))
            test_img=self.__set_input(imgs[i])
            show_image(test_img)
    def data_process(self):
        labels,imgs=self.__data_read()
        buffer_multiple=10
        start_index=0
        restart_flag=False
        while 1:
            if self.queue.qsize()<3*buffer_multiple:
                if start_index+buffer_multiple*self.batch_size<=self.data_len:
                    end_i=buffer_multiple
                else:
                    end_i=(int)((self.data_len-start_index)/self.batch_size)
                    restart_flag=True
                img_buffer=imgs[start_index:start_index+end_i*self.batch_size]
                img_buffer=resnet50_preprocess_input(img_buffer)
                label_buffer=labels[start_index:start_index+end_i*self.batch_size]
                index_buffer=[i for i in range(len(img_buffer))]
                if self.shuffle==True:
                    np.random.shuffle(index_buffer)
                for i in range(end_i):
                    self.queue.put(self.__get_data(img_buffer,label_buffer,index_buffer,i*self.batch_size))
                if restart_flag:
                    start_index=0
                    restart_flag=False
                else:
                    start_index+=buffer_multiple*self.batch_size
    def get_max_batch_index(self):
        return (int)(self.data_len/self.batch_size)
    def generator(self):
        self.p=Process(target=self.data_process,daemon=True)
        self.p.start()
        while True:
            if self.queue.empty()==False:
                yield self.queue.get()
    def test_data(self,start_index=0,pick=False):
        buffer_multiple=10
        labels,imgs=self.__data_read()
        end_flag=False
        while True:
            if start_index+buffer_multiple*self.batch_size<=self.data_len:
                end_i=buffer_multiple
            else:
                end_i=(int)((self.data_len-start_index)/self.batch_size)
                end_flag=True
            img_buffer=imgs[start_index:start_index+end_i*self.batch_size]
            img_buffer=resnet50_preprocess_input(img_buffer)
            label_buffer=labels[start_index:start_index+end_i*self.batch_size]
            index_buffer=[i for i in range(len(img_buffer))]
            for i in range(end_i):
                if pick:
                    (out_imgs,dict)=self.__get_data(img_buffer,label_buffer,index_buffer,i*self.batch_size)
                    origin_imgs=imgs[start_index:start_index+self.batch_size]
                    yield (origin_imgs,out_imgs)
                else:
                    yield self.__get_data(img_buffer,label_buffer,index_buffer,i*self.batch_size)
            start_index+=buffer_multiple*self.batch_size
            if end_flag:
                break


        