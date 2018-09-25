import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import time
import sys
import tensorflow as tf
import numpy as np
import math
from keras.applications import ResNet50
from keras.layers import Conv2D,Dense,Flatten,Activation,Lambda
from keras.optimizers import Adam
from keras.models import Model
from keras.losses import categorical_crossentropy,binary_crossentropy
from keras.callbacks import LearningRateScheduler,LambdaCallback
from keras import regularizers
from data import *
from data_stream import data_generator
weight_decay_rate=0.0001
class age_gender_classifier():
    def __init__(self,batch_size=64,lr=0.0001,test_size=(64,64),model_type="one"):
        self.age_width=101
        self.model_path="./model/age_gender_"+model_type+".h5"
        self.lr=lr
        self.test_size=test_size
        self.model_type=model_type
        self.batch_size=batch_size
        pretrain_model=ResNet50(input_shape=test_size+(3,),weights='imagenet',include_top=False)
        x=pretrain_model.input
        pretrain_output=pretrain_model.output
        pretrain_output=Flatten()(pretrain_output)
        temp_y=Dense(256,activation="relu"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate))(pretrain_output)
        temp_y=Dense(256,activation="relu"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_y)
        gender_y=Dense(1,activation="sigmoid"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate),name="gender_y")(temp_y)
        temp_one_age_y=Dense(1,activation="sigmoid"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_y)
        one_age_y=Lambda(lambda x:100*x,name="one_age_y")(temp_one_age_y)
        softmax_age_y=Dense(self.age_width,activation="softmax"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate),name="soft_age_y")(temp_y)
        out_age=Lambda(self.weighted_average,name="out_age_y")(softmax_age_y)

        self.optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        if model_type=="one":
            self.train_model=Model(inputs=x,outputs=[gender_y,one_age_y])
            self.pred_model=self.train_model
            self.train_model.compile(optimizer=self.optimizer,loss={"gender_y":self.gender_loss,"one_age_y":self.age_loss_R}
                ,metrics={"gender_y":[self.g_acc_3,self.g_acc_5,self.g_acc_7],"one_age_y":self.MAE_R})
        if model_type=="soft":
            self.train_model=Model(inputs=x,outputs=[gender_y,softmax_age_y])
            self.pred_model=Model(inputs=x,outputs=[gender_y,out_age])
            self.train_model.compile(optimizer=self.optimizer,loss={"gender_y":self.gender_loss,"soft_age_y":self.age_loss_C}
                ,metrics={"gender_y":[self.g_acc_3,self.g_acc_5,self.g_acc_7],"one_age_y":self.MAE_C})
        if model_type=="out":
            self.train_model=Model(inputs=x,outputs=[gender_y,out_age])
            self.pred_model=self.train_model
            self.train_model.compile(optimizer=self.optimizer,loss={"gender_y":self.gender_loss,"out_age_y":self.age_loss_R}
                ,metrics={"gender_y":[self.g_acc_3,self.g_acc_5,self.g_acc_7],"one_age_y":self.MAE_R})
        # self.train_model.summary()
    def weighted_average(self,x):
        number_tensor=tf.expand_dims(tf.range(0,self.age_width,1),axis=0)
        number_tensor=tf.cast(number_tensor,tf.float32)
        y=tf.reduce_sum(number_tensor*x)
        return y
    def age_loss_R(self,y_true,y_pred):
        loss=(y_true-y_pred)**2
        loss=tf.reduce_mean(loss)
        return loss
    def age_loss_C(self,y_true,y_pred):
        loss=categorical_crossentropy(y_true,y_pred)
        loss=tf.reduce_mean(loss)
        return loss
    def gender_loss(self,y_true,y_pred):
        loss=binary_crossentropy(y_true,y_pred)
        loss=tf.reduce_mean(loss)
        return loss
    def MAE_R(self,y_true,y_pred):#acc for age regression
        error=tf.abs(y_true-y_pred)
        acc=tf.reduce_mean(error)
        return acc
    def MAE_C(self,y_true,y_pred):#acc for age classification
        y_true=tf.argmax(y_true,axis=-1)
        y_pred=tf.argmax(y_pred,axis=-1)
        error=tf.abs(y_true-y_pred)
        acc=tf.reduce_mean(error)
        return acc
    def g_acc_3(self,y_true,y_pred):
        pos_y=tf.to_float(y_pred>0.3)
        equal_y=tf.where(tf.equal(y_true,pos_y),tf.ones(tf.shape(y_pred)),tf.zeros(tf.shape(y_pred)))
        acc=tf.reduce_mean(equal_y)
        return acc
    def g_acc_5(self,y_true,y_pred):
        pos_y=tf.to_float(y_pred>0.5)
        equal_y=tf.where(tf.equal(y_true,pos_y),tf.ones(tf.shape(y_pred)),tf.zeros(tf.shape(y_pred)))
        acc=tf.reduce_mean(equal_y)
        return acc
    def g_acc_7(self,y_true,y_pred):
        pos_y=tf.to_float(y_pred>0.7)
        equal_y=tf.where(tf.equal(y_true,pos_y),tf.ones(tf.shape(y_pred)),tf.zeros(tf.shape(y_pred)))
        acc=tf.reduce_mean(equal_y)
        return acc
    def predict(self,img_name,load_weight=False):
        if load_weight:
            self.train_model.load_weights(self.model_path)
        img_path = './test_photo/'+img_name
        if not os.path.isfile(img_path):
            print("img doesn't exist")
            exit()
        origin_x = cv2.imread(img_path,-1)
        x=cv2.resize(origin_x,self.test_size)
        x = resnet50_preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        preds = self.pred_model.predict(x)
        show_result([origin_x],preds,save=True,show=True)
    def train(self,epoch,keep_train=False):
        if keep_train:
            self.train_model.load_weights(self.model_path)
        if self.model_type=="soft":
            one_hot_flag=True
        else:
            one_hot_flag=False
        data_gen=data_generator("age_gender_UTK",batch_size=self.batch_size,one_hot_gender=one_hot_flag
            ,random_crop=True,random_mirror=True,random_width=True,random_rotate=True,random_size=True,shuffle=True)
        input_generator=data_gen.generator()
        step=data_gen.get_max_batch_index()
        lr_changer=LearningRateScheduler(self.lr_scheduler)
        pred_test=LambdaCallback(on_epoch_end=self.pred_test)
        validate=LambdaCallback(on_epoch_end=self.validate)
        print("start train")
        self.train_model.fit_generator(input_generator,steps_per_epoch=step,epochs=epoch,
            callbacks=[lr_changer,validate,pred_test])
        self.train_model.save_weights(self.model_path)
    def lr_scheduler(self,epoch):
        # if epoch==0:
        #     self.origin_lr=self.lr
        # elif epoch==50:
        #     self.lr=self.origin_lr*0.1
        # elif epoch<80:
        #     self.lr=self.lr*0.97
        now_lr=self.lr
        self.train_model.save_weights(self.model_path)
        return now_lr
    def validate(self,epoch,logs,load_weight=False):
        if load_weight:
            self.train_model.load_weights(self.model_path)
        if self.model_type=="soft":
            one_hot_flag=True
        else:
            one_hot_flag=False
        data_gen=data_generator("age_gender_appa_val",batch_size=16,one_hot_gender=one_hot_flag)
        input_generator=data_gen.test_data()
        step=data_gen.get_max_batch_index()
        eval = self.train_model.evaluate_generator(input_generator,steps=step)
        for index,name in enumerate(self.pred_model.metrics_names):
            print("%s:%.4f"%(name,eval[index]),end=" ")
        print()
    def pred_test(self,epoch,logs,load_weight=False):
        if load_weight:
            self.train_model.load_weights(self.model_path)
        if self.model_type=="soft":
            one_hot_flag=True
        else:
            one_hot_flag=False
        data_gen=data_generator("age_gender_appa_test",batch_size=64,one_hot_gender=one_hot_flag)#batch_size means number of tested photos
        input_generator=data_gen.test_data()
        for i in range(2):
            temp=time.time()
            # preprocess_data,origin_data=data_gen.pick_data(start_index=0)
            # preds = self.pred_model.predict(preprocess_data)
            (imgs,dict)=next(input_generator)
            preds = self.pred_model.predict(imgs)
            print(time.time()-temp)
            # show_result(origin_data,preds,save=True,show=False)
