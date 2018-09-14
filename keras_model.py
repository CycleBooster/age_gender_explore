import keras
import sys
import tensorflow as tf
import numpy as np
import math
from keras.applications import ResNet50
from keras.layers import Conv2D,Dense,Activation,Lambda
from keras.optimizers import Adam
from keras import regularizers
weight_decay_rate=0.0001
class age_gender_classifier():
    def __init__(self,model_path,lr=0.0001,test_size=(64,64)):
        self.age_width=101
        self.lr=lr
        pretrain_model=ResNet50(input_shape=test_size+(3,),weights='imagenet',include_top=False)
        x=pretrain_model.input
        pretrain_output=pretrain_model.output
        temp_y=Dense(256,activation="relu"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate))(pretrain_output)
        temp_y=Dense(256,activation="relu"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_y)
        gender_y=Dense(1,activation="sigmoid"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_y)
        one_age_y=Dense(1,activation="sigmoid"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_y)
        softmax_age_y=Dense(age_width,activation="softmax"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_y)
        out_age=Lambda(self.weighted_average)(softmax_age_y)

        self.optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.train_model=model(inputs=x,outputs=[gender_y,one_age_y])
        # self.train_model=model(inputs=x,outputs=[gender_y,softmax_age_y])
        self.pred_model=model(inputs=x,outputs=[gender_y,out_age])
        self.optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.train_model.compile(optimizer=self.optimizer,loss=self.loss_fun)
        
    def weighted_average(self,x):
        number_tensor=tf.range(0,self.age_width,1)
        y=tf.reduced_sum(number_tensor*x)
        return y
    def age_loss(self,y_true,y_pred):
        