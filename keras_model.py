import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'import keras
import sys
import tensorflow as tf
import numpy as np
import math
from keras.applications import ResNet50
from keras.layers import Conv2D,Dense,Flatten,Activation,Lambda
from keras.optimizers import Adam
from keras.models import Model
from keras.losses import categorical_crossentropy,binary_crossentropy
from keras import regularizers
from data import *

weight_decay_rate=0.0001
class age_gender_classifier():
    def __init__(self,model_path,lr=0.0001,test_size=(64,64)):
        self.age_width=101
        self.lr=lr
        pretrain_model=ResNet50(input_shape=test_size+(3,),weights='imagenet',include_top=False)
        x=pretrain_model.input
        pretrain_output=pretrain_model.output
        pretrain_output=Flatten()(pretrain_output)
        temp_y=Dense(256,activation="relu"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate))(pretrain_output)
        temp_y=Dense(256,activation="relu"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_y)
        gender_y=Dense(1,activation="sigmoid"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_y)
        one_age_y=Dense(1,activation="sigmoid"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_y)
        softmax_age_y=Dense(self.age_width,activation="softmax"
            ,kernel_regularizer=regularizers.l2(weight_decay_rate))(temp_y)
        out_age=Lambda(self.weighted_average)(softmax_age_y)

        self.optimizer=Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.train_model=Model(inputs=x,outputs=[gender_y,one_age_y])
        # self.train_model=model(inputs=x,outputs=[gender_y,softmax_age_y])
        self.pred_model=Model(inputs=x,outputs=[gender_y,out_age])
        self.train_model.compile(optimizer=self.optimizer,loss=[self.gender_loss,self.age_loss])
        self.train_model.summary()
    def weighted_average(self,x):
        number_tensor=tf.expand_dims(tf.range(0,self.age_width,1),axis=0)
        number_tensor=tf.cast(number_tensor,tf.float32)
        y=tf.reduce_sum(number_tensor*x)
        return y
    def age_loss(self,y_true,y_pred):
        # loss=categorical_crossentropy(y_true,y_pred)
        loss=(y_true-y_pred)**2
        return loss
    def gender_loss(self,y_true,y_pred):
        loss=binary_crossentropy(y_true,y_pred)
        return loss
    def predict(self,img_name)
        self.model.load_weights(self.model_path)
        img_path = './test_photo/'+img_name+'.'+file_type
        if not os.path.isfile(img_path):
            print("img doesn't exist")
            exit()
        origin_x = cv2.imread(img_path,-1)
        x=cv2.resize(origin_x,self.test_size)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        preds = self.model.predict(x)