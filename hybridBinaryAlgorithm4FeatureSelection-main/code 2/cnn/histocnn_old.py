# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 16:50:44 2023

@author: Oyelade
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
#from tensorflow.python.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
from tensorflow.keras.models import Sequential
from cnn.root.rootcnn import RootCNN
from copy import deepcopy
import os
import numpy as np
from tensorflow.python.keras.engine import keras_tensor


class HistoCNN(RootCNN):
    REGULARIZER_RATES=0.0002
       
    def __init__(self, histocnn_paras=None):
        RootCNN.__init__(self, histocnn_paras)
        self.learning_rates=histocnn_paras["learning_rates"]
        self.optimizers=histocnn_paras["optimizers"]
        self.activations=histocnn_paras["activations"]
        self.pooling=histocnn_paras["pooling"]
        self.regularizers=histocnn_paras["regularizers"]
        self.fcactivations=histocnn_paras["fcactivations"]
        self.lossfunc=histocnn_paras["lossfunc"]
        self.cnn_epoch=histocnn_paras["cnn_epoch"]
        self.config()
        
    def build_architecture(self):  
        self.model = Sequential()
        inputs, input_shape=self.cnn_input(model_type=0, is_zeropad=True)
        self.model.add(inputs)
        
        #block 1
        self.model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', 
                           activation=self.get_activation_function(func=self.activations),
                           name='conv_1_1', kernel_regularizer=self.get_regularizer()))
        self.model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', 
                           activation=self.get_activation_function(func=self.activations),
                           name='conv_1_2', kernel_regularizer=self.get_regularizer()))
        self.model.add(self._2Dpool__(pnumber=1, pfilter=2, ptype=self.pooling))
        
        #block 2
        self.model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', 
                           activation=self.get_activation_function(func=self.activations),
                           name='conv_2_1', kernel_regularizer=self.get_regularizer()))
        self.model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', 
                           activation=self.get_activation_function(func=self.activations),
                           name='conv_2_2', kernel_regularizer=self.get_regularizer()))
        self.model.add(self._2Dpool__(pnumber=2, pfilter=2, ptype=self.pooling))
        
        #block 3
        self.model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', 
                           activation=self.get_activation_function(func=self.activations),
                           name='conv_2_3', kernel_regularizer=self.get_regularizer()))
        self.model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', 
                           activation=self.get_activation_function(func=self.activations),
                           name='conv_2_4', kernel_regularizer=self.get_regularizer()))
        self.model.add(self._2Dpool__(pnumber=3, pfilter=2, ptype=self.pooling))
        
        #block 4
        self.model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', 
                           activation=self.get_activation_function(func=self.activations),
                           name='conv_3_1', kernel_regularizer=self.get_regularizer()))
        self.model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', 
                           activation=self.get_activation_function(func=self.activations),
                           name='conv_3_2', kernel_regularizer=self.get_regularizer()))
        self.model.add(self._2Dpool__(pnumber=4, pfilter=2, ptype=self.pooling))
        
        #block 5
        self.model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', 
                           activation=self.get_activation_function(func=self.activations),
                           name='conv_3_3', kernel_regularizer=self.get_regularizer()))
        self.model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', 
                           activation=self.get_activation_function(func=self.activations),
                           name='conv_3_4', kernel_regularizer=self.get_regularizer()))
        self.model.add(self._2Dpool__(pnumber=5, pfilter=2, ptype=self.pooling))
        self.model.add(self.flaten())
        self.model.add(self.flaten())
        
        self.model=self.fully_dense(activation=self.fcactivations, dropout=0.5, model=self.model)
        self.architecture_summary(self.model, input_shape)
        self.compile_model(self.model, self.optimizers, self.learning_rates, self.lossfunc)
        return None
    
    