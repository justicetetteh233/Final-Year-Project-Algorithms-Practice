# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:48:46 2023

@author: Oyelade
"""

import tensorflow as tf
from tensorflow.keras import backend as K  #from keras import backend as K
from numpy import array
from cnn.root.rootcnn import RootCNN
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from time import time
import os
import numpy as np
import cv2
import random


class InputProcessor(object):
    REGULARIZER_RATES=0.0002
    
    def __init__(self, data_params=None):
        self.num_classes=data_params["num_classes"]
        self.classes=data_params["class_names"]
        self.input_dataset=data_params['input_dataset']
        self.testing_dataset=data_params['testing_dataset']
        self.img_width=data_params["img_size"]["width"] 
        self.img_height=data_params["img_size"]["height"]
        self.num_channels=data_params["num_channels"]
        self.train_using=data_params["train_using"]
        self.train_split=data_params["train_split"]
        self.test_split=data_params["test_split"]
        self.eval_split=data_params["eval_split"]
        self.K=data_params["K"]
        self.x_train, self.y_train=[], []
        self.x_eval, self.y_eval=[], [],
        self.x_test, self.y_test=[], [],
        self.img_ids_train, self.img_ids_eval, self.img_ids_test=[], [], []
        
    def init_imgs(self, nFiles):
        self.K.set_image_data_format('channels_last')
        if self.K.image_data_format() == 'channels_first':
            dim=(3, self.img_width, self.img_height)
            img_data_array = np.empty((nFiles, self.num_channels, self.img_width, self.img_height))
            self.x_train, self.y_train=np.empty((0, self.num_channels, self.img_width, self.img_height)), np.empty((0, ))
            self.x_eval, self.y_eval=np.empty((0, self.num_channels, self.img_width, self.img_height)), np.empty((0, ))
            self.x_test, self.y_test=np.empty((0, self.num_channels, self.img_width, self.img_height)), np.empty((0, ))
        else:
            dim=(self.img_width, self.img_height, self.num_channels)
            img_data_array = np.empty((nFiles, self.img_width, self.img_height, self.num_channels))
            self.x_train, self.y_train=np.empty((0, self.img_width, self.img_height, self.num_channels)), np.empty((0, ))
            self.x_eval, self.y_eval=np.empty((0, self.img_width, self.img_height, self.num_channels)), np.empty((0, ))
            self.x_test, self.y_test=np.empty((0, self.img_width, self.img_height, self.num_channels)), np.empty((0, ))
        return img_data_array, dim
    
    def get_train_input(self):
        nFiles=len(os.listdir(self.input_dataset))
        img_data_array, dim=self.init_imgs(nFiles)            
        class_name = np.empty((nFiles, ))
        img_ids=[]
        num_imgs_per_label={}
        n=0
        prev_label=''
        keys=list(self.classes.keys())
        values=list(self.classes.values()) 
        for file in os.listdir(self.input_dataset):
            image_path= os.path.join(self.input_dataset, file)
            if self.num_channels > 1:
                image= cv2.imread( image_path, cv2.COLOR_BGR2RGB) #image= np.array(Image.open(image_path))
                image=cv2.resize(image, (self.img_width, self.img_height),interpolation = cv2.INTER_AREA)
            else:
                image= cv2.imread( image_path, cv2.COLOR_BGR2GRAY) #image= np.array(Image.open(image_path)) COLOR_BGR2GRAY
                image=cv2.resize(image, (self.img_width, self.img_height),interpolation = cv2.INTER_AREA)
                
            image=np.array(image)
            image = image.astype('float32')
            image=image.reshape(dim)
            image /= 255 
            
            # one hot encode
            if self.num_channels > 1:
                label=file.split('.')[0]
                label=label.split('_')[0]
            else:
                label=(file.split('_')[-1]).split('.')[0]
                
            if (prev_label != label and prev_label ==''):
                prev_label=label
                label_count=0        
                
            if (prev_label != label and prev_label !='') or (prev_label != label and n > 0):
                num_imgs_per_label[prev_label]=label_count
                prev_label=label
                label_count=1
            else:
                label_count=label_count+1
                
            if self.num_channels > 1:
                label = self.classes[label] #, self.num_classes)
            else:
                value_at_index = keys[values.index(label)]
                label = value_at_index
                
            img_data_array[n, :, :, :] = image
            class_name[n] = label
            n=n+1
        
        class_name=array(class_name)
        class_name = to_categorical(class_name, self.num_classes)
        
        self.y_train=array(self.y_train)
        self.y_train = to_categorical(self.y_train, self.num_classes)
        
        self.y_eval=array(self.y_eval)
        self.y_eval = to_categorical(self.y_eval, self.num_classes)
        
        self.y_test=array(self.y_test)
        self.y_test = to_categorical(self.y_test, self.num_classes)
        
        
        num_imgs_per_label[prev_label]=label_count
        clabel_count=0
        for clabel, count in num_imgs_per_label.items():
            train_split = int(self.train_split * count)//4 
            test_split = int(self.test_split * count)//4
            eval_split = int(self.eval_split * count)//4 
            
            #print(clabel, ":", count, ":", train_split, ":", eval_split, ":", test_split)
            train_stop=(clabel_count+train_split)
            self.x_train=np.concatenate((self.x_train,img_data_array[clabel_count:train_stop]))
            self.y_train=np.concatenate((self.y_train,class_name[clabel_count:train_stop]))
            self.img_ids_train=self.img_ids_train+img_ids[clabel_count:train_stop]
            
            eval_start=train_stop+1
            eval_end=(eval_start+eval_split)
            self.x_eval=np.concatenate((self.x_eval, img_data_array[eval_start:eval_end])) 
            self.y_eval=np.concatenate((self.y_eval, class_name[eval_start:eval_end]))       
            self.img_ids_eval=self.img_ids_eval+img_ids[eval_start:eval_end]
            
            test_start=eval_end+1
            test_end=(test_start+test_split)
            self.x_test=np.concatenate((self.x_test, img_data_array[test_start:test_end]))  
            self.y_test=np.concatenate((self.y_test, class_name[test_start:test_end]))
            self.img_ids_test=self.img_ids_test+img_ids[test_start:test_end]
            
            #print(clabel_count, "-", train_stop, ",", eval_start, "-", eval_end, ",", test_start, "-", test_end)
            clabel_count=clabel_count+count
        
        self.train_split = int(self.train_split * img_data_array.shape[0])//4
        self.test_split = int(self.test_split * img_data_array.shape[0])//4
        self.eval_split = int(self.eval_split * img_data_array.shape[0])//4
        
        self.x_train, self.y_train=img_data_array[:self.train_split], class_name[:self.train_split]
        
        eval_start=self.train_split+1
        eval_end=eval_start+self.eval_split
        self.x_eval, self.y_eval=img_data_array[eval_start:eval_end], class_name[eval_start:eval_end]       
        
        test_start=eval_end+1
        test_end=test_start+self.test_split
        self.x_test, self.y_test=img_data_array[test_start:test_end], class_name[test_start:test_end]  
        
        print(len(self.x_train), ";",  len(self.y_train))
        print(len(self.x_eval), ";",  len(self.y_eval))
        print(len(self.x_test), ";",  len(self.y_test))
        
    def get_training_data(self):
        return self.x_train, self.y_train
    
    def get_eval_data(self):
        return self.x_eval, self.y_eval
    
    def get_test_data(self):
        return self.x_test, self.y_test
    
    def set_training_data(self, x, y):
         self.x_train, self.y_train=np.concatenate(self.x_train,x), np.concatenate(self.y_train,x)
         return None
    
    def set_eval_data(self, x, y):
        self.x_eval, self.y_eval=np.concatenate(self.x_eval,x), np.concatenate(self.y_eval,y)
        return None
    
    def set_test_data(self, x, y):
        self.x_test, self.y_test=np.concatenate(self.x_test,x), np.concatenate(self.y_test,y)
        return None
    
    'Read all numpy files in dataset for training'
    def numpy_train_data(self, sess, bs, buf_size, NUM_CLASSES, img_width, img_height):
        train_files = self.get_mias_numpy_training_data()
        train_image = np.load(train_files[0])
        train_labels = np.load(train_files[1])
        train_dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))
        train_dataset = train_dataset.shuffle(buf_size).batch(bs).make_initializable_iterator()  
        image, label = train_dataset.get_next()
        label = tf.one_hot(label, NUM_CLASSES)
        image = tf.reshape(image, [bs, 1, img_width, img_height])
        sess.run(train_dataset.initializer)
        image=image.eval(session=sess)
        label=label.eval(session=sess)
        while True:
            yield image, label

    def numpy_validation_data(self, sess, bs, buf_size, NUM_CLASSES, img_width, img_height):
        validation_files = self.get_mias_numpy_validation_data()
        validation_image = np.load(validation_files[0])
        validation_labels = np.load(validation_files[1])
        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_image, validation_labels))
        validation_dataset = validation_dataset.batch(bs).make_initializable_iterator()  
        image, label = validation_dataset.get_next()
        label = tf.one_hot(label, NUM_CLASSES)
        image = tf.reshape(image, [bs, 1, img_width, img_height])
        sess.run(validation_dataset.initializer)
        image=image.eval(session=sess)
        label=label.eval(session=sess)
        while True:
            yield image, label

    def numpy_test_data(self, sess, bs, buf_size, NUM_CLASSES, img_width, img_height):
        test_files = self.get_mias_numpy_test_data()
        test_image = np.load(test_files[0])
        test_labels = np.load(test_files[1])
        test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_labels))
        test_dataset = test_dataset.shuffle(buf_size).batch(bs).make_initializable_iterator()  
        image, label = test_dataset.get_next()
        label = tf.one_hot(label, NUM_CLASSES)
        image = tf.reshape(image, [bs, 1, img_width, img_height])
        sess.run(test_dataset.initializer)
        image=image.eval(session=sess)
        label=label.eval(session=sess)        
        while True:
            yield image, label

    def get_mias_numpy_training_data(self):
        train_files = [self.all_mias_slices9, self.all_mias_labels9]
        return train_files

    def get_mias_numpy_test_data(self):
        test_files = [self.mias_test_images, self.mias_test_labels_enc]
        return test_files

    def get_mias_numpy_validation_data(self):
        validation_files = [self.mias_val_images, self.mias_val_labels_enc]
        return validation_files
