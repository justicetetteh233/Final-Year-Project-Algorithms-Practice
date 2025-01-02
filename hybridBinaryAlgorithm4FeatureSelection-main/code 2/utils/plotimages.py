# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 12:21:25 2021

@author: 77132
"""
import os
from os import getcwd, path, makedirs
import cv2 as cv
from PIL import Image # importing PIL -- useful in reading/writing/viewing images
import math # importing math -- useful in tan inverse, cos(theta) etc
import numpy as np # importing numpy -- useful in matrix operations
import matplotlib.pyplot as plt # importing numpy -- useful in matrix operations
import sys #importing sys library
import collections


def readImage(path): #Read a single image in numpy format
        return np.array(Image.open(path)) #asarray
    
def load_data(img_dir):        
        imgformat = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
        image_files = sorted([os.path.join(img_dir, file)
                          for file in os.listdir(img_dir)
                          if file.endswith(tuple(imgformat))])
        imgs=[]
        labels=[]
        target_names=[]
        for imgfile in image_files:
            np_im=readImage(imgfile)
            np_im =np.expand_dims(np_im, axis=0)
            imgfile=os.path.split(imgfile) 
            imgfile=imgfile[-1]
            target=(imgfile.split('.')[0]).split('_')[-1]
            if not(target in target_names):
                target_names.append(target)
            imgs.append(np_im)
            labels.append(target)
        return imgs, labels, target_names

def plotdatasetscount(imgs, labels, target_names):    
    plot_type=['Non-malignant', 'Malignant vs Non', 'Malignant', 'All classes']
    CLASS_NAMES={ 'Non-malignant':['N', 'BM','BC',], 
                 'Malignant vs Non':(['N', 'BM','BC',], ['CALC', 'M']),
                 'Malignant':['CALC', 'M'], 
                 'All classes':['N', 'BM','BC','CALC', 'M']
                 }
    
    labels= sorted(labels)
    target_names= sorted(target_names)
    values = collections.Counter(labels)# getting the elements frequencies using Counter class
    fig = plt.figure(figsize=(10,10))
    # loop on variables
    for i in range(4):
        # create subplot 
        plt.subplot(2,2,i+1)
        # select the variable of interest from the data
        tmplabels=plot_type[i]
        tmpvalues = {}
        if i==1:
            tmplabels1, tmplabels2=CLASS_NAMES[tmplabels]
            tmplabels=['Malign', 'Non-Malign']
            value1=0
            value2=0
            for key, value in values.items():
                print(str(key)+'  '+str(value))
                if key in CLASS_NAMES['Malignant']:
                    value1=value1+value
                if key in CLASS_NAMES['Non-malignant']:
                    value2=value2+value
            tmpvalues[tmplabels[0]]=value1
            tmpvalues[tmplabels[1]]=value2
            
        else:
            tmplabels=CLASS_NAMES[tmplabels]
            for key, value in values.items():                
                if key in tmplabels:
                    print(str(key)+'  '+str(value))
                    tmpvalues[key]=value
                
        # define histogram binning. 
        print(tmpvalues)
        mins=min(tmpvalues.values()) #np.min(tmpvalues) , key=(lambda k: tmpvalues[k])
        maxs=max(tmpvalues.values()) #np.max(tmpvalues), key=(lambda k: tmpvalues[k])
        print(str(mins)+'  '+str(maxs))
        bins = np.linspace( mins, maxs, 20)
        # loop on categories
        x=[]
        for j in range(len(tmplabels)): #np.unique()
            # select values for this category
            categ_values = tmpvalues[tmplabels[j]]
            print(str(tmplabels[j])+'  '+str(categ_values))
            x.append(categ_values)
            # plot histogram
        #q25, q75 = np.percentile(x, [0.25, 0.75])
        #bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
        #bins = round((maxs - mins) / bin_width)
            #plt.hist(categ_values, bins=bins, density=True, alpha=0.5, label=tmplabels[j])
        plt.bar(tmplabels, x, width=0.5, )#color=['black', 'orange', 'green', 'blue', 'cyan'])
        plt.title(plot_type[i])
        #plt.ylabel('Class frequency')
        plt.xlabel('Class');
    plt.legend()

def plotImagesAndLabels(imgs, labels):
    fig, m_axs = plt.subplots(4, 4, figsize = (8, 8))
    for (c_x, c_y, c_ax) in zip(imgs, labels, m_axs.flatten()):        
        c_x=np.squeeze(c_x, 0)
        c_ax.imshow(c_x, cmap = 'bone', vmin = -1.5, vmax = 1.5)#c_ax.imshow(c_x, cmap = 'gray', vmin=0, vmax=255)
        c_ax.set_title(c_y)
        c_ax.axis('off')
        

input_dataset='../data/processed/mias/'
imgs, labels, target_names=load_data(input_dataset)
plotdatasetscount(imgs, labels, target_names)
#input_dataset='../data/processed/plots4show/imgs/'
#imgs, labels, target_names=load_data(input_dataset)
#plotImagesAndLabels(imgs, labels)
#input_dataset='../data/processed/plots4show/gans/'
#imgs, labels, target_names=load_data(input_dataset)
#plotImagesAndLabels(imgs, labels)
