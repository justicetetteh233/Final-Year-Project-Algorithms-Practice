# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:55:27 2023

@author: User
"""
#from binaryOptimizer.model.algorithms.BEOSA_FUSION import beosa_fusion
import numpy as np
import random
import pytest
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from numpy import array
#from utils.Config import *

def fusion_fitness_func(hprob, mprob):
    histo_real_prob=1.0
    mammo_real_prob=1.0
    multimodal_real_prob=histo_real_prob+mammo_real_prob
    diff=multimodal_real_prob - (hprob+mprob)
    return diff

def fusion_label_remap(y_test, y_pred, lablekeys, labelvals, namedclasses, labeltype):
    histo_lables={"N":['N'], "B":['B', 'A', 'F', 'PT', 'TA'], 
                  "M":['IS', 'IV', 'DC', 'LC', 'MC', 'PC']}
    mammo_lables={"N":['N'], "B":['BC', 'BM'], "M":['CALC', 'M']}
     
    n=0
    trueclsnames, predclsnames=[], []
    for ytrue, ypred in zip(y_test, y_pred):
        labeltrue=ytrue[n]
        index_max = max(range(len(labeltrue)), key=labeltrue.__getitem__)
        classnametrue=namedclasses[index_max]
        if labeltype=='histo':
            return None
        
        if labeltype=='mammo':
            for y, idx in zip(mammo_lables.keys(), mammo_lables.values()):
                return None
        trueclsnames.append(index_max)
        
        labelpred=ypred[n]
        
        n=n+1        
    
    categoricaltrue=array(trueclsnames)
    categoricaltrue = to_categorical(categoricaltrue, len(namedclasses))        
    return None

def fusion_create_population():
    
    return None