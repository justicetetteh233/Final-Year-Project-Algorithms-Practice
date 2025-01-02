# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:50:44 2021

@author: Oyelade
"""

import numpy as np
import random
import os


"""
Constants and values describing rates and variables
"""
'''
Settings from the paper
--------------------------------------------------------------------------------------------
 Notation       Definition                                                     Range of Value
--------------------------------------------------------------------------------------------
    π     Recruitment rate of susceptible human individuals                          Variable
    ŋ    Decay rate of Ebola virus in the environment                               (0, )
    α    Rate of hospitalization of infected individuals                               (0, 1)
        Disease-induced death rate of human individuals                               [0.4, 0.9]
    β1    Contact rate of infectious human individuals                               Variable
    β2    Contact rate of pathogen individuals/environment                           Variable
    β3    Contact rate of deceased human individuals                                   Variable
    β4    Contact rate of recovered human individuals                                   Variable
        Recovery rate of human individuals                                           (0, 1)
        Natural death rate of human individuals                                       (0, 1)
        Rate of  burial of deceased human individuals                               (0, 1)
        Rate of vaccination of individuals                                           (0, 1)
        Rate of response to hospital treatment                                       (0, 1)
        Rate response to vaccination                                               (0, 1)
'''
π=0.1 #Recruitment rate of susceptible human individuals
ŋ=np.random.rand() #Decay rate of Ebola virus in the environment
α=np.random.rand() #Rate of hospitalization of infected individuals
dis=random.uniform(0.4, 0.9)#Disease-induced death rate of human individuals
β_1=0.1#Contact rate of infectious human individuals
β_2=0.1#Contact rate of pathogen individuals/environment
β_3=0.1#Contact rate of deceased human individuals
β_4=0.1#Contact rate of recovered human individuals
rr=np.random.rand() #Recovery rate of human individuals
dr=np.random.rand() #Natural death rate of human individuals
br=np.random.rand() #Rate of  burial of deceased human individuals
vr=np.random.rand() #Rate of vaccination of individuals
hr=np.random.rand() #Rate of response to hospital treatment
vrr=np.random.rand() #Rate response to vaccination
qrr=np.random.rand()	#Rate of quarantine of infected individuals

modelrates = {
    "recruitment_rate": π,
    "decay_rate": ŋ,
    "hospitalization_rate": α,
    "disease_induced_death_rate": dis,
    "contact_rate_infectious": β_1,
    "contact_rate_pathogen": β_2,
    "contact_rate_deceased": β_3,
    "contact_rate_recovered": β_4,
    "recovery_rate": rr,
    "natural_death_rate": dr,
    "burial_rate": br,
    "vacination_rate": vr,
    "hospital_treatment_rate": hr,
    "vaccination_response_rate": vrr,
    "quarantine_rate": qrr
}
modelrates = {
    "recruitment_rate": π,
    "decay_rate": ŋ,
    "hospitalization_rate": α,
    "disease_induced_death_rate": dis,
    "contact_rate_infectious": β_1,
    "contact_rate_pathogen": β_2,
    "contact_rate_deceased": β_3,
    "contact_rate_recovered": β_4,
    "recovery_rate": rr,
    "natural_death_rate": dr,
    "burial_rate": br,
    "vacination_rate": vr,
    "hospital_treatment_rate": hr,
    "vaccination_response_rate": vrr,
    "quarantine_rate": qrr
}

'''
Paths to datasets files
'''
input_base_dir='/content/gdrive/MyDrive/MyProjects/data/Datasets/' #'../Dataset/' #

'''
'mias numpy files: 299X299'
all_mias_labels9 = os.path.join( "..", "..", "Dataset", 'miasNumpy', "train", "all_mias_labels9.npy")
all_mias_slices9 = os.path.join( "..", "..", "Dataset", 'miasNumpy', "train", "all_mias_slices9.npy")
mias_test_labels_enc= os.path.join( "..", "..", "Dataset", 'miasNumpy',"test", "test12_labels.npy")
mias_test_images = os.path.join( "..", "..", "Dataset", 'miasNumpy', "test", "test12_data.npy")
mias_val_labels_enc= os.path.join( "..", "..", "Dataset", 'miasNumpy',"val", "all_mias_labels.npy")
mias_val_images = os.path.join( "..", "..", "Dataset",  'miasNumpy',"val", "all_mias_slices.npy")


all_mias_labels9 = input_base_dir+'miasNumpy/train/all_mias_labels9.npy'
all_mias_slices9 = input_base_dir+'miasNumpy/train/all_mias_slices9.npy'
mias_test_labels_enc= input_base_dir+'miasNumpy/test/test12_labels.npy'
mias_test_images = input_base_dir+'miasNumpy/test/test12_data.npy'
mias_val_labels_enc= input_base_dir+'miasNumpy/val/all_mias_labels.npy'
mias_val_images = input_base_dir+'miasNumpy/val/all_mias_slices.npy'
'''

'ddsm tfrecords files: 299x299 images and labels in tfrecords format'
train_path_10 = os.path.join( "..", "..", "Dataset", "ddsmTFrecords", "training10_0", "training10_0.tfrecords")
train_path_11 = os.path.join( "..", "..", "Dataset", "ddsmTFrecords", "training10_1","training10_1.tfrecords")
train_path_12 = os.path.join( "..", "..", "Dataset", "ddsmTFrecords", "training10_2", "training10_2.tfrecords")
train_path_13 = os.path.join( "..", "..", "Dataset",  "ddsmTFrecords", "training10_3", "training10_3.tfrecords")
train_path_14 = os.path.join( "..", "..", "Dataset",  "ddsmTFrecords", "training10_4", "training10_4.tfrecords")

input_base_dir='/content/gdrive/MyDrive/MyProjects/data/'
'mias image files: 299x299'
mias_input_dataset=input_base_dir+'WaveletProcessed/mias2/'

'ddsm image files: 299x299'
ddsm_input_dataset=input_base_dir+'WaveletProcessed/ddsm/'

'histopathology image files'
histo_input_dataset=input_base_dir+'Datasets/'

'''
File and directory paths naming
'''
base_dir='./outputs/' #'/content/gdrive/MyDrive/MyProjects/Multimodal/outputs/'  #'./outputs/' #
histo_checkpoint_path=base_dir+'checkpoints/histo/'
mammo_checkpoint_path=base_dir+'checkpoints/mammo/'
models_path=base_dir+'models/'
histo_model_filename='trainedmodelhisto'
mammo_model_filename='trainedmodelmammo'
save_results_dir=base_dir+'results/' 
save_histo_results_dir=save_results_dir+'training/histo/'
save_mammo_results_dir=save_results_dir+'training/mammo/'
metrics_dir=save_results_dir+'metrics/'

'''
General parameter settings
'''
show=1
batch_size=32
log_mode=1 
number_of_runs=1
cnn_epoch=50
train_split=0.75
test_split=0.15
eval_split=0.10
number_of_cnn_solutions=1
train_using_histo='histology'
train_using_mammo_ddsm='ddsm'
train_using_mammo_mias='mias'
isCombineMammoDatasets=False

'''
Image sizes
'''
histo_img_size={"width": 224, "height":224}
mammo_img_size={"width": 299, "height":299}

'''
Image channels
'''
histo_num_channels=3
mammo_num_channels=1

'''
Image labels
'''
histo_classes={"N":0, #normal (BACH dataset) 
         "B":1, #benign (BACH dataset) 
         "IS":2, #in situ carcinoma (BACH dataset)
         "IV":3, #invasive carcinoma, (BACH dataset)
         "A":4, #adenosis as benign (BreakHis dataset)
         "F":5, #fibroadenoma as benign (BreakHis dataset)
         "PT":6, #phyllodes tumor as benign (BreakHis dataset)
         "TA":7, #tubular adenona as benign (BreakHis dataset)
         "DC":8, #carcinoma as malignant (BreakHis dataset)
         "LC":9, #lobular carcinoma as malignant (BreakHis dataset)
         "MC":10, #mucinous carcinoma as malignant (BreakHis dataset)
         "PC":11 #papillary carcinoma as malignant (BreakHis dataset)
        }
histo_named_classes=["N","B","IS", "IV","A","F","PT", "TA", "DC", "LC", "MC","PC"]

#Mias image class info
mammo_mias_named_classes=['N', 'BC', 'BM', 'CALC', 'M']
mammo_mias_classes={0:'N', 1:'BC', 2:'BM', 3:'CALC', 4:'M'}

#DDSM image class info
mammo_ddsm_named_classes=['N', 'BC', 'BM', 'CALC', 'M']
mammo_ddsm_classes={0:'N', 1:'BC', 2:'BM', 3:'CALC', 4:'M'}


#seven(7) classes in MIAS numpy version
mias_numpy_classes=['CALC', #Calcification 
                  'CIRC', #Well-defined /circumscribed masses
                  'SPIC',  #Spiculated masses,
                  'M', #Other, ill-defined masses  MISC
                  'ARCH', #Architectural distortion
                  'ASY', #Asymmetry
                  'N',  #Normal
                  ]


'''
Definition of hyper-parameters
'''
learning_rates={0:1e-00, 1:1e-01, 2:1e-02, 3:1e-03, 4:1e-04, 5:1e-05, 6:1e-06, 7:1e-07, 8:1e-08,
                9:5e-00, 10:5e-01, 11:5e-02, 12:5e-03, 13:5e-04, 14:5e-05, 15:5e-06, 16:5e-07, 17:5e-08}

optimizers={0:"SGD", 1:"Adam", 2:"RMSprop", 3:"Adagrad", 4:"Nestrov", 5:"Adadelta", 6:"Adamax", 7:"Momentum"}

activations={0:"relu", 1:"leakyrelu", 2:"waveletdecompfunc"}

pooling={0:"Max", 1:"Avg"}

regularizers={0:"L1", 1:"L2", 2:"L1L2"}

fcactivations={0:"softmax"}

lossfunc={0: 'categorical_crossentropy', 1: 'sparse_categorical_crossentropy', 2: 'binary_crossentropy'}


'''
Binary optimization algorithms to use for experimentation
'''
binaryalgorithms=[ #'HBEOSA-DMO', 'HBEOSA-DMO-NT', 'HBEOSA-PSO', 'HBEOSA-PSO-NT', 
                  'BEOSA']