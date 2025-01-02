import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
tf.enable_eager_execution()
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:48:46 2023

@author: Oyelade
"""

#import tensorflow as tf
from tensorflow.python.keras import backend as K
from utils.Configs import *
from utils.IOUtil import save_results_to_csv 
from utils.SavedResult import ProcessResult
from cnn.histocnn import HistoCNN
from cnn.mammocnn import MammoCNN
from cnn.inputpipeline import InputProcessor
from cnn.binaryoptimzers import feature_optimizer
from cnn.fusionlayer import *
import statistics
from binaryOptimizer.model.algorithms.HBEOSA import hbeosa
import gc #Garbage Collector

if __name__ == "__main__": 
    '''
     Train and store models: Building and Training Siamese CNN models 
    '''

    data_params={
            'num_classes':len(histo_classes), 'class_names':histo_classes, 
            'input_dataset':histo_input_dataset, 'testing_dataset':histo_input_dataset,
            'img_size':histo_img_size,  'num_channels':histo_num_channels,
            'train_split':train_split, 'test_split':test_split, 'eval_split':eval_split,
            'train_using':train_using_histo, 'K':K
            }
    
    #Load histo datasets
    histo_input=InputProcessor(data_params=data_params)
    histo_input.get_train_input()   
    
    '''
     HISTO Section
    '''
    opt, lr=optimizers[1], learning_rates[6]
    histocnn_paras={
            'learning_rates':lr,  'optimizers':opt, 'activations':activations[1], 
            'pooling':pooling[1], 'regularizers':regularizers[1], 'fcactivations':fcactivations[0], 
            'lossfunc':lossfunc[0], 'cnn_epoch':cnn_epoch, 'batch_size':batch_size,        
            'num_classes':len(histo_classes), 'class_names':histo_classes, 'img_size':histo_img_size,  
            'log_mode':log_mode, 'models_path':models_path, 'model_filename':histo_model_filename,
            'checkpoint_path':histo_checkpoint_path, 'checkpoint_epoch':41, 
            'fromCheckpoint':True, 'train_model': True,
            'save_results_dir':save_results_dir, 'show':show, 'num_channels':histo_num_channels,
            'K':K, 'input_source':histo_input, 
            "save_multimodal_results_dir":save_histo_results_dir,
            "experiment":"histocnn_eph{}_optz{}_lr{}".format(cnn_epoch,opt,lr),
            "filename":"histocnn_eph{}_optz{}_lr{}".format(cnn_epoch,opt,lr),
            'optimzed_result_file':"optimized_histo_eph{}_{}_lr{}".format(cnn_epoch,opt,lr),
           }
    
    #train HistoCNN
    hcnn=HistoCNN(histocnn_paras=histocnn_paras)
    hcnn.build_architecture()
    xtrain, ytrain=histo_input.get_training_data()
    xeval, yeval=histo_input.get_eval_data()
    pr=ProcessResult(params=histocnn_paras)#Saving trained and evaluated CNN model
    if histocnn_paras['train_model']:
        if histocnn_paras['fromCheckpoint']:
            training_outcome=hcnn.load_trained_model(xtrain, ytrain, xeval, yeval)
        else:
            training_outcome=hcnn.trainmodel(hcnn.model, xtrain, ytrain, xeval, yeval)
        
        #Predict with HistoCNN model
        prediction=hcnn.predict()
        #pr.save_results(training_outcome, prediction)
        
        #Extract and optimize features from HistoCNN model
        #hcnn.extract_vector_features(hcnn.model, xtrain, ytrain, xeval, yeval, 'pool_4')
    
    #if histocnn_paras['train_model']==True, then we have save features, 
    #otherwise, we simply load the stored features which the method below achieves
    lb, ub, x, y, tx, ty =hcnn.optimize_vector_features()    
    
    _, _, y_test, y_pred, _=prediction
    #collapse classes of histo into the 3-class model of the siemese CNN
    histo_probmap_3labels, histo_probsinfo, categoricaltrue=fusion_label_remap(y_test, y_pred, histo_classes.keys(), histo_classes.values(), histo_named_classes, 'histo') 
    item={'labels':histo_probmap_3labels, 'histo_probsinfo':histo_probsinfo, 'categoricaltrue':categoricaltrue}
    save_results_to_csv(item, 'histo_probmap_3labels', metrics_dir)
    '''
    #Send features to binary optimizer for optimization process
    save_optimized_path=histocnn_paras['checkpoint_path']
    #We are using the BEOSA method for the optimization, hence the selection of first binaryalgorithms[0]
    method = binaryalgorithms[0] #for method in binaryalgorithms:
    new_x_train, y_train=feature_optimizer(save_optimized_path, method, hcnn.model, x, y, tx, ty, modelrates, lb, ub, pr, histocnn_paras['num_classes'], 'histo', metrics_dir)
        
    #Predict using the model on the newly optimized features
    #Predict with HistoCNN model
    pr.log_filename="histocnn_optimizedfeatures_eph{}_optz{}_lr{}".format(cnn_epoch,opt,lr)
    prediction=hcnn.predict(x=new_x_train, y=y_train)
    pr.save_results(training_outcome, prediction)
        
    #Obtain the probaility values for prediction by using the predict_prob() and not predict() 
    #since the latter returns index of class label while the former returns the probabilites of prediction
    y_pred=hcnn.model.predict(new_x_train)
    #collapse classes of histo into the 3-class model of the siemese CNN
    histo_probmap_3labels, categoricaltrue=fusion_label_remap(y_train, y_pred, histo_classes.keys(), histo_classes.values(), histo_named_classes, 'histo') 
    item={'labels':histo_probmap_3labels, 'categoricaltrue':categoricaltrue}
    save_results_to_csv(item, 'histo_probmap_3labels', metrics_dir)
    '''
    del xtrain
    del ytrain
    del yeval
    del xeval
    gc.collect()

    '''
      MAMMO Section
    ''' 
    
    opt, lr=optimizers[1], learning_rates[6]
    #Load mammo mias datasets
    data_params={
            'num_classes':len(mammo_mias_classes), 'class_names':mammo_mias_classes, 
            'input_dataset':mias_input_dataset, 'testing_dataset':mias_input_dataset,
            'img_size':mammo_img_size,  'num_channels':mammo_num_channels,
            'train_split':train_split, 'test_split':test_split, 'eval_split':eval_split,
            'train_using':train_using_mammo_mias, 'K':K
            }
    mammo_input_mias=InputProcessor(data_params=data_params)
    mammo_input_mias.get_train_input()
    
    #Load mammo ddsm datasets
    data_params['num_classes']=len(mammo_ddsm_classes)
    data_params['class_names']=mammo_ddsm_classes 
    data_params['input_dataset']=ddsm_input_dataset 
    data_params['testing_dataset']=ddsm_input_dataset            
    data_params['train_using']=train_using_mammo_ddsm
    mammo_input_ddsm=InputProcessor(data_params=data_params)
    #mammo_input_ddsm.get_train_input()
    
    if isCombineMammoDatasets:
        x, y=mammo_input_mias.get_train_input()
        mammo_input_ddsm.set_training_data(x, y)
        x, y=mammo_input_mias.get_eval_data()
        mammo_input_ddsm.set_eval_data(x, y)
        x, y=mammo_input_mias.get_test_data()
        mammo_input_ddsm.set_test_data(x, y)
    
    #End of data reading and processing
    
    #Train MammoCNN
    opt, lr=optimizers[1], learning_rates[6]
    mamocnn_paras={
            'learning_rates':lr,  'optimizers':opt, 'activations':activations[1], 
            'pooling':pooling[1], 'regularizers':regularizers[1], 'fcactivations':fcactivations[0], 
            'lossfunc':lossfunc[0], 'cnn_epoch':cnn_epoch, 'batch_size':batch_size,        
            'log_mode':log_mode, 'models_path':models_path, 'model_filename':mammo_model_filename,
            'save_results_dir':save_results_dir, 'show':show, 'K':K, 
            'checkpoint_epoch':41, 'fromCheckpoint':True, 'train_model': True,
            
            'num_classes':len(mammo_mias_classes), 'class_names':mammo_mias_classes, 
            'img_size':mammo_img_size, 'num_channels':mammo_num_channels, 
            'input_source':mammo_input_mias,
            "experiment":"mammocnn_eph{}_optz{}_lr{}".format(cnn_epoch,opt,lr),
            "filename":"mammocnn_eph{}_optz{}_lr{}".format(cnn_epoch,opt,lr),
            'optimzed_result_file':"optimized_mammo_eph{}_{}_lr{}".format(cnn_epoch,opt,lr),
            
            'checkpoint_path':mammo_checkpoint_path, "save_multimodal_results_dir":save_mammo_results_dir,
            
           }
    
    mcnn=MammoCNN(mamocnn_paras=mamocnn_paras)
    mcnn.build_architecture()
    xtrain, ytrain=mammo_input_mias.get_training_data()
    xeval, yeval=mammo_input_mias.get_eval_data()
    pr=ProcessResult(params=mamocnn_paras)#Saving trained and evaluated CNN model
    
    if mamocnn_paras['train_model']:
        if mamocnn_paras['fromCheckpoint']:
            training_outcome=mcnn.load_trained_model(xtrain, ytrain, xeval, yeval)
        else:
            training_outcome=mcnn.trainmodel(mcnn.model, xtrain, ytrain, xeval, yeval)
        
         
        #Predict with HistoCNN model
        prediction=mcnn.predict()
        #pr.save_results(training_outcome, prediction)
        
        #Extract and optimize features from HistoCNN model
        mcnn.extract_vector_features(mcnn.model, xtrain, ytrain, xeval, yeval, 'pool_4')
    
    #if mamocnn_paras['train_model']==True, then we have save features, 
    #otherwise, we simply load the stored features which the method below achieves
    lb, ub, x, y, tx, ty=mcnn.optimize_vector_features()
    
    _, _, y_test, y_pred, _=prediction
    #collapse classes of histo into the 3-class model of the siemese CNN
    mammo_probmap_3labels, mammo_probsinfo, categoricaltrue=fusion_label_remap(y_test, y_pred, mammo_mias_classes.keys(), mammo_mias_classes.values(), mammo_mias_named_classes, 'mammo') 
    item={'labels':mammo_probmap_3labels, 'probsinfo':mammo_probsinfo, 'categoricaltrue':categoricaltrue}
    save_results_to_csv(item, 'mammo_probmap_3labels', metrics_dir)
    
    '''
    #Send features to binary optimizer for optimization process
    save_optimized_path=mamocnn_paras['checkpoint_path']
    #We are using the BEOSA method for the optimization, hence the selection of first binaryalgorithms[0]
    method = binaryalgorithms[0] #for method in binaryalgorithms:
    new_x_train, y_train=feature_optimizer(save_optimized_path, method, mcnn.model, x, y, tx, ty, modelrates, lb, ub, pr, mamocnn_paras['num_classes'], 'mammo', metrics_dir)
        
    #Predict using the model on the newly optimized features
    #Predict with MammoCNN model
    pr.log_filename="mammocnn_optimizedfeatures_eph{}_optz{}_lr{}".format(cnn_epoch,opt,lr)
    prediction=mcnn.predict(x=new_x_train, y=y_train)
    pr.save_results(training_outcome, prediction)
    
    #Obtain the probaility values for prediction by using the predict_prob() and not predict() 
    #since the latter returns index of class label while the former returns the probabilites of prediction
    y_pred=predict_prob(new_x_train)
    #collapse classes of histo into the 3-class model of the siemese CNN
    mammo_probmap_3labels, categoricaltrue=fusion_label_remap(y_train, y_pred, mammo_mias_classes.keys(), mammo_mias_classes.values(), mammo_named_classes, 'mammo') 
    item={'labels':mammo_probmap_3labels, 'categoricaltrue':categoricaltrue}
    save_results_to_csv(item, 'mammo_probmap_3labels', metrics_dir)
    '''
    
    '''
     FUSION of HisCNN and MammoCNN
    '''
    #generate the population of probs
    pop=generate_fusion_population(histo_probmap_3labels, mammo_probmap_3labels, histo_probsinfo, mammo_probsinfo, fusion_info_dir)
    probsmap_optimizer(pop, modelrates, pr, optimize_fusion_info_dir)