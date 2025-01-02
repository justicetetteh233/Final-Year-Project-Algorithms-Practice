# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:43:43 2023

@author: User
"""
from binaryOptimizer.model.algorithms import HBEOSA
from binaryOptimizer.model.algorithms.HBEOSA import hbeosa
from binaryOptimizer.model.algorithms.BEOSA import beosa
import numpy as np
import random
import pytest
from copy import deepcopy
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

def problem(lb, ub, model, x_train, y_train):  
    def fitness_function(solution):
        nonlocal model
        pos=np.min(solution)
        pos=int(pos)# if int(pos)%2==0 else int(pos)+1
        x=x_train[:, :pos]
        y=y_train[:pos]                 
        y_true=y
        y= [max(range(len(label)), key=label.__getitem__) for label in y] 
        from sklearn.tree import DecisionTreeClassifier# training a DescisionTreeClassifier
        dtree_model = DecisionTreeClassifier(max_depth = 2).fit(x, y)
        dtree_predictions = dtree_model.predict(x)
        y_pred=tf.keras.utils.to_categorical(dtree_predictions, num_classes=5)
        acc=np.sum(y_pred==y_true)/len(y_pred)
        #print(acc) # ACCURCY
        return acc

    problem = {
        "fit_func": fitness_function,
        "lb": [lb, lb, lb, lb, lb],
        "ub": [ub, ub, ub, ub, ub],
        "minmax": "min",
    }
    return problem

def make_prediction(new_x, x, y):
    y_true=y
    y= [max(range(len(label)), key=label.__getitem__) for label in y] 
    
    results, optimizedresults=[], []
    from sklearn.tree import DecisionTreeClassifier # training a DescisionTreeClassifier
    dtree_model = DecisionTreeClassifier(max_depth = 2).fit(x, y)
    dtree_predictions = dtree_model.predict(x)
    tree_cm = confusion_matrix(y, dtree_predictions)# creating a confusion matrix
    cr=classification_report(y, dtree_predictions)
    y_pred=tf.keras.utils.to_categorical(dtree_predictions, num_classes=5)
    tree_acc=(np.sum(y_pred==y_true)/len(y_pred)) # ACCURCY
    recall = cross_val_score(dtree_model, x, y, cv=5, scoring='recall')
    precision = cross_val_score(dtree_model, x, y, cv=5, scoring='precision')
    f1 = cross_val_score(dtree_model, x, y, cv=5, scoring='f1')
    results.append(['Decision Tree', tree_acc, precision, recall, f1, cr, tree_cm])
    
    dtree_model = DecisionTreeClassifier(max_depth = 2).fit(new_x, y)
    dtree_predictions = dtree_model.predict(new_x)
    tree_cm = confusion_matrix(y, dtree_predictions)# creating a confusion matrix
    cr=classification_report(y, dtree_predictions)
    y_pred=tf.keras.utils.to_categorical(dtree_predictions, num_classes=5)
    tree_acc=(np.sum(y_pred==y_true)/len(y_pred)) # ACCURCY
    recall = cross_val_score(dtree_model, new_x, y, cv=5, scoring='recall')
    precision = cross_val_score(dtree_model, new_x, y, cv=5, scoring='precision')
    f1 = cross_val_score(dtree_model, new_x, y, cv=5, scoring='f1')
    optimizedresults.append(['Optimized Decision Tree', tree_acc, precision, recall, f1, cr, tree_cm])
    
    
    from sklearn.neighbors import KNeighborsClassifier# training a KNN classifier
    knn = KNeighborsClassifier(n_neighbors = 7).fit(x, y)
    knn_acc = knn.score(x, y)# accuracy on X_test
    knn_predictions = knn.predict(x)# creating a confusion matrix
    cr=classification_report(y, knn_predictions)
    knn_cm = confusion_matrix(y, knn_predictions)
    recall = cross_val_score(knn, x, y, cv=5, scoring='recall')
    precision = cross_val_score(knn, x, y, cv=5, scoring='precision')
    f1 = cross_val_score(knn, x, y, cv=5, scoring='f1')
    results.append(['KNN', knn_acc, precision, recall, f1, cr, knn_cm])
    
    knn = KNeighborsClassifier(n_neighbors = 7).fit(new_x, y)
    knn_acc = knn.score(new_x, y)# accuracy on X_test
    knn_predictions = knn.predict(new_x)# creating a confusion matrix
    cr=classification_report(y, knn_predictions)
    knn_cm = confusion_matrix(y, knn_predictions)
    recall = cross_val_score(knn, new_x, y, cv=5, scoring='recall')
    precision = cross_val_score(knn, new_x, y, cv=5, scoring='precision')
    f1 = cross_val_score(knn, new_x, y, cv=5, scoring='f1')
    optimizedresults.append(['Optimize KNN', knn_acc, precision, recall, f1, cr, knn_cm])
    
    from sklearn.naive_bayes import GaussianNB # training a Naive Bayes classifier
    gnb = GaussianNB().fit(x, y)
    gnb_predictions = gnb.predict(x)        
    gaus_acc = gnb.score(x, y)# accuracy on X_test
    cr=classification_report(y, gnb_predictions)
    guas_cm = confusion_matrix(y, gnb_predictions)# creating a confusion matrix
    recall = cross_val_score(gnb, x, y, cv=5, scoring='recall')
    precision = cross_val_score(gnb, x, y, cv=5, scoring='precision')
    f1 = cross_val_score(gnb, x, y, cv=5, scoring='f1')
    results.append(['Naive Bayes', gaus_acc, precision, recall, f1, cr, guas_cm])
    
    gnb = GaussianNB().fit(new_x, y)
    gnb_predictions = gnb.predict(new_x)        
    gaus_acc = gnb.score(new_x, y)# accuracy on X_test
    cr=classification_report(y, gnb_predictions)
    guas_cm = confusion_matrix(y, gnb_predictions)# creating a confusion matrix
    recall = cross_val_score(gnb, new_x, y, cv=5, scoring='recall')
    precision = cross_val_score(gnb, new_x, y, cv=5, scoring='precision')
    f1 = cross_val_score(gnb, new_x, y, cv=5, scoring='f1')
    optimizedresults.append(['Optimize Naive Bayes', gaus_acc, precision, recall, f1, cr, guas_cm])
    '''
    from sklearn.svm import SVC
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(x, y)
    svm_predictions = svm_model_linear.predict(x)         
    svm_acc = svm_model_linear.score(x, y)# model accuracy for X_test        
    cr=classification_report(y, svm_predictions)
    svm_cm = confusion_matrix(y, svm_predictions)# creating a confusion matrix
    recall = cross_val_score(svm_model_linear, x, y, cv=5, scoring='recall')
    precision = cross_val_score(svm_model_linear, x, y, cv=5, scoring='precision')
    f1 = cross_val_score(svm_model_linear, x, y, cv=5, scoring='f1')
    results.append(['SVM', svm_acc, precision, recall, f1, cr, svm_cm])
    '''
    return results, optimizedresults

def solutions_2_feature_transform(x_train, AllSol):
    idx=0
    for individual in AllSol:
        #obtain a corresponding item in x_train
        c_individual=x_train[idx]
        #find all indexes of occurance of zero in individual
        zero_indexes=np.flatnonzero(individual==0)
        #locate the zero_indexes in individual and blind values at those index using 0
        #e.g individual[zero_indexes[i]]=0 item at that index is made 0; where i is used in a for-loop
        #NB: zero_indexes is an array containing the indexes of a numpy array (individual) which is to have 0
        individual[zero_indexes]=0
        #then increament so we move to the next individual in the solution space
        idx=idx+1
    return deepcopy(x_train)

def save_optimize_features(checkpoint_path, new_x_train, y_train):
    np.save(checkpoint_path+"new_x_train.npy", new_x_train)
    np.save(checkpoint_path+"new_train_labels.npy", y_train)
    
## Run the algorithm
def feature_optimizer(checkpoint_path, method, model, x_train, y_train, xeval, yeval, model_rates, lb, ub, pr, num_classes, runfilename, metrics_result_dir):
    prob=problem(lb, ub, model, x_train, y_train)
    MaxIter=50
    pop_size=x_train.shape[0]
    print('================== Population Size ======================')
    print(pop_size)
    
    if method=='HBEOSA-DMO':   
        allfit, allcost, testAcc, featCnt, gbest, AllSol=hbeosa(pop_size, MaxIter, x_train, y_train, xeval, yeval, model_rates, runfilename, metrics_result_dir, 'DMO', True)
    if method=='HBEOSA-DMO-NT':   
        allfit, allcost, testAcc, featCnt, gbest, AllSol=hbeosa(pop_size, MaxIter, x_train, y_train, xeval, yeval, model_rates, runfilename, metrics_result_dir, 'DMO', False)
    if method=='HBEOSA-PSO':   
        allfit, allcost, testAcc, featCnt, gbest, AllSol=hbeosa(pop_size, MaxIter, x_train, y_train, xeval, yeval, model_rates, runfilename, metrics_result_dir, 'PSO', True)
    if method=='HBEOSA-PSO-NT':   
        allfit, allcost, testAcc, featCnt, gbest, AllSol=hbeosa(pop_size, MaxIter, x_train, y_train, xeval, yeval, model_rates, runfilename, metrics_result_dir, 'PSO', False)
    if method=='BEOSA':   
        allfit, allcost, testAcc, featCnt, gbest, AllSol=beosa(pop_size, MaxIter, x_train, y_train, xeval, yeval, model_rates, runfilename, metrics_result_dir, None, False)
    
    #Note that AllSol represents the optimize array of arrays for 1s|0s
    #recall that x_train contains features for say N images with M features
    #e.g if pop_size=x_train.shape[0]=120, meaning N=120
    #and dim of each row in pop_size=x_train is 8, meaning M=8,
    #then
    '''
       AllSol=[
               [1, 0, 1, 1, 1, 0, ...upto the 8th 1s|0s], row #1 in x_train
               [1, 0, 0, 1, 0, 1, ...upto the 8th 1s|0s], row #2 in x_train
               [1, 0, 1, 0, 1, 1, ...upto the 8th 1s|0s], row #3 in x_train
               ...
               [0, 0, 1, 0, 1, 0, ...upto the 8th 1s|0s], row #N in x_train
              ]
       Then we assume that all places where 0s are we blind the corresponding features in x_train
       while were have 1s, the feature values are left as they are.
       We need a transformation function that we rewrite x_train before passing it to the prediction phase
       
    '''
    #Here is the transform function
    new_x_train = solutions_2_feature_transform(x_train, AllSol)
    
    #save the optimized features sets
    save_optimize_features(checkpoint_path, new_x_train, y_train)
    
    #apply the orginal features and the transform features for classfication
    results, optimizedresults=make_prediction(new_x_train, x_train, y_train)
    
    #Store results for original features
    pr._save_classifiers_results__(method, results, allfit, allcost, testAcc, featCnt, gbest)
    
    #Store results for optimized features
    pr._save_classifiers_results__(method, optimizedresults, allfit, allcost, testAcc, featCnt, gbest)
    
    return new_x_train, y_train
