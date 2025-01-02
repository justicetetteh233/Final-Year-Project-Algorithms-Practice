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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier# training a DescisionTreeClassifier
from sklearn.naive_bayes import GaussianNB # training a Naive Bayes classifier
from sklearn.svm import SVC
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, make_scorer, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

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

def make_prediction(train_data, test_data, trainy, testy):
    print(train_data.shape)
    print(trainy.shape)
    import seaborn as sns
    from matplotlib import pyplot as plt
    import time
    from datetime import datetime
    import os
    
    mias_lables = ['N', 'BC', 'BM', 'CALC', 'M']    
    folder = './outputs/results/metrics/'
    
    
    kfold=5
    knnclf=KNeighborsClassifier(n_neighbors=kfold)
    knnclf.fit(train_data,trainy)
    knnval=knnclf.score(test_data,testy)
    #print(knnval)
    knnpred = knnclf.predict(test_data)         
    knncr=classification_report(testy, knnpred)
    #print(knncr)
    knncm = confusion_matrix(testy.argmax(axis=1), knnpred.argmax(axis=1))# creating a confusion matrix
    #print(knncm)
    ax= plt.subplot()
    sns.heatmap(knncm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    #ax.xaxis.set_ticklabels(mias_lables); ax.yaxis.set_ticklabels(mias_lables);
    plt.savefig(folder + 'knn/' +time.strftime("%Y_%m_%d_%H_%M_%S")+'.png')
    knnrecall = np.mean(cross_val_score(knnclf, test_data,testy, cv=kfold, scoring=make_scorer(recall_score, average='micro')))
    knnprecision = np.mean(cross_val_score(knnclf, test_data, testy, cv=kfold, scoring=make_scorer(precision_score, average='micro')))
    knnf1 = np.mean(cross_val_score(knnclf, test_data, testy, cv=kfold, scoring=make_scorer(f1_score, average='micro')))
    knnauc = np.mean(cross_val_score(knnclf, test_data, testy, cv=kfold, scoring=make_scorer(roc_auc_score, average='micro')))
    knnresult=[knnval, knnprecision, knnrecall, knnf1, knnauc, knncr, knncm]
    #print(knnresult)
    print('KNN' +datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(knncr))
    
    rforestclf = RandomForestClassifier(n_estimators=300)
    rforestclf.fit(train_data,trainy)
    rforestval=rforestclf.score(test_data,testy)
    rforestpred = rforestclf.predict(test_data)         
    rforestcr=classification_report(testy, rforestpred)
    rforestcm = confusion_matrix(testy.argmax(axis=1), rforestpred.argmax(axis=1))# creating a confusion matrix
    ax= plt.subplot()
    sns.heatmap(rforestcm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    #ax.xaxis.set_ticklabels(mias_lables); ax.yaxis.set_ticklabels(mias_lables);
    plt.savefig(folder + 'rf/' +time.strftime("%Y_%m_%d_%H_%M_%S")+'.png')
    rforestrecall = np.mean(cross_val_score(rforestclf, test_data,testy, cv=kfold, scoring=make_scorer(recall_score, average='micro')))
    rforestprecision = np.mean(cross_val_score(rforestclf, test_data, testy, cv=kfold, scoring=make_scorer(precision_score, average='micro')))
    rforestf1 = np.mean(cross_val_score(rforestclf, test_data, testy, cv=kfold, scoring=make_scorer(f1_score, average='micro')))
    rforestauc = np.mean(cross_val_score(rforestclf, test_data, testy, cv=kfold, scoring=make_scorer(roc_auc_score, average='micro')))
    rforestresult=[rforestval, rforestprecision, rforestrecall, rforestf1, rforestauc, rforestcr, rforestcm]
    #print(rforestresult)
    print('RF '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(rforestcr))
    
    dtreeclf = DecisionTreeClassifier(max_depth = 2)
    dtreeclf.fit(train_data,trainy)
    dtreeval = dtreeclf.score(test_data,testy)
    dtreepred = dtreeclf.predict(test_data)         
    dtreecr=classification_report(testy, dtreepred)
    dtreecm = confusion_matrix(testy.argmax(axis=1), dtreepred.argmax(axis=1))# creating a confusion matrix
    ax= plt.subplot()
    sns.heatmap(dtreecm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    #ax.xaxis.set_ticklabels(mias_lables); ax.yaxis.set_ticklabels(mias_lables);
    plt.savefig(folder + 'dt/' +time.strftime("%Y_%m_%d_%H_%M_%S")+'.png')
    dtreerecall = np.mean(cross_val_score(dtreeclf, test_data,testy, cv=kfold, scoring=make_scorer(recall_score, average='micro')))
    dtreeprecision = np.mean(cross_val_score(dtreeclf, test_data, testy, cv=kfold, scoring=make_scorer(precision_score, average='micro')))
    dtreef1 = np.mean(cross_val_score(dtreeclf, test_data, testy, cv=kfold, scoring=make_scorer(f1_score, average='micro')))
    dtreeauc = np.mean(cross_val_score(dtreeclf, test_data, testy, cv=kfold, scoring=make_scorer(roc_auc_score, average='micro')))
    dtreeresult=[dtreeval, dtreeprecision, dtreerecall, dtreef1, dtreeauc, dtreecr, dtreecm]
    #print(str(dtreeval)+'  '+str(dtreepred)+'  '+str(dtreeprecision)+'  '+str(dtreerecall)+'  '+str(dtreef1))
    print('DT '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(dtreecr))
    
    '''
    svmclf = SVC(kernel = 'linear', C = 1)
    svmclf.fit(train_data,trainy)
    svmpred = svmclf.predict(test_data)         
    svmval = svmclf.score(test_data, testy)# model accuracy for X_test        
    svmcr=classification_report(testy, svmpred)
    svmcm = confusion_matrix(testy.argmax(axis=1), svmpred.argmax(axis=1))# creating a confusion matrix
    ax= plt.subplot()
    sns.heatmap(svmcm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(mias_lables); ax.yaxis.set_ticklabels(mias_lables);
    plt.savefig(folder + 'svm/' +time.strftime("%Y_%m_%d_%H_%M_%S")+'.png')
    svmrecall = np.mean(cross_val_score(svmclf, test_data,testy, cv=kfold, scoring=make_scorer(recall_score, average='micro')))
    svmprecision = np.mean(cross_val_score(svmclf, test_data, testy, cv=kfold, scoring=make_scorer(precision_score, average='micro')))
    svmf1 = np.mean(cross_val_score(svmclf, test_data, testy, cv=kfold, scoring=make_scorer(f1_score, average='micro')))
    svmauc = np.mean(cross_val_score(svmclf, test_data, testy, cv=kfold, scoring=make_scorer(roc_auc_score, average='micro')))
    svmresult=[svmval, svmprecision, svmrecall, svmf1, svmauc, svmcr, svmcm]
    #print(str(svmval)+'  '+str(svmpred)+'  '+str(svmprecision)+'  '+str(svmrecall)+'  '+str(svmf1))
    print('SVM '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    '''
    svmresult=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    '''
    gnbclf = GaussianNB()
    gnbclf.fit(train_data,trainy)
    gnbval = gnbclf.score(test_data,testy)# accuracy on X_test
    gnbpred = gnbclf.predict(test_data)         
    gnbcr=classification_report(testy, gnbpred)
    gnbcm = confusion_matrix(testy.argmax(axis=1), gnbpred.argmax(axis=1))# creating a confusion matrix
    ax= plt.subplot()
    sns.heatmap(gnbcm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(mias_lables); ax.yaxis.set_ticklabels(mias_lables);
    plt.savefig(folder + 'gb/' +time.strftime("%Y_%m_%d_%H_%M_%S")+'.png')
    gnbrecall = np.mean(cross_val_score(gnbclf, test_data,testy, cv=kfold, scoring=make_scorer(recall_score, average='micro')))
    gnbprecision = np.mean(cross_val_score(gnbclf, test_data, testy, cv=kfold, scoring=make_scorer(precision_score, average='micro')))
    gnbf1 = np.mean(cross_val_score(gnbclf, test_data, testy, cv=kfold, scoring=make_scorer(f1_score, average='micro')))
    gnbauc = np.mean(cross_val_score(gnbclf, test_data, testy, cv=kfold, scoring=make_scorer(roc_auc_score, average='micro')))
    gnbresult=[gnbval, gnbprecision, gnbrecall, gnbf1, gnbauc, gnbcr, gnbcm]
    #print(str(gnbval)+'  '+str(gnbpred)+'  '+str(gnbprecision)+'  '+str(gnbrecall)+'  '+str(gnbf1))
    print('GB '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    '''
    gnbresult=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    mlpclf=MLPClassifier(alpha=0.001, hidden_layer_sizes=(500,200,100),max_iter=200,random_state=4)
    mlpclf.fit(train_data,trainy)
    mlpval=mlpclf.score(test_data,testy)
    mlppred = mlpclf.predict(test_data)         
    mlpcr=classification_report(testy, mlppred)
    mlpcm = confusion_matrix(testy.argmax(axis=1), mlppred.argmax(axis=1))# creating a confusion matrix
    ax= plt.subplot()
    sns.heatmap(mlpcm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    #ax.xaxis.set_ticklabels(mias_lables); ax.yaxis.set_ticklabels(mias_lables);
    plt.savefig(folder + 'mlp/' +time.strftime("%Y_%m_%d_%H_%M_%S")+'.png')
    mlprecall = np.mean(cross_val_score(mlpclf, test_data,testy, cv=kfold, scoring=make_scorer(recall_score, average='micro')))
    mlpprecision = np.mean(cross_val_score(mlpclf, test_data, testy, cv=kfold, scoring=make_scorer(precision_score, average='micro')))
    mlpf1 = np.mean(cross_val_score(mlpclf, test_data, testy, cv=kfold, scoring=make_scorer(f1_score, average='micro')))
    mlpauc = np.mean(cross_val_score(mlpclf, test_data, testy, cv=kfold, scoring=make_scorer(roc_auc_score, average='micro')))
    mlpresult=[mlpval, mlpprecision, mlprecall, mlpf1, mlpauc, mlpcr, mlpcm]
    #print(str(mlpval)+'  '+str(mlppred)+'  '+str(mlpprecision)+'  '+str(mlprecall)+'  '+str(mlpf1))
    print('MLP '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(mlpcr))
    all_outputs = {'knn': knnresult, 'rf':rforestresult, 'mlp':mlpresult, 'dt':dtreeresult, 'svm':svmresult, 'gnb':gnbresult}
    print(all_outputs)
    return all_outputs

    '''
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
    return results, optimizedresults
    '''

def solutions_2_feature_transform(x_train, AllSol):
    for idx, individual in enumerate(AllSol):
        #obtain a corresponding item in x_train
        c_individual=x_train[idx]
        #find all indexes of occurance of non_zero in individual. This returns an array of the indices
        zero_indexes=np.where(individual==0)
        #locate the zero_indexes in individual and preserve them while making others = 0
        #e.g individual[zero_indexes[i]]=0 item at that index is made 0; where i is used in a for-loop
        #NB: zero_indexes is an array containing the indexes of a numpy array (individual) which is to have 0
        for i in zero_indexes:
            c_individual[i]=0
        x_train[idx] = c_individual
    return deepcopy(x_train)

def save_optimize_features(checkpoint_path, new_x_train, y_train):
    np.save(checkpoint_path+"new_x_train.npy", new_x_train)
    np.save(checkpoint_path+"new_train_labels.npy", y_train)
    
## Run the algorithm
def feature_optimizer(checkpoint_path, method, model, x_train, y_train, xeval, yeval, model_rates, lb, ub, pr, num_classes, runfilename, metrics_result_dir):
    prob=problem(lb, ub, model, x_train, y_train)
    MaxIter=1
    pop_size=x_train.shape[0]
    print('================== Population Size ======================')
    print(pop_size)
    method='HBEOSA-PSO'
    
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
    #save_optimize_features(checkpoint_path, new_x_train, y_train)
    
    #apply the orginal features and the transform features for classfication
                                              #train_data, test_data, trainy, testy
    results, optimizedresults=make_prediction(new_x_train, x_train, y_train, y_train)
    
    #Store results for original features
    pr._save_classifiers_results__(method, results, allfit, allcost, testAcc, featCnt, gbest)
    
    #Store results for optimized features
    pr._save_classifiers_results__(method, optimizedresults, allfit, allcost, testAcc, featCnt, gbest)
    
    return new_x_train, y_train
