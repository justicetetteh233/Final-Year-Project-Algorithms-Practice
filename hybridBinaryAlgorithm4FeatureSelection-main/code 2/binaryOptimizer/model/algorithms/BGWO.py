import numpy as np
import random
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import math,time,sys
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from functools import partial
import seaborn as sns 
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from utils.convergencePlot import save_results_to_csv
from model.root import *

omega = 0.99

def bgwo(dataset, pop_size, MaxIter, runfilename, metrics_result_dir):
    df = pd.read_csv(dataset)
    a, b = np.shape(df)
    data = df.values[:,0:b-1]
    label = df.values[:,b-1]
    dimension = data.shape[1]
    
    cross = 5
    test_size = (1/cross)
    trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=(7+17*int(time.time()%1000)))
    testAcc=test_accuracy(False, None, trainX, testX, trainy, testy)
    itemknn={'type':'basic', 'classifier':'KNN', 'acc':testAcc['knn'][0], 'precision1':max(testAcc['knn'][1]), 'recall1':max(testAcc['knn'][2]), 'f11':max(testAcc['knn'][3]), 'auc1':max(testAcc['knn'][4]), 'precision':testAcc['knn'][1], 'recall':testAcc['knn'][2], 'f1':testAcc['knn'][3], 'auc':testAcc['knn'][4]}
    save_results_to_csv(itemknn, runfilename, metrics_result_dir)
    itemrf={'type':'basic', 'classifier':'RF', 'acc':testAcc['rf'][0], 'precision1':max(testAcc['rf'][1]), 'recall1':max(testAcc['rf'][2]), 'f11':max(testAcc['rf'][3]), 'auc1':max(testAcc['rf'][4]), 'precision':testAcc['rf'][1], 'recall':testAcc['rf'][2], 'f1':testAcc['rf'][3], 'auc':testAcc['rf'][4]}
    save_results_to_csv(itemrf, runfilename, metrics_result_dir)
    itemmlp={'type':'basic', 'classifier':'MLP', 'acc':testAcc['mlp'][0], 'precision1':max(testAcc['mlp'][1]), 'recall1':max(testAcc['mlp'][2]), 'f11':max(testAcc['mlp'][3]), 'auc1':max(testAcc['mlp'][4]),'precision':testAcc['mlp'][1], 'recall':testAcc['mlp'][2], 'f1':testAcc['mlp'][3], 'auc':testAcc['mlp'][4]}
    save_results_to_csv(itemmlp, runfilename, metrics_result_dir)
    itemdt={'type':'basic', 'classifier':'DTree', 'acc':testAcc['dt'][0], 'precision1':max(testAcc['dt'][1]), 'recall1':max(testAcc['dt'][2]), 'f11':max(testAcc['dt'][3]), 'auc1':max(testAcc['dt'][4]),'precision':testAcc['dt'][1], 'recall':testAcc['dt'][2], 'f1':testAcc['dt'][3], 'auc':testAcc['dt'][4]}
    save_results_to_csv(itemdt, runfilename, metrics_result_dir)
    itemsvm={'type':'basic', 'classifier':'SVM', 'acc':testAcc['svm'][0], 'precision1':max(testAcc['svm'][1]), 'recall1':max(testAcc['svm'][2]), 'f11':max(testAcc['svm'][3]), 'auc1':max(testAcc['svm'][4]),'precision':testAcc['svm'][1], 'recall':testAcc['svm'][2], 'f1':testAcc['svm'][3], 'auc':testAcc['svm'][4]}
    save_results_to_csv(itemsvm, runfilename, metrics_result_dir)
    itemgnb={'type':'final', 'classifier':'GNB', 'acc':testAcc['gnb'][0], 'precision1':max(testAcc['gnb'][1]), 'recall1':max(testAcc['gnb'][2]), 'f11':max(testAcc['gnb'][3]), 'auc1':max(testAcc['gnb'][4]),'precision':testAcc['gnb'][1], 'recall':testAcc['gnb'][2], 'f1':testAcc['gnb'][3], 'auc':testAcc['gnb'][4]}
    save_results_to_csv(itemgnb, runfilename, metrics_result_dir)
    
    pop = initialise(pop_size, dimension, trainX, testX, trainy, testy)
    #fitList, costs = allfit(pop,trainX,trainy,testX,testy)
    
    #Array to Hold Best Cost Values
    curve=np.zeros((MaxIter,1))
    allcost=[]
    gcost=0.1
    
    for n in range(MaxIter):
        fit = []
        incosts=[]
        for i in range(pop_size):
            mefit, mecost=fitness(pop[i], trainX, testX, trainy, testy)
            fit.append(mefit)
            incosts.append(mecost)
            
        ind = np.argsort(fit)
        alpha = pop[ind[0]]
        alpha_fit = fit[ind[0]]
        beta = pop[ind[1]]
        beta_fit = pop[ind[1]]
        delta = pop[ind[2]]
        delta_fit = fit[ind[2]]
        
        a=2-n*((2)/MaxIter); # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        for i in range(pop_size):
            for j in range (dimension):     
                           
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)
                
                D_alpha=abs(C1*alpha[j]-pop[i,j]); # Equation (3.5)-part 1
                X1=alpha[j]-A1*D_alpha; # Equation (3.6)-part 1
                           
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2*beta[j]-pop[i,j]); # Equation (3.5)-part 2
                X2=beta[j]-A2*D_beta; # Equation (3.6)-part 2       
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)
                
                D_delta=abs(C3*delta[j]-pop[i,j]); # Equation (3.5)-part 3
                X3=delta[j]-A3*D_delta; # Equation (3.5)-part 3             
                
                pop[i,j]=(X1+X2+X3)/3  # Equation (3.7)

                if (sigmoid(pop[i][j]) > random.random()):
                    pop[i][j] = 1
                else:
                    pop[i][j] = 0
                    
        curve[n]=fit[ind[0]]
        gcost=incosts[0]
        allcost.append(gcost)
        
    ind = np.argsort(fit)
    bestpop = pop[ind[0]]
    bestfit = fit[ind[0]]
    
    testAcc = test_accuracy(True, bestpop, trainX, testX, trainy, testy)
    itemknn={'type':'final', 'classifier':'KNN', 'acc':testAcc['knn'][0], 'precision1':max(testAcc['knn'][1]), 'recall1':max(testAcc['knn'][2]), 'f11':max(testAcc['knn'][3]), 'auc1':max(testAcc['knn'][4]), 'precision':testAcc['knn'][1], 'recall':testAcc['knn'][2], 'f1':testAcc['knn'][3], 'auc':testAcc['knn'][4]}
    save_results_to_csv(itemknn, runfilename, metrics_result_dir)
    itemrf={'type':'final', 'classifier':'RF', 'acc':testAcc['rf'][0], 'precision1':max(testAcc['rf'][1]), 'recall1':max(testAcc['rf'][2]), 'f11':max(testAcc['rf'][3]), 'auc1':max(testAcc['rf'][4]), 'precision':testAcc['rf'][1], 'recall':testAcc['rf'][2], 'f1':testAcc['rf'][3], 'auc':testAcc['rf'][4]}
    save_results_to_csv(itemrf, runfilename, metrics_result_dir)
    itemmlp={'type':'final', 'classifier':'MLP', 'acc':testAcc['mlp'][0], 'precision1':max(testAcc['mlp'][1]), 'recall1':max(testAcc['mlp'][2]), 'f11':max(testAcc['mlp'][3]), 'auc1':max(testAcc['mlp'][4]),'precision':testAcc['mlp'][1], 'recall':testAcc['mlp'][2], 'f1':testAcc['mlp'][3], 'auc':testAcc['mlp'][4]}
    save_results_to_csv(itemmlp, runfilename, metrics_result_dir)
    itemdt={'type':'final', 'classifier':'DTree', 'acc':testAcc['dt'][0], 'precision1':max(testAcc['dt'][1]), 'recall1':max(testAcc['dt'][2]), 'f11':max(testAcc['dt'][3]), 'auc1':max(testAcc['dt'][4]),'precision':testAcc['dt'][1], 'recall':testAcc['dt'][2], 'f1':testAcc['dt'][3], 'auc':testAcc['dt'][4]}
    save_results_to_csv(itemdt, runfilename, metrics_result_dir)
    itemsvm={'type':'final', 'classifier':'SVM', 'acc':testAcc['svm'][0], 'precision1':max(testAcc['svm'][1]), 'recall1':max(testAcc['svm'][2]), 'f11':max(testAcc['svm'][3]), 'auc1':max(testAcc['svm'][4]),'precision':testAcc['svm'][1], 'recall':testAcc['svm'][2], 'f1':testAcc['svm'][3], 'auc':testAcc['svm'][4]}
    save_results_to_csv(itemsvm, runfilename, metrics_result_dir)
    itemgnb={'type':'final', 'classifier':'GNB', 'acc':testAcc['gnb'][0], 'precision1':max(testAcc['gnb'][1]), 'recall1':max(testAcc['gnb'][2]), 'f11':max(testAcc['gnb'][3]), 'auc1':max(testAcc['gnb'][4]),'precision':testAcc['gnb'][1], 'recall':testAcc['gnb'][2], 'f1':testAcc['gnb'][3], 'auc':testAcc['gnb'][4]}
    save_results_to_csv(itemgnb, runfilename, metrics_result_dir)
    
    featCnt = onecnt(bestpop)
    #print("best agent: ", bestpop)
    #print("Test Accuracy: ", testAcc)
    #print("#Features: ", featCnt)
            
    return curve, allcost, testAcc, featCnt, bestpop

