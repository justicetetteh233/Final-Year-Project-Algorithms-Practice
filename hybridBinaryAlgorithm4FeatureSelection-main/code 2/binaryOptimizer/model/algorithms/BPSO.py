import numpy as np
import pandas as pd
import random
import math,time,sys,os
from matplotlib import pyplot
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from utils.convergencePlot import save_results_to_csv
from model.root import *

def initialise_pop(popSize,dim, trainX, testX, trainy, testy):
    population=np.zeros((popSize,dim))
    minn = 1
    maxx = math.floor(0.8*dim)
    if maxx<minn:
        minn = maxx
    for i in range(popSize):
        random.seed(i**3 + 19 + 83*time.time() ) 
        no = random.randint(minn,maxx)
        if no == 0:
            no = 1
        random.seed(time.time()*37 + 29)
        pos = random.sample(range(0,dim-1),no)
        for j in pos:
            population[i][j]=1
    return population

def bpso(datasetname,popSize,MaxIter, runfilename, metrics_result_dir):
    df=pd.read_csv(datasetname)
    (a,b)=np.shape(df)
    #print(a,b)
    data = df.values[:,0:b-1]
    label = df.values[:,b-1]
    dimension = np.shape(data)[1] #particle dimension

    cross = 5
    test_size = (1/cross)
    trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size)
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

    x_axis = []
    y_axis = []

    population = initialise_pop(popSize,dimension, trainX, testX, trainy, testy) 
    velocity = np.zeros((popSize,dimension))
    # print(population)
    gbestVal = 1000
    gbestVec = np.zeros(np.shape(population[0])[0])

    pbestVal = np.zeros(popSize)
    pbestVec = np.zeros(np.shape(population))    
    
    for i in range(popSize):
        pbestVal[i] = 1000
    
    #Array to Hold Best 
    curve=np.zeros((MaxIter,1))
    allfits=[]
    allcost=[]
    
    gcost=0.1
    
    start_time = datetime.now()
    for curIter in range(MaxIter):
        popnew = np.zeros((popSize,dimension))
        #print(testX) 
        fitList, costs = allfit(population,trainX,testX,trainy,testy)
        #update pbest
        for i in range(popSize):
            if (fitList[i] < pbestVal[i]):
                pbestVal[i] = fitList[i]
                pbestVec[i] = population[i].copy()
                gcost=costs[i]
                

        #update gbest
        for i in range(popSize):
            if (fitList[i] < gbestVal):
                gbestVal = fitList[i]
                gbestVec = population[i].copy()
                gcost=costs[i]
        # print(gbestVec)
        #print("gbest: ",gbestVal,onecount(gbestVec))
        #update W
        W = WMAX - (curIter/MaxIter)*(WMAX - WMIN )
        # print("w: ",W)
        ychosen , zchosen = 0 , 0
        for inx in range(popSize):
            #inx <- particle index
            random.seed(time.time()+10)
            r1 = C1 * random.random()
            random.seed(time.time()+19)
            r2 = C2 * random.random()

            x = np.subtract(pbestVec[inx] , population[inx])
            y = np.subtract(gbestVec , population[inx])
            velocity[inx] = np.multiply(W,velocity[inx]) + np.multiply(r1,x) + np.multiply(r2,y)
            popnew[inx] = np.add(population[inx],velocity[inx])
            y, z = np.array([]), np.array([])
            for j in range(dimension):
                temp = sigmoid1(popnew[inx][j])
                if temp > 0.5:
                    y = np.append(y,1)
                else:
                    y = np.append(y,0)

                temp = sigmoid1c(popnew[inx][j])
                if temp > 0.5:
                    z = np.append(z,1)
                else:
                    z = np.append(z,0)
            yfit, yone_cost = fitness(y,trainX,testX,trainy,testy) 
            zfit, zone_cost= fitness(z,trainX,testX,trainy,testy)
            if yfit<zfit:
                ychosen += 1
                popnew[inx] = y.copy()
                gcost=yone_cost
            else:
                zchosen += 1
                popnew[inx] = z.copy()
                gcost=zone_cost
            
        # print("ychosen:",ychosen,"zchosen:",zchosen)

        print('Iteration ', str(curIter),  ': Best fit = ',  str(zfit))
        curve[curIter]=zfit
        allcost.append(gcost)
        population = popnew.copy()

    time_required = datetime.now() - start_time
    output = gbestVec.copy()
    #print(output)
    
    testAcc = test_accuracy(True, gbestVec, trainX, testX, trainy, testy)
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
    
    cols=np.flatnonzero(output)
    #print(cols)
    X_test=testX[:,cols]
    X_train=trainX[:,cols]
    #print(np.shape(feature))

    clf=KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train,trainy)
    val=clf.score(X_test, testy )
    #print(val,onecount(output))

    return curve, allcost, testAcc, max(output), gbestVal


omega = 0.9
C1 = 2
C2 = 2
WMAX = 0.9
WMIN = 0.4
