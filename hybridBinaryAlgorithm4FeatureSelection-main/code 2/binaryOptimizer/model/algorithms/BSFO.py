import numpy as np
import random
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import math,time,sys
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from utils.convergencePlot import save_results_to_csv
from model.root import *

pp = 0.1 # parameter
A, epxilon = 4, 0.001
ID_MIN_PROBLEM = 0
ID_MAX_PROBLEM = -1
ID_POS = 0
ID_FIT = 1
omega = 0.9

def initialise_pop(partCount, dim, trainX, testX, trainy, testy):    
    population=np.zeros((partCount,dim))
    minn = 1
    maxx = math.floor(0.5*dim)
    fit = np.array([])
    if maxx<minn:
        maxx = minn + 1
        #not(c[i].all())
    
    for i in range(partCount):
        random.seed(i**3 + 10 + time.time() ) 
        no = random.randint(minn,maxx)
        if no == 0:
            no = 1
        random.seed(time.time()+ 100)
        pos = random.sample(range(0,dim-1),no)
        for j in pos:
            population[i][j]=1
        
        # print(population[i])
    #print(population.shape)
    for i in range(population.shape[0]):
        #fit = np.append(fit, fitness(population[i], trainX, testX, trainy, testy))
        fit = np.append(fit, fitness(population[i], trainX, testX, trainy, testy))

    list_of_tuples = list(zip(population, fit))
    return list_of_tuples
               
def adaptiveBeta(agent, trainX, testX, trainy, testy):
    bmin = 0.1 #parameter: (can be made 0.01)
    bmax = 1
    maxIter = 10 # parameter: (can be increased )
    agentFit = agent[1]
    agent = agent[0].copy()
    for curr in range(maxIter):
        neighbor = agent.copy()
        size = np.shape(neighbor)[0]
        neighbor = randomwalk(neighbor)

        beta = bmin + (curr / maxIter)*(bmax - bmin)
        for i in range(size):
            random.seed( time.time() + i )
            if random.random() <= beta:
                neighbor[i] = agent[i]
        neighFit = fitness(neighbor,trainX,testX,trainy,testy)
        if neighFit <= agentFit:
            agent = neighbor.copy()
            agentFit = neighFit
    return (agent,agentFit)

def bsfo(dataset, pop_size, MaxIter, runfilename, metrics_result_dir):
        #url = "https://raw.githubusercontent.com/Rangerix/UCI_DATA/master/CSVformat/BreastCancer.csv"
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
    
        s_size = int(pop_size / pp)
        sf_pop = initialise_pop(pop_size, dimension, trainX, testX, trainy, testy) 
        s_pop = initialise_pop(s_size, dimension, trainX, testX, trainy, testy) 
        sf_gbest = get_global_best(sf_pop, ID_FIT, ID_MIN_PROBLEM)
        s_gbest = get_global_best(s_pop, ID_FIT, ID_MIN_PROBLEM)
        
        temp = np.array([])
        #Array to Hold Best 
        curve=np.zeros((MaxIter,1))
        allfits=[]
        allcost=[]
        gcost=sf_pop[0][ID_FIT]
    
        for iterno in range(0, MaxIter):
            ## Calculate lamda_i using Eq.(7)
            ## Update the position of sailfish using Eq.(6)
            for i in range(0, pop_size):
                PD = 1 - len(sf_pop) / ( len(sf_pop) + len(s_pop) )
                lamda_i = 2 * np.random.uniform() * PD - PD
                sf_pop_arr = s_gbest[ID_POS] - lamda_i * ( np.random.uniform() *
                                        ( sf_gbest[ID_POS] + s_gbest[ID_POS] ) / 2 - sf_pop[i][ID_POS] )
                sf_pop_fit = sf_pop[i][ID_FIT]
                new_tuple = (sf_pop_arr, sf_pop_fit)
                
                sf_pop[i] = new_tuple
            ## Calculate AttackPower using Eq.(10)
            AP = A * ( 1 - 2 * (iterno) * epxilon )
            if AP < 0.5:
                alpha = int(len(s_pop) * AP )
                beta = int(dimension * AP)
                ### Random choice number of sardines which will be updated their position
                list1 = np.random.choice(range(0, len(s_pop)), alpha)
                for i in range(0, len(s_pop)):
                    if i in list1:
                        #### Random choice number of dimensions in sardines updated
                        list2 = np.random.choice(range(0, dimension), beta)
                        s_pop_arr = s_pop[i][ID_POS]
                        for j in range(0, dimension):
                            if j in list2:
                                ##### Update the position of selected sardines and selected their dimensions
                                s_pop_arr[j] = np.random.uniform()*( sf_gbest[ID_POS][j] - s_pop[i][ID_POS][j] + AP )
                        s_pop_fit = s_pop[i][ID_FIT]
                        new_tuple = ( s_pop_arr, s_pop_fit)
                        s_pop[i] = new_tuple
            else:
                ### Update the position of all sardine using Eq.(9)
                for i in range(0, len(s_pop)):
                    s_pop_arr = np.random.uniform()*( sf_gbest[ID_POS] - s_pop[i][ID_POS] + AP )
                    s_pop_fit = s_pop[i][ID_FIT]
                    new_tuple = (s_pop_arr, s_pop_fit)
                    s_pop[i] = new_tuple
            
            # population in binary
            # y, z = np.array([]), np.array([])
            # ychosen = 0
            # zchosen = 0
            # # print(np.shape(s_pop))
            for i in range(np.shape(s_pop)[0]):
                agent = s_pop[i][ID_POS]
                tempFit = s_pop[i][ID_FIT]
                random.seed(time.time())
                #print("agent shape :",np.shape(agent))
                y, z = np.array([]), np.array([])
                for j in range(np.shape(agent)[0]): 
                    random.seed(time.time()*200+999)
                    r1 = random.random()
                    random.seed(time.time()*200+999)
                    if sigmoid1(agent[j]) < r1:
                        y = np.append(y,0)
                    else:
                        y = np.append(y,1)

                yfit, one_cost = fitness(y, trainX, testX, trainy, testy)
                agent = deepcopy(y)
                tempFit = yfit
                
                new_tuple = (agent,tempFit)
                s_pop[i] = new_tuple
                
                
            ## Recalculate the fitness of all sardine
            # print("y chosen:",ychosen,"z chosen:",zchosen,"total: ",ychosen+zchosen)
            for i in range(0, len(s_pop)):
                s_pop_arr = s_pop[i][ID_POS]
                s_pop_fit = fitness(s_pop[i][ID_POS],trainX, testX, trainy, testy)
                new_tuple = (s_pop_arr, s_pop_fit)
                s_pop[i] = new_tuple

            # local search algo
            for i in range(np.shape(s_pop)[0]):
                new_tuple = adaptiveBeta(s_pop[i],trainX,testX,trainy,testy)
                s_pop[i] = new_tuple

            ## Sort the population of sailfish and sardine (for reducing computational cost)
            sf_pop = sorted(sf_pop, key=lambda temp: temp[ID_FIT])
            s_pop = sorted(s_pop, key=lambda temp: temp[ID_FIT])
            for i in range(0, pop_size):
                s_size_2 = len(s_pop)
                if s_size_2 == 0:
                    s_pop = initialise(s_pop, dimension, trainX, testX, trainy, testy)
                    s_pop = sorted(s_pop, key=lambda temp: temp[ID_FIT])
                for j in range(0, s_size):
                    ### If there is a better solution in sardine population.
                    if (sf_pop[i][ID_FIT] > s_pop[j][ID_FIT]).all():
                        sf_pop[i] = deepcopy(s_pop[j])
                        del s_pop[j]
                    break   #### This simple keyword helped reducing ton of comparing operation.
                            #### Especially when sardine pop size >> sailfish pop size
            
            # OBL
            # sf_pop = OBL(sf_pop, trainX, testX, trainy, testy)
            sf_current_best = get_global_best(sf_pop, ID_FIT, ID_MIN_PROBLEM)
            s_current_best = get_global_best(s_pop, ID_FIT, ID_MIN_PROBLEM)
            
            _, cst=fitness(s_gbest[ID_POS], trainX, testX, trainy, testy)
            if (sf_current_best[ID_FIT] < sf_gbest[ID_FIT]).all():
                sf_gbest = np.array(deepcopy(sf_current_best))
                _, cst=fitness(sf_gbest[ID_POS], trainX, testX, trainy, testy)
            
            if (isinstance(s_current_best[ID_FIT], tuple) and isinstance(s_gbest[ID_FIT], tuple) and 
                len(s_current_best[ID_FIT]) > 1 and len(s_gbest[ID_FIT]) > 1):
                if (s_current_best[ID_FIT][0] < s_gbest[ID_FIT][0]):
                    s_gbest = np.array(deepcopy(s_current_best))
                    _, cst=fitness(s_gbest[ID_POS], trainX, testX, trainy, testy)
            elif (isinstance(s_current_best[ID_FIT], tuple) and 
                  len(s_current_best[ID_FIT]) > 1):
                if (s_current_best[ID_FIT][0] < s_gbest[ID_FIT]):
                    s_gbest = np.array(deepcopy(s_current_best))
                    _, cst=fitness(s_gbest[ID_POS], trainX, testX, trainy, testy)
            elif (isinstance(s_gbest[ID_FIT], tuple) and len(s_gbest[ID_FIT]) > 1):
                if (s_current_best[ID_FIT] < s_gbest[ID_FIT][0]):
                    s_gbest = np.array(deepcopy(s_current_best))
                    _, cst=fitness(s_gbest[ID_POS], trainX, testX, trainy, testy)
            else:
                if (s_current_best[ID_FIT] < s_gbest[ID_FIT]):
                    s_gbest = np.array(deepcopy(s_current_best))
                    _, cst=fitness(s_gbest[ID_POS], trainX, testX, trainy, testy)
            gcost=cst
            
            print('Iteration ', str(iterno),  ': Best fit = ',  str(s_gbest[1])) 
            allfits.append(s_gbest[1])
            allcost.append(gcost)
        
        testAcc = test_accuracy(True, sf_gbest[ID_POS], trainX, testX, trainy, testy)
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
    
        featCnt = onecnt(sf_gbest[ID_POS])
        #print("Test Accuracy: ", testAcc)
        #print("#Features: ", featCnt)

        return allfits, allcost, testAcc, featCnt, sf_gbest[ID_POS]