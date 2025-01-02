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
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from utils.convergencePlot import save_results_to_csv
from model.root import *
import time

omega =  0.99

## Simulated Anealing
def SA(agent, fitAgent, trainX, testX, trainy, testy):
    # initialise temprature
    T = 4*np.shape(agent)[0]; T0 = 2*np.shape(agent)[0];

    S = agent.copy();
    bestSolution = agent.copy();
    bestFitness = fitAgent;

    while T > T0:
        neighbor = randomwalk(S)
        neighborFitness, cost = fitness(neighbor, trainX, testX, trainy, testy)

        if neighborFitness < bestFitness:
            S = neighbor.copy()
            bestSolution = neighbor.copy()
            bestFitness = neighborFitness

        elif neighborFitness == bestFitness:
            if np.sum(neighbor) == np.sum(bestSolution):
                S = neighbor.copy()
                bestSolution = neighbor.copy()
                bestFitness = neighborFitness

        else:
            theta = neighborFitness - bestFitness
            if np.random.rand() < math.exp(-1*(theta/T)):
                S = neighbor.copy()

        T *= 0.925

    return bestSolution, bestFitness

def bsndo(dataset, pop_size, MaxIter, runfilename, metrics_result_dir):
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
    
    clf=KNeighborsClassifier(n_neighbors=5)
    clf.fit(trainX,trainy)
    val=clf.score(testX,testy)
    pop = initialise(pop_size, dimension, trainX, testX, trainy, testy)
    acc, cost=fitness(pop[0], trainX, testX, trainy, testy)
    best_agent, best_fit = pop[0], acc
    
    
    #Array to Hold Best Cost Values
    curve=np.zeros((MaxIter,1))
    allfits=[]
    allcost=[]
    cost=[cost]
    gcost=cost[0]
    
    BestSol=pop[0]
    for Iter in range(MaxIter):
        fit, allc2 = allfit(pop, trainX, testX, trainy, testy)
        allc2.tolist()
        best_i = bestAgent(fit[0]) # index of the best agent in population
        mo = np.mean(pop, axis=0)

        for i in range(pop_size):

            a = np.random.randint(pop_size)
            b = np.random.randint(pop_size)
            c = np.random.randint(pop_size)

            while a==i or a==b or a==c or b==c or b==i or c==i:

                a = np.random.randint(pop_size)
                b = np.random.randint(pop_size)
                c = np.random.randint(pop_size)

            if fit[a][0] < fit[i][0]:
                v1 = np.subtract(pop[a], pop[i])
            else:
                v1 = np.subtract(pop[i], pop[a])

            if fit[b][0] < fit[c][0]:
                v2 = np.subtract(pop[b], pop[c])
            else:
                v2 = np.subtract(pop[c], pop[b])

            random.seed(time.time()*10%7)
            if random.random() > 0.5:
                u = (1/3) * np.add(pop[i], np.add(pop[best_i], mo))

                delta = np.sqrt((1/3) * (np.add(np.square(np.subtract(pop[i], u)), np.add(np.square(np.subtract(pop[best_i], u)), np.square(np.subtract(mo, u))))))

                random.seed(time.time()*10%10)
                vc1 = np.random.rand(1, dimension)
                vc2 = np.random.rand(1, dimension)

                vc1 = vc1.flatten()
                vc2 = vc2.flatten()

                Z1 = np.sqrt(-1*np.log(vc2)) * np.cos(2*np.pi*vc1)
                Z2 = np.sqrt(-1*np.log(vc2)) * np.cos(2*np.pi*vc1 + np.pi)

                random.seed(time.time()*10%9)
                a = np.random.uniform()
                random.seed(time.time()*10%2)
                b = np.random.uniform()

                if a <= b:
                    eta = np.add(u, np.multiply(delta, Z1))
                else:
                    eta = np.add(u, np.multiply(delta, Z2))

                newsol = eta.copy()

            else:

                random.seed(time.time()*10%70)
                beta = np.random.uniform()
                v = np.add(pop[i], np.add(beta * np.random.uniform() * v1, (1-beta) * np.random.uniform() * v2))

                newsol = v.copy()


            for k in range(dimension):
                if(sigmoid(newsol[k]) < random.random()):
                    newsol[k] = 0
                else:
                    newsol[k] = 1

            newfit,  one_cost= fitness(newsol, trainX, testX, trainy, testy)

            if newfit < fit[i][0]:
                pop[i] = newsol.copy()
                fit[i][0] = newfit
                if(newfit < fit[best_i][0]):
                    fit[best_i][0] = newfit
                    pop[best_i] = newsol

        #########################################################
        # LOCAL SEARCH
        ########################################################
        for agentNum in range(pop_size):
            pop[agentNum], fit[agentNum][0] = SA(pop[agentNum], fit[agentNum][0], trainX, testX, trainy, testy)


        least_fit = min(fit)
        least_fit = float(least_fit)
        ### ADDED LATER ###
        if best_fit > least_fit:
            best_fit = least_fit
            gcost=cost[best_i]
        
        curve[Iter]=best_fit
        allfits.append(best_fit) 
        allcost.append(gcost)
        print('Iteration ', str(Iter),  ': Best fit = ',  str(best_fit))

    testAcc = test_accuracy(True, pop[best_i], trainX, testX, trainy, testy)
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
    
    featCnt = onecnt(pop[best_i])
    return curve, allcost, testAcc, featCnt, pop[best_i]
