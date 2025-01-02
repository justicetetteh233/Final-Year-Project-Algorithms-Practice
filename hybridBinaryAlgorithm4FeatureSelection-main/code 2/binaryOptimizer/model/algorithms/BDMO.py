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
        neighborFitness = fitness(neighbor, trainX, testX, trainy, testy)

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

def RouletteWheelSelection(P):
    C=sum(P)
    r = np.random.uniform(low=0, high=C)
    for idx, f in enumerate(P):
            r = r + f
            if r > C:
                return idx
    return np.random.choice(range(0, len(P)))
    #r=np.random.rand()  #random.randint
    #i=np.where(r <= C, 1, r) #find(r<=C,1,'first')   #np.nonzero(x)
    #return i

def bdmo(dataset, pop_size, MaxIter, isSA, runfilename, metrics_result_dir):
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

    #Variable initialaization
    nVar=dimension             #Number of Decision Variables
    VarSize=[]           #Decision Variables Matrix Size  1:nVar
    VarMin=0             #Decision Variables Lower Bound
    VarMax=1             #Decision Variables Upper Bound
    nBabysitter= 3           #Number of babysitters
    nAlphaGroup=pop_size - nBabysitter         #Number of Alpha group
    nScout=nAlphaGroup         #Number of Scouts
    L=round(0.6*nVar*nBabysitter)  #Babysitter Exchange Parameter 
    peep=1             #Alpha female \.12s vocalization 
    tau=random.uniform(0, 1)
    sm=[]
    TestaccG=None
    
    #Population iniitalization and fitting, mongoose strucure initialization
    #Empty Mongoose Structure
    Position=[]
    Cost=[]
    Acc=[]
    pop = initialise(pop_size, dimension, trainX, testX, trainy, testy)
    Position=pop
    Acc, Cost=BDMO_allfit(pop, trainX, testX, trainy, testy)
    bestagent_acc, best_fit = BDMO_fitness(pop[0], trainX, testX, trainy, testy)
    best_agent= pop[0]
    gcost=Cost[0]
    
    #Abandonment Counter
    C=np.zeros((nAlphaGroup,1))
    Iter=1
    CF=(1-Iter/MaxIter) ** (2*Iter/MaxIter)  #np.linalg.matrix_power((1-Iter/MaxIter), (2*Iter/MaxIter))

    #Array to Hold Best Cost Values
    curve=np.zeros((MaxIter,1))
    allfit=[]
    allcost=[]
    
    BestSol=pop[0]
    #DMOA Main Loop
    for Iter in range(MaxIter):
        #Alpha group
        F=np.zeros((nAlphaGroup,1))
        MeanCost = np.mean(Cost)
        for i in range(nAlphaGroup):
            # Calculate Fitness Values and Selection of Alpha
            F[i] = np.exp(-Cost[i]/MeanCost);   #Convert Cost to Fitness

        P=F/sum(F);
        
        if isSA:
            #########################################################
            # LOCAL SEARCH
            ########################################################
            for i in range(pop_size):
                pop[i], Cost[i] = SA(pop[i], Cost[i], trainX, testX, trainy, testy)
        else:
            #Foraging led by Alpha female
            for m in range(nAlphaGroup):
                i=RouletteWheelSelection(P) #Select Alpha female        
                #Choose k randomly, not equal to Alpha
                #K=np.empty([i-1, nAlphaGroup]) #1:i-1, i+1:nAlphaGroup  ???
                rand=random.randint(1, nAlphaGroup)  #random.randint(1, K.size)
                k=rand #K[rand]
            
                #Define Vocalization Coeff.
                phi=(peep/2) * np.random.uniform(-1,+1,VarSize)
            
                # New Mongoose Position
                newpop_Position=pop[i] + phi * (pop[i] - pop[k])
                
                #Check boundary VarMin,VarMax
                #         for j=1:size(X,2)   
                Flag_UB=newpop_Position > VarMax    #check if they exceed (up) the boundaries
                Flag_LB=newpop_Position < VarMin     #check if they exceed (down) the boundaries
                newpop_Position=(newpop_Position * (~(Flag_UB+Flag_LB))) + (VarMax * Flag_UB) + (VarMin * Flag_LB);
    
                #Evaluation
                newpop_Acc, newpop_Cost= BDMO_fitness(newpop_Position, trainX, testX, trainy, testy)  #CostFunction(X,Y,(newpop.Position > 0.5),HO); ???
                
                #Comparision
                if newpop_Cost <= Cost[i]:
                    pop[i]=newpop_Position
                else:
                    C[i]=C[i]+1
                    
        #Scout group
        for i in range(nScout):
            #Choose k randomly, not equal to i
            #K=np.empty([i-1, nAlphaGroup]) #1:i-1, i+1:nAlphaGroup ???
            rand=random.randint(1, nAlphaGroup)  #random.randint(1, K.size)
            k=rand #K[rand]
        
            #Define Vocalization Coeff.
            phi=(peep/2) * np.random.uniform(-1,+1,VarSize)
        
            #New Mongoose Position
            newpop_Position=pop[i] + phi * (pop[i] - pop[k])
            
            #Check boundary
            Flag_UB=newpop_Position > VarMax     #check if they exceed (up) the boundaries
            Flag_LB=newpop_Position < VarMin     #check if they exceed (down) the boundaries
            newpop_Position=(newpop_Position * (~(Flag_UB+Flag_LB))) + (VarMax * Flag_UB) + (VarMin * Flag_LB)
            
            #Evaluation
            newpop_Acc, newpop_Cost= BDMO_fitness(newpop_Position, trainX, testX, trainy, testy) #CostFunction(X,Y,(newpop.Position > 0.5),HO); ???
            
            #Sleeping mould
            sm.append((newpop_Cost - Cost[i])/max(newpop_Cost, Cost[i]))
            
            #Comparision
            if newpop_Cost <= Cost[i]:
                pop[i]=newpop_Position
            else:
                C[i]=C[i] + 1
        
        #Babysitters
        for i in range(1, nBabysitter):
            newtau=np.mean(sm)
            if C[i] >= L:
                #pop (i).Position=unifrnd(VarMin,VarMax,VarSize);
                #pop (i).Cost=benchmark_functions(pop (i).Position,Function_name,dim);
                M=(pop[i] * sm)/pop[i]
                if newtau < tau:
                   newpop_Position=pop[i] - CF * phi * np.random.rand() * (pop[i] - M)
                else:
                   newpop_Position=pop[i] + CF * phi * np.random.rand() * (pop[i] - M)
                
                tau=newtau
                Flag_UB=newpop_Position > VarMax     #% check if they exceed (up) the boundaries
                Flag_LB=newpop_Position < VarMin     #% check if they exceed (down) the boundaries
                newpop_Position=(newpop_Position * (~(Flag_UB+Flag_LB))) + (VarMax * Flag_UB) + (VarMin * Flag_LB)
                C[i]=0
                
        #Update Best Solution Ever Found
        for i in range(1, nAlphaGroup):
           if Cost[i] <=  best_fit:
               BestSol=pop[i]
               gcost=Cost[i]
        
        #Store Best Cost Ever Found
        curve[Iter]=best_fit
        allfit.append(best_fit)
        allcost.append(gcost)
        runtime=time.time()
        #Display Iteration Information        
        print('Iteration ', str(Iter),  ': Best Cost = ',  str(curve[Iter]))
    
    testAcc = test_accuracy(True, BestSol, trainX, testX, trainy, testy)
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
    
    featCnt = onecnt(BestSol)
    return curve, allcost, testAcc, featCnt, BestSol
