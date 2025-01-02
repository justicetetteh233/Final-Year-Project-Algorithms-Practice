# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 08:52:50 2022

@author: Oyelade
"""
import numpy as np
import math
import random
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier# training a DescisionTreeClassifier
from sklearn.naive_bayes import GaussianNB # training a Naive Bayes classifier
from sklearn.svm import SVC
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

omega =  0.99
EOSA_ID_POS = 0
EOSA_ID_FIT = 1
EOSA_INDIVIDUAL = 1
ID_POS = 0  # Index of position/location of solution/agent
ID_TAR = 1  # Index of target list, (includes fitness value and objectives list)
ID_FIT = 0  # Index of target (the final fitness) in fitness
ID_OBJ = 1  # Index of objective list in target
EPSILON = 10E-10

def initialise(partCount, dim, trainX, testX, trainy, testy):
    population=np.zeros((partCount,dim))
    minn = 1
    maxx = math.floor(0.5*dim)
    if maxx<minn:
        maxx = minn + 1
    for i in range(partCount):
        random.seed(i**3 + 10 + time.time() )
        no = random.randint(minn,maxx)
        if no == 0:
            no = 1
        random.seed(time.time()+ 100)
        pos = random.sample(range(0,dim-1),no)
        for j in pos:
            population[i][j]=1
    return population

def eosa_initialise(partCount, dim, trainX, testX, trainy, testy, algorithm):
    susc_pop=[]
    population=np.zeros((partCount,dim))
    minn = 1
    maxx = math.floor(0.5*dim)
    if maxx<minn:
        maxx = minn + 1
    n=0
    prev_solution=None
    for i in range(partCount):
        random.seed(i**3 + 10 + time.time() )
        no = random.randint(minn,maxx)
        if no == 0:
            no = 1
        random.seed(time.time()+ 100)
        pos = random.sample(range(0,dim-1),no)
        for j in pos:
            population[i][j]=1
        
        if algorithm=='bieosa':
            if n==0:
                prev_solution=population[i]
                g=3
            else:
                partB= 1 - prev_solution
                partA= g * np.array(prev_solution)
                population[i]=partA * partB    # using eq. (1.3)
                prev_solution=population[i]           
        n=n+1
        susc_pop.append((i, [population[i], 0]))
    return susc_pop#,population

def onecnt(agent):
    return sum(agent)

def bestAgent(fit):
    ind = np.argsort(fit, axis=0)
    return ind[0]

def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))

def test_accuracy(hasAgent, agent, train_data, test_data, trainy, testy):
    if hasAgent:
        cols=np.flatnonzero(agent)
        val=1
        if np.shape(cols)[0]==0:
            return val
        
        train_data=train_data[:,cols]
        test_data=test_data[:,cols]
    
    kfold=5
    knnclf=KNeighborsClassifier(n_neighbors=kfold)
    knnclf.fit(train_data,trainy)
    knnval=knnclf.score(test_data,testy)
    knnpred = knnclf.predict(test_data)         
    #knncr=classification_report(testy, knnpred)
    #knncm = confusion_matrix(testy, knnpred)# creating a confusion matrix
    knnrecall = cross_val_score(knnclf, test_data,testy, cv=kfold, scoring='recall')
    knnprecision = cross_val_score(knnclf, test_data, testy, cv=kfold, scoring='precision')
    knnf1 = cross_val_score(knnclf, test_data, testy, cv=kfold, scoring='f1')
    knnauc = cross_val_score(knnclf, test_data, testy, cv=kfold, scoring='roc_auc')
    knnresult=[knnval, knnprecision, knnrecall, knnf1, knnauc]
    #print(str(knnval)+'  '+str(knnpred)+'  '+str(knnprecision)+'  '+str(knnrecall)+'  '+str(knnf1))
    
    rforestclf = RandomForestClassifier(n_estimators=300)
    rforestclf.fit(train_data,trainy)
    rforestval=rforestclf.score(test_data,testy)
    rforestpred = rforestclf.predict(test_data)         
    #rforestcr=classification_report(testy, rforestpred)
    #rforestcm = confusion_matrix(testy, rforestpred)# creating a confusion matrix
    rforestrecall = cross_val_score(rforestclf, test_data,testy, cv=kfold, scoring='recall')
    rforestprecision = cross_val_score(rforestclf, test_data, testy, cv=kfold, scoring='precision')
    rforestf1 = cross_val_score(rforestclf, test_data, testy, cv=kfold, scoring='f1')
    rforestauc = cross_val_score(rforestclf, test_data, testy, cv=kfold, scoring='roc_auc')
    rforestresult=[rforestval, rforestprecision, rforestrecall, rforestf1, rforestauc]
    #print(str(rforestval)+'  '+str(rforestpred)+'  '+str(rforestprecision)+'  '+str(rforestrecall)+'  '+str(rforestf1))
    
    mlpclf=MLPClassifier(alpha=0.001, hidden_layer_sizes=(1000,500,100),max_iter=2000,random_state=4)
    mlpclf.fit(train_data,trainy)
    mlpval=mlpclf.score(test_data,testy)
    mlppred = mlpclf.predict(test_data)         
    #mlpcr=classification_report(testy, mlppred)
    #mlpcm = confusion_matrix(testy, mlppred)# creating a confusion matrix
    mlprecall = cross_val_score(mlpclf, test_data,testy, cv=kfold, scoring='recall')
    mlpprecision = cross_val_score(mlpclf, test_data, testy, cv=kfold, scoring='precision')
    mlpf1 = cross_val_score(mlpclf, test_data, testy, cv=kfold, scoring='f1')
    mlpauc = cross_val_score(mlpclf, test_data, testy, cv=kfold, scoring='roc_auc')
    mlpresult=[mlpval, mlpprecision, mlprecall, mlpf1, mlpauc]
    #print(str(mlpval)+'  '+str(mlppred)+'  '+str(mlpprecision)+'  '+str(mlprecall)+'  '+str(mlpf1))
    
    dtreeclf = DecisionTreeClassifier(max_depth = 2)
    dtreeclf.fit(train_data,trainy)
    dtreeval = dtreeclf.score(test_data,testy)
    dtreepred = dtreeclf.predict(test_data)         
    #dtreecr=classification_report(testy, dtreepred)
    #dtreecm = confusion_matrix(testy, dtreepred)# creating a confusion matrix
    dtreerecall = cross_val_score(dtreeclf, test_data,testy, cv=kfold, scoring='recall')
    dtreeprecision = cross_val_score(dtreeclf, test_data, testy, cv=kfold, scoring='precision')
    dtreef1 = cross_val_score(dtreeclf, test_data, testy, cv=kfold, scoring='f1')
    dtreeauc = cross_val_score(dtreeclf, test_data, testy, cv=kfold, scoring='roc_auc')
    dtreeresult=[dtreeval, dtreeprecision, dtreerecall, dtreef1, dtreeauc]
    #print(str(dtreeval)+'  '+str(dtreepred)+'  '+str(dtreeprecision)+'  '+str(dtreerecall)+'  '+str(dtreef1))
    
    gnbclf = GaussianNB()
    '''
    gnbclf.fit(train_data,trainy)
    gnbval = gnbclf.score(test_data,testy)# accuracy on X_test
    gnbpred = gnbclf.predict(test_data)         
    #gnbcr=classification_report(testy, gnbpred)
    #gnbcm = confusion_matrix(testy, gnbpred)# creating a confusion matrix
    gnbrecall = cross_val_score(gnbclf, test_data,testy, cv=kfold, scoring='recall')
    gnbprecision = cross_val_score(gnbclf, test_data, testy, cv=kfold, scoring='precision')
    gnbf1 = cross_val_score(gnbclf, test_data, testy, cv=kfold, scoring='f1')
    gnbauc = cross_val_score(gnbclf, test_data, testy, cv=kfold, scoring='roc_auc')
    gnbresult=[gnbval, gnbprecision, gnbrecall, gnbf1, gnbauc]
    '''
    gnbresult=[0.0, 0.0, 0.0, 0.0, 0.0]
    #print(str(gnbval)+'  '+str(gnbpred)+'  '+str(gnbprecision)+'  '+str(gnbrecall)+'  '+str(gnbf1))
    
    svmclf = SVC(kernel = 'linear', C = 1)
    '''
    svmclf.fit(train_data,trainy)
    svmpred = svmclf.predict(test_data)         
    svmval = svmclf.score(test_data, testy)# model accuracy for X_test        
    #svmcr=classification_report(testy, svmpred)
    #svmcm = confusion_matrix(testy, svmpred)# creating a confusion matrix
    svmrecall = cross_val_score(svmclf, test_data, testy, cv=kfold, scoring='recall')
    svmprecision = cross_val_score(svmclf, test_data, testy, cv=kfold, scoring='precision')
    svmf1 = cross_val_score(svmclf, test_data, testy, cv=kfold, scoring='f1')
    svmauc = cross_val_score(svmclf, test_data, testy, cv=kfold, scoring='roc_auc')
    svmresult=[svmval, svmprecision, svmrecall, svmf1, svmauc]
    '''
    svmresult=[0.0, 0.0, 0.0, 0.0, 0.0]
    #print(str(svmval)+'  '+str(svmpred)+'  '+str(svmprecision)+'  '+str(svmrecall)+'  '+str(svmf1))
    
    return {'knn': knnresult, 'rf':rforestresult, 'mlp':mlpresult, 'dt':dtreeresult, 'svm':svmresult, 'gnb':gnbresult}

def get_global_best_solution(pop):
    id_individual=1
    sorted_pop = sorted(pop, key=lambda agent: agent[id_individual][EOSA_ID_FIT])  
    best=sorted_pop[0]
    return sorted_pop, best

def get_global_best_solution_norm_pop(pop):
    sorted_pop = sorted(pop)#, key=lambda agent: agent[id_individual][EOSA_ID_FIT])  
    best=sorted_pop[0]
    return sorted_pop, best

def EOSA_fitness(agent, trainX, testX, trainy, testy):
    cols=np.flatnonzero(agent)
    val=1
    if np.shape(cols)[0]==0:
        return val, 1-val
    clf=KNeighborsClassifier(n_neighbors=5)
    train_data=trainX[:,cols]
    test_data=testX[:,cols]
    clf.fit(train_data,trainy)
    val=1-clf.score(test_data,testy)
    
    set_cnt=sum(agent)
    set_cnt=set_cnt/np.shape(agent)[0]
    val=omega*val+(1-omega)*set_cnt
    return val, 1-val

def EOSA_allfit(pop, trainX, testX, trainy, testy):
    acc = []
    cost = []
    pops=[]
    for p in range(len(pop)):
        #print(pop[p])
        idx, ind=pop[p]
        indX=ind[EOSA_ID_POS]
        ac, ct = EOSA_fitness(indX, trainX, testX, trainy, testy)
        acc.append(ac)
        cost.append(ct)
        ind[EOSA_ID_FIT]=ac #the actual fitness values
        pop[p]=idx, ind
        pops.append([pop[p], [ac, ct]])
    return sort_using_fitness(pops)

def sort_using_fitness(pops):
    ID_POP=0
    ID_ACC_COST=1
    ID_ACC=0
    ID_COST=1
    acc, cost, pop=[], [], []
    sorted_pop = sorted(pops, key=lambda agent: agent[ID_ACC_COST][ID_ACC])  
    for p in range(len(sorted_pop)):
        '''
        print('///////////////////////////////////////////////////')
        print(sorted_pop[p][ID_POP])
        print(sorted_pop[p][ID_ACC_COST])
        print(sorted_pop[p][ID_ACC_COST][ID_ACC])
        print(sorted_pop[p][ID_ACC_COST][ID_COST])
        '''
        pop.append(sorted_pop[p][ID_POP])
        acc.append(sorted_pop[p][ID_ACC_COST][ID_ACC])
        cost.append(sorted_pop[p][ID_ACC_COST][ID_COST])
    return acc, cost, pop

def BDMO_fitness(agent, trainX, testX, trainy, testy):
    cols=np.flatnonzero(agent)
    val=1
    if np.shape(cols)[0]==0:
        return val, 1-val
    clf=KNeighborsClassifier(n_neighbors=5)
    train_data=trainX[:,cols]
    test_data=testX[:,cols]
    clf.fit(train_data,trainy)
    val=1-clf.score(test_data,testy)
    set_cnt=sum(agent)
    set_cnt=set_cnt/np.shape(agent)[0]
    val=omega*val+(1-omega)*set_cnt
    return val, 1-val

def BDMO_allfit(pop, trainX, testX, trainy, testy):
    acc = np.zeros((len(pop), 1))
    cost = np.zeros((len(pop), 1))
    for p in range(len(pop)):
        acc[p], cost[p] = BDMO_fitness(pop[p], trainX, testX, trainy, testy)
    return acc, cost

def fitness(agent, trainX, testX, trainy, testy):
    cols=np.flatnonzero(agent)
    val=1
    if np.shape(cols)[0]==0:
        return val, 1-val
    
    clf=KNeighborsClassifier(n_neighbors=5)
    train_data=trainX[:,cols]
    test_data=testX[:,cols]
    clf.fit(train_data,trainy)
    val=1-clf.score(test_data,testy)
    set_cnt=sum(agent)
    set_cnt=set_cnt/np.shape(agent)[0]
    val=omega*val+(1-omega)*set_cnt
    return val, 1-val

def allfit(pop, trainX, testX, trainy, testy):
    fit = np.zeros((len(pop), 1))
    cost = np.zeros((len(pop), 1))
    for p in range(len(pop)):
        fit[p], cost[p] = fitness(pop[p], trainX, testX, trainy, testy)
    return fit, cost

def get_global_best(pop, id_fitness, id_best):
    minn = 100
    temp = pop[0]
    for i in pop:
        #print(i)
        #print(i[1])
        if isinstance(i[1], tuple):
            fit, cost=i[1]
        else:
            fit=i[1]
        minn = min(minn, fit)
        temp = i
    return temp

def randomwalk(agent):
    percent = 30
    percent /= 100
    neighbor = agent[EOSA_INDIVIDUAL][EOSA_ID_POS].copy()
    size = np.shape(agent[EOSA_INDIVIDUAL][EOSA_ID_POS])[0]
    upper = int(percent*size)
    if upper <= 1:
        upper = size
    x = random.randint(1,upper)
    realsize=size - 1 #if (x > size -1) else x - 2
    alist=range(0,realsize)
    pos = random.sample(alist, min(x, len(alist)-1))
    for i in pos:
        neighbor[i] = 1 - neighbor[i]
    agent[EOSA_INDIVIDUAL][EOSA_ID_POS]=neighbor
    return agent

def compare_agent(agent_a, agent_b, minmax):
    if minmax == "min":
        if agent_a < agent_b:
            return True
        return False
    else:
        if agent_a < agent_b:
            return False
        return True

def RouletteWheelSelection(P):
    C=sum(P)
    r = np.random.uniform(low=0, high=C)
    for idx, f in enumerate(P):
            r = r + f
            if r > C:
                return idx
    return np.random.choice(range(0, len(P)))

def vtransform(x, dim):
    for k in range(dim):
        rand=np.random.rand()
        determinant=random.randint(0,1)
        tfunc=t2(x[k]) if determinant ==1 else t1(x[k])
        if (tfunc > rand):
            x[k] = 1
        else:
            x[k] = 0
    return x
 
def stransform(x, dim):
    for k in range(dim):
        rand=np.random.rand()
        determinant=random.randint(0,1)
        tfunc=s2(x[k]) if determinant ==1 else s1(x[k])
        if (tfunc > rand):
            x[k] = 1
        else:
            x[k] = 0
    return x

def Vfunction1(gamma):
    return abs(np.tanh(gamma))

def Vfunction2(gamma):
    val = (math.pi)**(0.5)
    val /= 2
    val *= gamma
    val = math.erf(val)
    return abs(val)

def Vfunction3(gamma):
    val = 1 + gamma*gamma
    val = math.sqrt(val)
    val = gamma/val
    return abs(val)

def Vfunction4(gamma):
    val=(math.pi/2)*gamma
    val=np.arctan(val)
    val=(2/math.pi)*val
    return abs(val)

def t1(x): 
    return abs((x)/math.sqrt(2 + x * x))

def t2(x):
    return abs(np.tanh(x))

def s1(x):
    return 1/(1 + np.exp(-x/2))

def s2(x):
    return 1 - 1/(1 + np.exp(x))

def sigmoid1_old(gamma):
    #print(gamma)
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))

def sigmoid1(gamma):     #convert to probability
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))

def sigmoid1c(gamma):     #convert to probability
    gamma = -gamma
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))

def sigmoid2(gamma):
    gamma /= 2
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))
        
def sigmoid3(gamma):
    gamma /= 3
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))

def sigmoid4(gamma):
    gamma *= 2
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))


 