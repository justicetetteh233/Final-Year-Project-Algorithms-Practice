# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:14:04 2022

@author: Oyelade
"""
import numpy as np
import random
from copy import deepcopy
import math,time,sys
from binaryOptimizer.model.root import *

## Simulated Anealing
def SA(agents, fitAgent, trainX, testX, trainy, testy, dim, isNormalPop):
    # initialise temprature
    T = 4*len(agents[0][EOSA_INDIVIDUAL][EOSA_ID_POS])
    T0 = 2*len(agents[0][EOSA_INDIVIDUAL][EOSA_ID_POS])
    S = agents[0] #[EOSA_INDIVIDUAL][EOSA_ID_POS].copy();
    bestSolution = agents[0]  #.copy();
    bestFitness = agents[0][EOSA_INDIVIDUAL][EOSA_ID_FIT];
        
    for agent in agents:
        while T > T0:
            neighbor = randomwalk(S)
            neighborFitness, cost = EOSA_fitness(neighbor[EOSA_INDIVIDUAL][EOSA_ID_POS], trainX, testX, trainy, testy)
            if neighborFitness < bestFitness:
                S = neighbor #.copy()
                bestSolution = neighbor  #.copy()
                bestFitness = neighborFitness
            elif neighborFitness == bestFitness:
                if np.sum(neighbor[EOSA_INDIVIDUAL][EOSA_ID_POS]) == np.sum(bestSolution[EOSA_INDIVIDUAL][EOSA_ID_POS]):
                    S = neighbor #.copy()
                    bestSolution = neighbor #.copy()
                    bestFitness = neighborFitness
            else:
                theta = neighborFitness - bestFitness
                if np.random.rand() < math.exp(-1*(theta/T)):
                    S = neighbor #.copy()
            T *= 0.925
            agent[EOSA_INDIVIDUAL][EOSA_ID_FIT]=bestFitness
            agent[EOSA_INDIVIDUAL][EOSA_ID_POS]=bestSolution[EOSA_INDIVIDUAL][EOSA_ID_POS]
    return agents #bestSolution, bestFitness

def FFA(agents, g_best, trainX, testX, trainy, testy, dim, isNormalPop):
    pop_size=len(agents)
    gamma = 0.001
    beta_base = 2
    alpha = 0.2
    alpha_damp = 0.99
    delta = 0.05
    exponent = 2    
    dmax = np.sqrt(dim) # Maximum Distance
    minmax = "min"
    for idx in range(0, pop_size):
        agent = deepcopy(agents[idx])
        pop_child = []
        for j in range(idx + 1, pop_size):
            # Move Towards Better Solutions
            if isNormalPop:
                fitA, costA=fitness(agents[j], trainX, testX, trainy, testy)
                fitB, costB=fitness(agent, trainX, testX, trainy, testy)
            else:
                fitA=agents[j][EOSA_INDIVIDUAL][EOSA_ID_FIT]
                fitB=agents[j][EOSA_INDIVIDUAL][EOSA_ID_FIT]
            agentA_fit, agentB_fit=fitA, fitB,
            if compare_agent(agentA_fit, agentB_fit, minmax):
                # Calculate Radius and Attraction Level
                rij = np.linalg.norm(agent - (agents[j] if isNormalPop  else agents[j][EOSA_INDIVIDUAL][EOSA_ID_POS])) / dmax
                beta = beta_base * np.exp(-gamma * rij ** exponent)
                # Mutation Vector
                mutation_vector = delta * np.random.uniform(0, 1, dim)
                temp = np.matmul(((agents[j] if isNormalPop  else agents[j][EOSA_INDIVIDUAL][EOSA_ID_POS]) - agent), np.random.uniform(0, 1, (dim, dim)))
                pos_new = agent + dyn_alpha * mutation_vector + beta * temp
                pop_child.append(pos_new)
        if len(pop_child) < 2:
            continue
        _, local_best = get_global_best_solution_norm_pop(pop_child) if isNormalPop  else get_global_best_solution(pop_child)
        # Compare to Previous Solution
        if compare_agent(local_best, agent, minmax):
            if isNormalPop:
                agents[idx]= local_best
            else:
                agents[idx][EOSA_INDIVIDUAL][EOSA_ID_POS] = local_best
    agents.append(g_best)
    dyn_alpha = alpha_damp * alpha
    return agents

    
def DMO(pop, fitAgent, trainX, testX, trainy, testy, dim, isNormalPop, Iter, MaxIter):
    pop_size=len(pop)
    if pop_size <= 5:
        return pop
    #Variable initialaization
    nVar=dim             #Number of Decision Variables
    VarSize=[]           #Decision Variables Matrix Size  1:nVar
    VarMin=0             #Decision Variables Lower Bound
    VarMax=1             #Decision Variables Upper Bound
    nBabysitter= 3         #Number of babysitters
    nAlphaGroup=pop_size - nBabysitter         #Number of Alpha group
    nScout=nAlphaGroup         #Number of Scouts
    L=round(0.6*nVar*nBabysitter)  #Babysitter Exchange Parameter 
    peep=1             #Alpha female \.12s vocalization 
    tau=random.uniform(0, 1)
    sm=[]
    TestaccG=None   
    
    Cost=[]
    if isNormalPop:
        _, Cost=BDMO_allfit(pop, trainX, testX, trainy, testy)
    else:
        for p in pop:
            Cost.append(p[EOSA_INDIVIDUAL][EOSA_ID_FIT])
        
    #Abandonment Counter
    C=np.zeros((nAlphaGroup,1))
    Iter=1
    CF=(1-Iter/MaxIter) ** (2*Iter/MaxIter)  #np.linalg.matrix_power((1-Iter/MaxIter), (2*Iter/MaxIter))

    #Alpha group
    F=np.zeros((nAlphaGroup,1))
    MeanCost = np.mean(Cost)
    for i in range(nAlphaGroup):
        # Calculate Fitness Values and Selection of Alpha
        F[i] = np.exp(-Cost[i]/MeanCost);   #Convert Cost to Fitness

    P=F/sum(F);
    
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
        newpop_Position=pop[i] + phi * (pop[i] - pop[k]) if isNormalPop else pop[i][EOSA_INDIVIDUAL][EOSA_ID_POS] + phi * (pop[i][EOSA_INDIVIDUAL][EOSA_ID_POS] - pop[k][EOSA_INDIVIDUAL][EOSA_ID_POS])
            
        #Check boundary VarMin,VarMax
        #         for j=1:size(X,2)   
        Flag_UB=newpop_Position > VarMax    #check if they exceed (up) the boundaries
        Flag_LB=newpop_Position < VarMin     #check if they exceed (down) the boundaries
        newpop_Position=(newpop_Position * (~(Flag_UB+Flag_LB))) + (VarMax * Flag_UB) + (VarMin * Flag_LB);

        #Evaluation
        newpop_Acc, newpop_Cost= BDMO_fitness(newpop_Position, trainX, testX, trainy, testy)  #CostFunction(X,Y,(newpop.Position > 0.5),HO); ???
            
        #Comparision
        if newpop_Cost <= Cost[i]:
            if isNormalPop:
                pop[i]=newpop_Position
            else:
                pop[i][EOSA_INDIVIDUAL][EOSA_ID_POS]=newpop_Position
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
        newpop_Position=pop[i] + phi * (pop[i] - pop[k]) if isNormalPop else pop[i][EOSA_INDIVIDUAL][EOSA_ID_POS] + phi * (pop[i][EOSA_INDIVIDUAL][EOSA_ID_POS] - pop[k][EOSA_INDIVIDUAL][EOSA_ID_POS])
        
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
            if isNormalPop:
                pop[i]=newpop_Position
            else:
                pop[i][EOSA_INDIVIDUAL][EOSA_ID_POS]=newpop_Position
        else:
            C[i]=C[i] + 1
    
    #Babysitters
    for i in range(1, nBabysitter):
        newtau=np.mean(sm)
        if C[i] >= L:
            #pop (i).Position=unifrnd(VarMin,VarMax,VarSize);
            #pop (i).Cost=benchmark_functions(pop (i).Position,Function_name,dim);
            M=(pop[i] * sm)/pop[i] if isNormalPop else (pop[i][EOSA_INDIVIDUAL][EOSA_ID_POS] * sm)/pop[i][EOSA_INDIVIDUAL][EOSA_ID_POS]
            if newtau < tau:
                if isNormalPop:
                    newpop_Position=pop[i] - CF * phi * np.random.rand() * (pop[i] - M)
                else:
                    newpop_Position=pop[i][EOSA_INDIVIDUAL][EOSA_ID_POS] - CF * phi * np.random.rand() * (pop[i][EOSA_INDIVIDUAL][EOSA_ID_POS] - M)
            else:
               if isNormalPop:
                   newpop_Position=pop[i] + CF * phi * np.random.rand() * (pop[i] - M)
               else:
                   newpop_Position=pop[i][EOSA_INDIVIDUAL][EOSA_ID_POS] + CF * phi * np.random.rand() * (pop[i][EOSA_INDIVIDUAL][EOSA_ID_POS] - M)
            tau=newtau
            Flag_UB=newpop_Position > VarMax     #% check if they exceed (up) the boundaries
            Flag_LB=newpop_Position < VarMin     #% check if they exceed (down) the boundaries
            newpop_Position=(newpop_Position * (~(Flag_UB+Flag_LB))) + (VarMax * Flag_UB) + (VarMin * Flag_LB)
            C[i]=0
    return pop

def PSO(agents, fitAgent, trainX, testX, trainy, testy, dim, isNormalPop, curIter, MaxIter):
    popSize=len(agents)
    velocity = np.zeros((popSize,dim))
    gbestVal = 1000
    C1 = 2
    C2 = 2
    WMAX = 0.9
    WMIN = 0.4
    gbestVec = np.zeros(popSize) #np.shape(agents[0]))
    pbestVal = np.zeros(popSize)
    pbestVec = np.zeros((popSize, dim))    
    for i in range(popSize):
        pbestVal[i] = 1000
    
    popnew = [] #np.zeros((popSize,dim))
    if isNormalPop:
        fitList = allfit(agents,trainX,trainy,testX,testy)
    else:
        fitList=[]
        for agent in agents:
            fitList.append(agent[EOSA_INDIVIDUAL][EOSA_ID_FIT])
    
    #update pbest
    for i in range(popSize):
        if (fitList[i] < pbestVal[i]):
            pbestVal[i] = fitList[i]
            #print(np.shape(agents[0][EOSA_INDIVIDUAL][EOSA_ID_POS]))
            #print(agents[i][EOSA_INDIVIDUAL][EOSA_ID_POS])
            pbestVec[i] = np.array(agents[i][EOSA_INDIVIDUAL][EOSA_ID_POS]) #.copy()
            #agents[i].copy() if isNormalPop else 
    #update gbest
    for i in range(popSize):
        if (fitList[i] < gbestVal):
            gbestVal = fitList[i]
            gbestVec = agents[i].copy() if isNormalPop else (agents[i][EOSA_INDIVIDUAL][EOSA_ID_POS]).copy()
    
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
        
        if isNormalPop:
            x = np.subtract(pbestVec[inx] , agents[inx])
        else:
            x = np.subtract(pbestVec[inx] , agents[inx][EOSA_INDIVIDUAL][EOSA_ID_POS])
        
        if isNormalPop:
            y = np.subtract(gbestVec , agents[inx])
        else:
            y = np.subtract(gbestVec , agents[inx][EOSA_INDIVIDUAL][EOSA_ID_POS])
        velocity[inx] = np.multiply(W,velocity[inx]) + np.multiply(r1,x) + np.multiply(r2,y)

        if isNormalPop:
            popnew[inx] = np.add(agents[inx],velocity[inx])
        else:
            position=np.add(agents[inx][EOSA_INDIVIDUAL][EOSA_ID_POS],velocity[inx])    
            index, _=agents[inx]
            fit, cost=EOSA_fitness(position, trainX, testX, trainy, testy)
            individual=index, [position, fit]
            popnew.append(individual)
    agents = popnew.copy()
    return agents