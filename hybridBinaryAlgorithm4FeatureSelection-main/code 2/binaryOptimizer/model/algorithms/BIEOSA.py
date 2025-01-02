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
from utils.differential_equations import DiffEquation
from utils.paper_equations import *
from scipy.stats import expon
from model.root import *
from utils.convergencePlot import save_results_to_csv

"""
   Standard version of Ebola Virus Optimization Algorithm (belongs to Biology-based Algorithms)
   In this algorithms: 
"""
omega =  0.99
EOSA_ID_POS = 0
EOSA_ID_FIT = 1
ID_INDIVIDUAL=1
ID_INDIVIDUAL_INDEX=0
NEIGHBOURHOOD_THRESHHOLD=0.5
MIN_MAX_INFECTED_SOL=1
SEARCHABLE_LIMIT=0.1
π=0.1 #Recruitment rate of susceptible human individuals
ŋ=np.random.rand() #Decay rate of Ebola virus in the environment
α=np.random.rand() #Rate of hospitalization of infected individuals
dis=random.uniform(0.4, 0.9)#Disease-induced death rate of human individuals
β_1=0.1#Contact rate of infectious human individuals
β_2=0.1#Contact rate of pathogen individuals/environment
β_3=0.1#Contact rate of deceased human individuals
β_4=0.1#Contact rate of recovered human individuals
rr=np.random.rand() #Recovery rate of human individuals
dr=np.random.rand() #Natural death rate of human individuals
br=np.random.rand() #Rate of  burial of deceased human individuals
vr=np.random.rand() #Rate of vaccination of individuals
hr=np.random.rand() #Rate of response to hospital treatment
vrr=np.random.rand() #Rate response to vaccination
qrr=np.random.rand()	#Rate of quarantine of infected individuals

def create_solution(dim):
    minn = 1
    maxx = math.floor(0.5*dim)
    if maxx<minn:
        maxx = minn + 1
    random.seed(0**3 + 10 + time.time() )
    no = random.randint(minn,maxx)
    if no == 0:
        no = 1
    random.seed(time.time()+ 100)
    pos = random.sample(range(0,dim-1),no)
    solution=np.zeros((1,dim))
    for j in pos:
        solution[0][j]=1
        
    return solution[0]

def allocate_immuned_susceptible_population(S, g_best, trainX, testX, trainy, testy):
    startsf=1 #becuase individual at position 0 is already an index case for infected population
    #Using eq. 2.1, derive a
    endsf=math.floor(len(S)//3)
    SF=[deepcopy(S[i]) for i in range(startsf, endsf)]
    #since the population is sorted, we pick the last individual as the worst
    g_worst=S[-1][ID_INDIVIDUAL][EOSA_ID_POS] 
    #Enhance the immunology of all individuals in SF
    for i, indv in enumerate(SF):
        idx, individual_fit=indv
        individual=individual_fit[EOSA_ID_POS]
        b1=(deepcopy(g_best[EOSA_ID_POS])  - g_worst)/2
        #compute immunity vector
        im=b1 * (g_best[EOSA_ID_POS]  - individual) + individual 
        individual=im
        indxfit, cost=EOSA_fitness(individual, trainX, testX, trainy, testy)
        SF[i]=idx, [individual, indxfit]        
        #Each member in SF should replace corresponding one in S
        for i in range(startsf, endsf):
            #recall that we allocate Sf from S[1-endsf], S[1]=SF[0], S[2]=SF[1] ... S[ensf]=SF[endsf-1]
            S[i]=SF[i-1]        
        #Using eq. 2.2
        partOfS=len(S)//8 #Get only a quarter of the population in the system 
        endsc=math.floor(endsf//partOfS)
        #We will simply keep elements in this sub-population and prevent them from being infected. 
        SC=[deepcopy(S[i]) for i in range(endsf, (endsf+endsc))]
    return SC, SF
 
def allocate_immuned_infected(pop_infected, I, IF, trainX, testX, trainy, testy):
    startif=0 
    #Using eq. 2.2, derive a
    endif=math.floor(len(pop_infected)//4)
    #We want to compute the global best for all infected individuals
    #We can only proceed if the number of members in I and pop_infected is > 0
    if len(I) > 0 and len(pop_infected) > 0:
        new_I_pop, infected_gbest = get_global_best_solution(I)
        #Compute the global worst among all I
        #since the population is sorted, we pick the last individual as the worst
        infected_gworst=new_I_pop[-1][ID_INDIVIDUAL][EOSA_ID_POS] 
        #Generate members of IF from those newly infected members
        IF=[pop_infected[i] for i in range(startif, endif)]
        #mutate all members in IF based on eq. 3.2
        for i, individual in enumerate(IF):
            idx, individual_fit=individual
            individual=individual_fit[EOSA_ID_POS]
            b2=(infected_gbest[EOSA_ID_POS]  - infected_gworst)/2
            #compute immunity vector using eq. 3.2
            im=b2 * (infected_gbest[EOSA_ID_POS]  - individual) + individual 
            fit, cost= EOSA_fitness(im, trainX, testX, trainy, testy)
            IF[i]=idx, [im, fit]            
    return IF
    
def bieosa(datasetname, pop_size, MaxIter, runfilename, metrics_result_dir):
    modelrates = {
        "recruitment_rate": π,
        "decay_rate": ŋ,
        "hospitalization_rate": α,
        "disease_induced_death_rate": dis,
        "contact_rate_infectious": β_1,
        "contact_rate_pathogen": β_2,
        "contact_rate_deceased": β_3,
        "contact_rate_recovered": β_4,
        "recovery_rate": rr,
        "natural_death_rate": dr,
        "burial_rate": br,
        "vacination_rate": vr,
        "hospital_treatment_rate": hr,
        "vaccination_response_rate": vrr,
        "quarantine_rate": qrr
    }
    df = pd.read_csv(datasetname)
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
    curve, cost, testAcc, featCnt, BestSol=evolve(MaxIter, modelrates, pop_size, dimension, trainX, testX, trainy, testy, runfilename, metrics_result_dir)
    return curve, cost, testAcc, featCnt, BestSol

def create_initial_timelines(pop_size):
    timelines =[np.random.rand() for _ in range(pop_size)]
    return timelines     

def init_solve(MaxIter, pop_size, dimension, trainX, testX, trainy, testy, E, I, incubation_period):
    # initialize the  container storing positions of infected 
    #including the position of the selected indexcase
    pos = create_initial_timelines(pop_size)
    # non-human host of infected pathogens in environment PE
    PE=eosa_initialise(2, dimension, trainX, testX, trainy, testy, 'bieosa')  
    #assign population to S
    pop=eosa_initialise(pop_size, dimension, trainX, testX, trainy, testy, 'bieosa')
    acc, cost, S=EOSA_allfit(pop, trainX, testX, trainy, testy)
    gbest=S[0]
    
    test_accuracy(True, gbest[ID_INDIVIDUAL][EOSA_ID_POS], trainX, testX, trainy, testy)
        
    #initialize the exposed and infected cases 
    E.append(gbest) #add individual from the suceptible to exposed group
    I.append(gbest) # add the first individual on the Susceptible list to the Infected list  
    #Since the first susceptible individual is now infected, we mark it out of S
    _, ind=gbest
    S[0]= _symbol_of_infected_individual__(), ind  
    
    gcost=cost[0]
    # make that first infected case as the current and global best
    current_best=gbest# make the fitness value of indexcase the current & global best
    #generate incubation time for an individual in population
    incub=[]
    incub.append(generate_incubation_period_for_an_individual(incubation_period)) 
    #we chose the exponential distribution because it models movement of infected persons
    #which rises and falls due to isolation resulting from detection.
    prob=(expon.rvs(scale=1,loc=0,size=pop_size))
    
    #apply the immunity-based population accoring to equations: 2.1 and 2.2
    SC, SF=allocate_immuned_susceptible_population(S, gbest, trainX, testX, trainy, testy)
    return pos, S, PE, E, I, incub, prob.tolist(), gbest, gcost, SC, SF

def evolve(MaxIter, model_rates, pop_size, dimension, trainX, testX, trainy, testy, runfilename, metrics_result_dir):     
    incubation_period=5
    epxilon=0.0001
    SEARCHABLE_LIMIT=(pop_size*10)/100 # to swtich between random and linear infection in a population
    S, E, I, H, R, V, D, Q, PE=[], [], [], [], [], [], [], [], []
    SF, SC, IF=[], [], []
    
    pos, S, PE, E, I, incub, prob, gbest, gcost, SC, SF=init_solve(MaxIter, pop_size, dimension, trainX, testX, trainy, testy, E, I, incubation_period)
    
    #Array to Hold Best Cost Values
    allfit=[]
    allcost=[]
    for epoch in range(MaxIter):
        #Initialization of parameters for differentiation operations
        diff_params = {
            "epoch": epoch+1, "S": S, "I": I, "H": H, "V": V, "R": R, "D": D, "PE": PE, "Q": Q
        }
        dif=DiffEquation(diffparams=diff_params, model_rates=model_rates)
        
        #initialize containers for storing array-value
        newI=[]
        inc_newI=[]
        prob_newI=[]
        pos_newI=[]
                
        #select a random number of infected individuals and isolate them
        if len(I) > 1:
            qrate=(equation12(dif, len(Q)))
            qrate=(np.abs(qrate))[epoch]
            qsize=math.ceil((qrate * len(I))+epxilon)  #
            qsize=random.randint(1, qsize)
            Q=_quarantine_infectecd_individuals__(qsize, I, Q)            
                
        #limit the number of infected who can propagate the disease by eliminating quarantined indi from I
        sizeInfectable=len(I) - len(Q) #remove some for quarantine
        actualSizeInfectable=math.ceil((sizeInfectable*25)/100) #only a percentage can actually infect
        #print('A '+str(actualSizeInfectable)+' S='+str(sizeInfectable)+' I='+str(len(self.I))+'   Q='+str(len(self.Q)))
        #print('infectable='+str(sizeInfectable)+'  I='+str(len(self.I))+'  Q='+str(len(self.Q)))
        for j in range(actualSizeInfectable): #len(self.I)
            pos[j], drate=equation1(pos[j], max(pos)) # computes the new postion of infected individual at [j]
            d=incub[j] 
                    
            if (d >= incubation_period):             
                #print('j='+str(j)+'   prob='+str(prob)+'  prob[j]='+str(prob[j])+'  epoch='+str(epoch))
                neighbourhood = prob[j] #probability that self.pos[j] exceeds NEIGHBOURHOOD_THRESHHOLD at time (i)
                rate=(equation7(dif, len(I)))
                rate=(np.abs(rate))[epoch] 
                        
                newS=_size_uninfected_susceptible__(S)
                fracNewS=math.ceil((newS*0.5)/100) #two percent of newS
                t_type=1
                if neighbourhood < NEIGHBOURHOOD_THRESHHOLD:
                    size=math.ceil((0.1 * rate)+epxilon+ (fracNewS))     # add a fractiion of newS
                    indvd_change_factor=0.1 * rate
                    t_type=0
                else :
                    size=math.ceil((0.7 * rate)+epxilon+ (fracNewS))              # add a fractiion of newS
                    indvd_change_factor=0.7 * rate
                        
                s=newS
                proposed_of_infected=random.randint(1, size)
                                            
                #randomly pick the size_of_infected from Susceptible and make them now infected
                tmp, size_of_infected=_infect_susceptible_population__(proposed_of_infected, newS, indvd_change_factor, gbest, S, trainX, testX, trainy, testy, SEARCHABLE_LIMIT, t_type, dimension, SC)    
                #print('genSize='+str(size)+' availS='+str(s)+' propIn='+str(proposed_of_infected)+' actualn='+str(size_of_infected))                        
                for ni in range(size_of_infected):
                    #generate the incubation time for this newly infected individual
                    inc_newI.append(generate_incubation_period_for_an_individual(incubation_period))
                    #generate the probabilities value of neighbourhood for all epoch for this individual
                    mpb=expon.rvs(scale=1,loc=0,size=1)
                    prob_newI.append(mpb[0])
                    #copy its initial position and store it
                    pos_newI.append(pos[j])
                    #Add the newly infected individual
                    newI.append(tmp[ni]) 
        
        #for this new set of infected cases in newI, we shall compute self.IF to apply eq. 2.3 and 3.2 accordingly
        allocate_immuned_infected(newI, I, IF, trainX, testX, trainy, testy)
        
        I.extend(newI)
        incub.extend(inc_newI)
        prob.extend(prob_newI)
        pos.extend(pos_newI)
        #print('1   prob='+str(prob))
        
        infected_size=_new_infected_change__(newi=I, eqtn=equation8(dif, len(newI)), e=epoch, fl='h')
        #print('H >>newI '+str(len(newI))+' size_of_infected='+str(infected_size))                        
        h =_hospitalize_infected_population__(I, infected_size)    
        H =[H.append(h[i]) for i in range(len(h))]
                
        infected_size=_new_infected_change__(newi=h, eqtn=equation10(dif, len(h)), e=epoch, fl='v')
        #print('V >>h '+str(len(h))+' size_of_infected='+str(infected_size))                        
        v =_vaccinate_hospitalized_population__(h, infected_size)    
        V =[V.append(v[i]) for i in range(len(v))]
                
        infected_size=_new_infected_change__(newi=I, eqtn=equation9(dif, len(newI)), e=epoch, fl='r')
        #print('R >>newI '+str(len(newI))+' size_of_infected='+str(infected_size))                        
        r =_recover_infected_population__(I, infected_size)    
        R =[R.append(r[i]) for i in range(len(r))]
        if r:
            I, incub, prob, pos=_remove_dead_or_recovered_from_infected__(inf=I, rc=r, inc=incub, pb=prob, pos=pos)
        _addback_recovered_2_susceptible__(deepcopy(r), S)
        #print('2   prob='+str(prob))
        infected_size=_new_infected_change__(newi=I, eqtn=equation11(dif, len(newI)), e=epoch, fl='d')
        #print('D >>newI '+str(len(newI))+' size_of_infected='+str(infected_size))                        
        d =_die_infected_population__(I, infected_size)    
        D =[D.append(d[i]) for i in range(len(d))]
        if d:
            I, incub, prob, pos=_remove_dead_or_recovered_from_infected__(inf=I, rc=d, inc=incub, pb=prob, pos=pos)
        _rebirth_2replace_dead_in_susceptible__(deepcopy(d), S, dimension, trainX, testX, trainy, testy) 
        #print('3   prob='+str(prob))
        #update population: combines len(S + I) = pop_size
        new_pop=deepcopy(S)#+deepcopy(self.I)
            
        #re-calculate fitness after each
        acc, cost, S=EOSA_allfit(new_pop, trainX, testX, trainy, testy)
        
        #update current best and global best based on the mutation of population during this epoch
        current_best = S[0]
        if current_best[ID_INDIVIDUAL][EOSA_ID_FIT] < gbest[ID_INDIVIDUAL][EOSA_ID_FIT]:
            gbest = current_best
            gcost=cost[0]
            
        #empty list of those quarantined in this epoch
        Q=[]
        
        #Store Best solution so far
        #curve[epoch]=gbest[ID_INDIVIDUAL][EOSA_ID_POS]
        allfit.append(gbest[ID_INDIVIDUAL][EOSA_ID_FIT])
        allcost.append(gcost)
        #Display Iteration Information        
        print('Iteration ', str(epoch),  ': Best Fit = ',  str(allfit[epoch]))
        
    testAcc = test_accuracy(True, gbest[ID_INDIVIDUAL][EOSA_ID_POS], trainX, testX, trainy, testy)
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
    
    featCnt = onecnt(gbest[ID_INDIVIDUAL][EOSA_ID_POS])
    return allfit, allcost, testAcc, featCnt, gbest[ID_INDIVIDUAL][EOSA_ID_POS]

def _quarantine_infectecd_individuals__(qsize, I, Q):    
    if qsize > len(I):
        qsize=len(I)-2 
    for i in range(qsize):
        Q.append(I[i])
    return Q
    
def _remove_dead_or_recovered_from_infected__(inf=None, rc=None, inc=None, pb=None, pos=None):        
    tmp_infected=[]
    tmp_incub=[]
    tmp_prob=[]
    tmp_pos=[]
    already_selected=[]
        
    indexs=[]
    for i in range(len(rc)):            
        idx_r, dr_individ=rc[i]
        indexs.append(idx_r)
            
    for i in range(len(rc)):            
        idx_r, dr_individ=rc[i]
        #print('idx_r  <**** '+str(idx_r))
        for j in range(len(inf)):
            idx_i, i_individ=inf[j]  #(i_individ[0] != dr_individ[0]).all()
            if  idx_i != idx_r and idx_i not in already_selected and idx_i not in indexs:
                tmp_infected.append(inf[j])
                tmp_incub.append(inc[j])
                tmp_prob.append(pb[j])
                tmp_pos.append(pos[j])
                #print('idx_i  ==>'+str(idx_i)+'  idx_r  ==> '+str(idx_r))
                already_selected.append(idx_i)
            #else:
                #print(str(already_selected)+' idx_i  <**** '+str(idx_i)+'  idx_r  <**** '+str(idx_r)) 
        
    return tmp_infected, tmp_incub, tmp_prob, tmp_pos
    
def _symbol_of_infected_individual__():
    solution = None #[np.random.uniform(0, 0, self.problem_size)]
    fit=0
    return solution#, fit
    
def _new_infected_change__(newi=None, eqtn=None,  e=None, fl=None):
    equat_value=(eqtn)
    rate=(np.abs(equat_value))[e]
    rate=0 if math.isnan(rate) else rate
    maxi=math.ceil((0.1 * rate * len(newi))) #+self.epxilon
    infected_size=random.randint(0, maxi)
    return infected_size
        
def _size_uninfected_susceptible__(pop=None):
    suscp=[]
    for i in range(len(pop)):
        x, individ=pop[i]
        if x is not None:
            suscp.append(pop[i])
    return len(suscp)
    
def _remove_infected_individuals_from_S__(pop=None):
    suscp=[]
    for i in range(len(pop)):
        x, individ=pop[i]
        if x is not None:#(individ == self._symbol_of_infected_individual__()).all():
            suscp.append(pop[i])
    return suscp 
    
def _addback_recovered_2_susceptible__(recovered=None, S=None):
    #if an individual recovers, change its status from None to original INDEX before infection
    #but don't change the fit/genetic of the individual
    for r in recovered:
        r_indx, r_individual=r
        for s in S:
            s_indx, s_individual=s 
            if  (s_individual[0] == r_individual[0]).all() and s_indx is None:
                #print(str(r_indx)+' index recovered '+str(len(recovered)))
                S[r_indx]=(r_indx, s_individual)        
    
def _rebirth_2replace_dead_in_susceptible__(dead=None, S=None, dimension=None, trainX=None, testX=None, trainy=None, testy=None): 
   #if an individual dies, birth a new individual entirly to replace the dead
   #Then change its status from None to original INDEX before infection and death
    for d in dead:
        d_indx, d_individual=d
        for s in S:
            s_indx, s_individual=s
            if (s_individual[0] == d_individual[0]).all() and s_indx is None:
                new_solution=create_solution(dimension) 
                acc, cost=EOSA_fitness(new_solution, trainX, testX, trainy, testy)
                S[d_indx]=(d_indx, [new_solution, acc])
    
def _die_infected_population__(population=None, size_of_infected=None):
    f = lambda x: random.randint(0, (x))  
    tmp=[]
    pop_size=len(population)-1
    for _ in range(size_of_infected):
        x=f(pop_size)
        tmp.append(deepcopy(population[x]))
    return tmp
    
def _hospitalize_infected_population__(population=None, size_of_infected=None):
    f = lambda x: random.randint(0, (x))  
    tmp=[]
    pop_size=len(population)-1
    for _ in range(size_of_infected):
        x=f(pop_size)
        tmp.append(deepcopy(population[x]))
    return tmp
    
def _vaccinate_hospitalized_population__(population=None, size_of_infected=None):
    f = lambda x: random.randint(0, (x))  
    tmp=[]
    pop_size=len(population)-1
    for _ in range(size_of_infected):
        x=f(pop_size)
        tmp.append(deepcopy(population[x]))
    return tmp
    
def _recover_infected_population__(population=None, size_of_infected=None):
    f = lambda x: random.randint(0, (x))  
    tmp=[]
    pop_size=len(population)-1
    for _ in range(size_of_infected):
        x=f(pop_size)
        tmp.append(deepcopy(population[x]))
    return tmp
    
def _infect_susceptible_population__(size_to_infect=None, uninfectedS=None, indvd_change_factor=None, gbest=None, S=None, trainX=None, testX=None, trainy=None, testy=None, SEARCHABLE_LIMIT=None, t_type=None, dim=None, SC=None):
    f = lambda x: random.randint(0, (x))  
    tmp=[]
                
    diff=uninfectedS-size_to_infect
    if diff <= 0:
        size_to_infect=uninfectedS
    
    pop_size=len(S)-1

    #we need to keep count of actual number infected since IEOSA recognises that some individuals (i.e self.SC) 
    #are covered from infection due to immunity from self.SF members
    actual_size_infected=0
    for p in range(size_to_infect):            
        if uninfectedS <= SEARCHABLE_LIMIT: #linearly search for candidate to infect since pop is small
            for j in range(pop_size+1):                    
                idx, individual=S[j]                
                if idx is not None:
                    x=j
                #else:
                    #_boost_imunity_for_infection_escpades__(j, indvd_change_factor, gbest)            
        else: #randomly infect since uninfected population is still large
            isChecking=True
            while isChecking: #to ensure that we do not select an index which will return already infected
                x=f(pop_size)
                idx, individual=S[x]         
                if idx is not None:
                    isChecking=False
                #else:
                    #_boost_imunity_for_infection_escpades__(x, indvd_change_factor, gbest, S)
        
        #validating the requirment not to infect SC members
        isMember_SC=False
        for sc_mem in SC:
            _, mem_fit=sc_mem
            _, indx_fit=S[x]
            #compare to check if currently exposed member in S is a member of SC
            if (mem_fit[0]==indx_fit[0]).all():
                isMember_SC=True
            
        #Ensure that only non-SC members are infected
        if not(isMember_SC):
            original_index, individual=_weaken_imunity_of_infected__(x, indvd_change_factor, gbest, S, trainX, testX, trainy, testy)
            if t_type==1: 
                tx=stransform(deepcopy(individual[EOSA_ID_POS]), dim)
            else:
                tx=vtransform(deepcopy(individual[EOSA_ID_POS]), dim)
            individual[EOSA_ID_POS]=tx        
            tmp.append( (original_index, deepcopy(individual)))
            actual_size_infected=actual_size_infected+1 
        
    return tmp, actual_size_infected
    
def _boost_imunity_for_infection_escpades__(x, indvd_change_factor, gbest, S):
    #boost the imunity and self-protectionism of those individual who escapes infection
    escape_index, escape_ix=S[x]
    v = np.abs((np.random.rand()* indvd_change_factor) * (gbest[ID_POS]- deepcopy(escape_ix[ID_POS])))
    v = domain_range[0] if (v < domain_range[0]).all() else v
    v = domain_range[1] if (v > domain_range[1]).all() else v           
    escape_infected_ind= v
    escape_fit_infected = problem.fit_func(escape_infected_ind) 
    #_fitness_model__(solution=escape_infected_ind, minmax=self.MIN_MAX_INFECTED_SOL)
    S[x]=escape_index, [escape_infected_ind, escape_fit_infected]
    
def _weaken_imunity_of_infected__(x, indvd_change_factor, gbest, S, trainX, testX, trainy, testy):
    #weakens the imunity and self-protectionism of those individual who are infection
    original_index, ix=S[x] 
    l = np.random.uniform(-1, 1)
    infected_ind=(ix[EOSA_ID_POS] - gbest[ID_INDIVIDUAL][EOSA_ID_POS]) * (indvd_change_factor * np.exp(1 * l) * np.cos(2 * np.pi * l)) 
    fit_infected, cost = EOSA_fitness(infected_ind, trainX, testX, trainy, testy)
    individual=[infected_ind, fit_infected]
    S[x]=_symbol_of_infected_individual__(), individual #since it has been selected, mark a None i.e, infected individual                
    return original_index, individual