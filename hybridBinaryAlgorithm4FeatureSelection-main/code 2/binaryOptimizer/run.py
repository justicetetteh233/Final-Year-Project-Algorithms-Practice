# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 05:14:25 2022

@author: Oyelade
"""
import numpy as np
from utils.convergencePlot import plot_convergence, save_results_to_csv, save_best_fit, plots_from_dict,plots_s_v_tranforms
from utils.Config import *
from time import time
import statistics
from model.algorithms.BDMO import bdmo
from model.algorithms.BSFO import bsfo
from model.algorithms.BWOA import bwoa
from model.algorithms.BPSO import bpso
from model.algorithms.BGWO import bgwo
from model.algorithms.BSNDO import bsndo
from model.algorithms.BEOSA import beosa
from model.algorithms.BIEOSA import bieosa
from csv import reader

if __name__ == "__main__":
    #plots_s_v_tranforms()
    #exit()
    #Select the population size that you want to use to plot all
    #the convergence graph for fit vs iteration. Note that: p_size[3]=50 population
    selected_fit_pop=p_size[0] 
    recovery=False #False
    recovery_file_name='datasets-light-off-loadshedding-restart-helper'
    pop_size_recovery_file_name='popsize-light-off-loadshedding-restart-helper'
    i=0
    for filename  in datasetlist:
        accdata={}
        fitdata={}   
        costdata={}   
        
        #item={'dataset':filename,'algorithm': method,'fitdata':fitdata,
        #'accdata':accdata,'costdata':costdata,}
        completed_methods=[]
        if recovery:#then load existing data into the following arrays to continue experiment
            with open(result_dir+recovery_file_name+'.csv', 'r') as read_obj:
                csv_reader = reader(read_obj)
                index=0
                for row in csv_reader:
                    if index > 0 and len(row) > 0 and row[0]==filename:
                        
                        fitdata[row[1]]=row[2]
                        accdata[row[1]]=row[3]
                        costdata[row[1]]=row[4]
                        if not(row[1] in (completed_methods)):
                            completed_methods.append(row[1])
                    index=index+1
            
        for method in algorithms:
            #Check if a method has been completed before light-off/loadshedding
            #in that case, there is no need to experiment completed method(algorithm)
            print(completed_methods)
            if not(method in completed_methods):
                print(filename+' for '+str(completed_methods)+': Recovering from the '+method+' algorithm...')
                start_time=time()
                acc_pop=[]
                pop_size_acc={}
                
                #Create empty dict to store the average of accuracies in each classfier for those number_of_runs
                for clf in classifiers:
                    #To store the value of avg accuracy for the current pop_size for number_of_runs on classfier 'clf'
                    pop_size_acc[clf]={}
                    
                competed_pop_sizes=[]
                #setup for pop_size recovery purpose
                #item={'dataset':filename,'algorithm': method,'pop_size':pop_size,'method_fit_pop':method_fit_pop,
                #'method_cost_pop':method_cost_pop, 'fit_pop':fit_pop, 'cost_pop':cost_pop,
                #'pop_size_acc':pop_size_acc, }
                recovered_fit_pop={}
                recovered_cost_pop={}
                recovered_method_fit_pop={}
                recovered_method_cost_pop={}
                if recovery:#then load existing data into the following arrays to continue experiment
                    with open(result_dir+pop_size_recovery_file_name+'.csv', 'r') as read_obj:
                        csv_reader = reader((line.replace('\0','') for line in read_obj))
                        index=0
                        for row in csv_reader:
                            #print(str(index)+' '+row[0]+' '+row[1]+' '+row[2])
                            if index > 0 and row[0]==filename and row[1]==method: # and row[2]==str(pop_size):
                                #print(str(index)+filename+row[0]+str(row[2])+'n'+str(pop_size))                                                        
                                recovered_method_fit_pop[row[2]]=row[3]
                                recovered_method_cost_pop[row[2]]=row[4]
                                recovered_fit_pop[row[2]]=row[5]
                                recovered_cost_pop[row[2]]=row[6]
                                if not(row[2] in (competed_pop_sizes)):
                                    competed_pop_sizes.append(row[2])
                                tcount=7
                                for clf in classifiers:
                                    pop_size_acc[clf]['PSize:'+str(row[2])]=row[tcount]
                                    tcount=tcount+1
                            index=index+1
            
                #Check if a pop_size has been completed before light-off/loadshedding
                #in that case, there is no need to experiment completed pop_size for this method(algorithm)
                competed_pop_sizes=[int(item) for item in competed_pop_sizes]
                print(competed_pop_sizes)
                    
                for pop_size in p_size:
                    fit_pop={}
                    cost_pop={}
                    resulfilename=method
                    datasetname=dataset_dir+filename
                    
                    if not(pop_size in competed_pop_sizes): 
                        print(filename+' for '+str(competed_pop_sizes)+': Recovering from the '+method+' algorithm '+str(pop_size)+' pop_size...')
                        #exit()
                        #A dict to keep all accuracies for each classfier for all number_of_runs
                        clfAcc={}
                        for clf in classifiers:
                            clfAcc[clf]=[]
                            
                        #Array to store accuracies for only KNN classifier for all number_of_runs
                        accuArr = []
                        
                        #Arrays to keep values obtained for all number_of_runs
                        featArr = []
                        agenArr = []
                        allfitArr = []
                        allCostArr=[]
                        
                        for i in range(number_of_runs): #Run N times, you can change it to any number of runs
                            print('>>>Experiement:'+method+' with '+filename+', popsize:'+str(pop_size)+', run:'+str(i+1))
                            runfilename=filename+method+str(i)
                            metrics_result_dir=result_dir+metrics_dir
                                
                            if method=='BDMO':
                                allfit, allcost, testAcc, featCnt, gbest = bdmo(datasetname, pop_size, MaxIter, False, runfilename, metrics_result_dir)
                            if method=='BSNDO':
                                allfit, allcost, testAcc, featCnt, gbest = bsndo(datasetname, pop_size, MaxIter, runfilename, metrics_result_dir)
                            if method=='BSFO':    
                                allfit, allcost, testAcc, featCnt, gbest= bsfo(datasetname, pop_size, MaxIter, runfilename, metrics_result_dir)
                            if method=='BWOA':    
                                allfit, allcost, testAcc, featCnt, gbest= bwoa(datasetname, pop_size, MaxIter, runfilename, metrics_result_dir)
                            if method=='BPSO':    
                                allfit, allcost, testAcc, featCnt, gbest=bpso(datasetname, pop_size, MaxIter, runfilename, metrics_result_dir)
                            if method=='BGWO':        
                                allfit, allcost, testAcc, featCnt, gbest=bgwo(datasetname, pop_size, MaxIter, runfilename, metrics_result_dir)
                            if method=='BIEOSA':        
                                allfit, allcost, testAcc, featCnt, gbest=bieosa(datasetname, pop_size, MaxIter, runfilename, metrics_result_dir)
                            if method=='BEOSA':   
                                allfit, allcost, testAcc, featCnt, gbest=beosa(datasetname, pop_size, MaxIter, runfilename, metrics_result_dir)
                            
                            for clf in classifiers:
                                tmp=clfAcc[clf]
                                tmp.append(testAcc[clf][0])
                                clfAcc[clf]=tmp
                                
                            accuArr.append(testAcc['knn'][0])
                            featArr.append(featCnt)
                            allfitArr.append(allfit)
                            allCostArr.append(allcost)
                            agenArr.append(gbest)
                            run_item={
                            'dataset':filename,
                            'algorithm': method,
                            'popsize': pop_size,
                            'run_no': i,
                            'iteration':MaxIter,
                            'testAcc':testAcc['knn'][0],
                            'FeatureCount':featCnt,
                            'fitness':allfit,
                            'best':gbest,
                            }
                            save_results_to_csv(run_item, filename+method, result_dir+runs_dir)
                        #END of number_of_runs for a particular method on a particular dataset using a specified pop_size
                            
                        #Obtain the average of accuracies in each classfier for those number_of_runs
                        for clf in classifiers:
                            #Get a list of all accuracies for a specific classfier 'clf'
                            tmp=clfAcc[clf]
                            #Compute the average accuracy obtained for number_of_runs on classfier 'clf'
                            avgAcc=statistics.mean(tmp)
                            #Then store the value of avg accuracy for the current pop_size for number_of_runs on classfier 'clf'
                            pop_size_acc[clf]['PSize:'+str(pop_size)]=avgAcc
                        
                        #Get average/max: choose any
                        maxxAcc = statistics.mean(accuArr)
                        acc_pop.append(maxxAcc)
                        fit_pop[pop_size]=allfitArr[0]
                        cost_pop[pop_size]=allCostArr[0]
                        if selected_fit_pop == pop_size:
                            method_fit_pop=allfitArr[0]
                            method_cost_pop=allCostArr[0]
                            
                        k = np.argsort(accuArr)
                        bagent = agenArr[k[-1]]
                        currFeat= 20000
                        for i in range(np.shape(accuArr)[0]):
                            if accuArr[i]==maxxAcc and featArr[i] < currFeat:
                                currFeat = featArr[i]
                                bagent = agenArr[i]
                        datasetname = datasetname.split('.')[0]
                        print('Best: ', maxxAcc, currFeat)
                        save_best_fit(result_dir+bests_dir, resulfilename, datasetname, method, pop_size, maxxAcc, currFeat)
                        
                        '''
                        Data storage to help restarting the experiment after light-off/loadshedding
                        This data will recover the experiment for all results for pop_sizes on a 
                        particular dataset for a particular method
                        When loading it, we need to laod the data in the form of:
                            pop_size=pop_size
                            method_fit_pop=method_fit_pop
                            method_cost_pop=method_cost_pop
                            fit_pop=fit_pop
                            cost_pop=cost_pop
                        and then continue the experiment from where it stopped.
                        '''
                        item={
                            'dataset':filename,
                            'algorithm': method,
                            'pop_size':pop_size,
                            'method_fit_pop':method_fit_pop,
                            'method_cost_pop':method_cost_pop,
                            'fit_pop':fit_pop,
                            'cost_pop':cost_pop,
                        }
                        for clf in classifiers:
                            item[clf]=pop_size_acc[clf]['PSize:'+str(pop_size)]                    
                        save_results_to_csv(item, pop_size_recovery_file_name, result_dir)           
                    #End of the IF-bloc to check if a pop_size have been completed before light-off/loadshedding
                    else:
                        '''
                         Cache in memory all necessay values for the following so tha they will not be missing
                         when the all pop_sizes for this method on this dataset are completed:
                             fit_pop={}
                             cost_pop={}
                        '''
                        maxxAcc=None
                        datasetname = datasetname.split('.')[0]                            
                        with open(result_dir+bests_dir+method+'.csv', 'r') as read_obj:
                            csv_reader = reader((line.replace('\0','') for line in read_obj))
                            dset_algol=result_dir+bests_dir
                            for row in csv_reader:
                                if row[0]==datasetname and row[1]==method and row[2]==pop_size:                                
                                    maxxAcc = row[3]
                        acc_pop.append(maxxAcc)
                        fit_pop[pop_size]=recovered_fit_pop[str(pop_size)]
                        cost_pop[pop_size]=recovered_cost_pop[str(pop_size)]
                        if selected_fit_pop == pop_size:
                            method_fit_pop=recovered_method_fit_pop[str(pop_size)]
                            method_cost_pop=recovered_method_cost_pop[str(pop_size)]
                    #End of the ELSE-Block
                #End of all pop_sizes for a particular method on a particular dataset
                
                #computes and store values for some parameters/metrics
                time_required = time() - start_time
                accdata[method]=acc_pop
                fitdata[method]=method_fit_pop
                costdata[method]=method_cost_pop
                item={
                        'dataset':filename,
                        'algorithm': method,
                        'popsize': p_size,#i.e all pop_sizes e.g [25, 50, 75, 100]
                        'iteration':MaxIter,
                        'allfits':fit_pop,
                        'cost_pop':cost_pop,
                        'allacc':acc_pop,
                        'computation_time':time_required,
                    }
                save_results_to_csv(item, filename+method, result_dir)            
                
                
                '''
                Data storage to help restarting the experiment after light-off/loadshedding
                This data will recover the experiment so that results of completed methods and
                all their pop_sizes on a particular dataset are available for recovery to plot final graphs below
                When loading it, we need to laod the data in the form of:
                    accdata[method]=acc_pop
                    fitdata[method]=method_fit_pop
                    costdata[method]=method_cost_pop
                and then continue the experiment from where it stopped.
                Please check file: filename+method,
                '''
                item={
                        'dataset':filename,
                        'algorithm': method,
                        'fitdata':fitdata,
                        'accdata':accdata,
                        'costdata':costdata,                    
                    }
                save_results_to_csv(item, recovery_file_name, result_dir)           
                
                '''
                So, we will plot the values of accuracy obtained for all
                pop_sizes for this particular algorithm on this particular dataset
                '''
                pathsave=result_dir+popsize_accuracy_dir
                #Plot acc/p_size graphs for all classifiers 
                plotfilename=method+'_psize_acc_'+filename
                #print(pop_size_acc)
                #print(classifiers)
                #print(p_size)
                plots_from_dict(method+':PSizes Accuracy - '+filename, 'Accuracy', 'Population Size', pathsave, 
                             plotfilename, pop_size_acc, p_size, classifiers)
            #End of the IF-bloc to check if a method(algorithm) have been completed before light-off/loadshedding
        #END of all methods for a particular dataset
            
        #plot accuracy
        print(accdata)
        print(p_size)
        print(algorithms)
        pathsave=result_dir+accuracy_dir
        plotfilename=filename+'_acc_chart'
        plot_convergence('accuracy - '+filename, 'Accuracy', 'Population Size', pathsave, 
                         plotfilename, accdata, p_size, algorithms)
        #plot fitness
        pathsave=result_dir+fitness_dir
        plotfilename=filename+'_fit_chart'
        plot_convergence('convergence - '+filename, 'Fitness', 'Iteration', pathsave, 
                         plotfilename, fitdata, np.arange(MaxIter), algorithms)
        
        #plot cost values
        pathsave=result_dir+cost_dir
        plotfilename=filename+'_cost_chart'
        plot_convergence('cost - '+filename, 'Cost', 'Iteration', pathsave, 
                         plotfilename, costdata, np.arange(MaxIter), algorithms)
        i=i+1
