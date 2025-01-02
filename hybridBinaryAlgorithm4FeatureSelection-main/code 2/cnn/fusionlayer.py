# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:55:27 2023

@author: User
"""
from binaryOptimizer.model.algorithms.BEOSA_FUSION import beosafusion
import numpy as np
import random
import pytest
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from utils.IOUtil import save_results_to_csv
from numpy import array
#from utils.Config import *

EOSA_ID_POS = 0
EOSA_ID_FIT = 1
EOSA_PROBS_ID = 2
predictedLabelPos=-3 #position of predicted label 
actualLabelPos=-2 #actual label
fusedLabelPos=-1 #fused label
predictedProbsPos=-3 #position of predicted probs 
actualProbsPos=-2 #actual probs
fusedProbsPos=-1 #fused probs


def fusion_label_remap(y_test, y_pred, lablekeys, labelvals, namedclasses, labeltype):
    histo_lables={"N":['N'], "B":['B', 'A', 'F', 'PT', 'TA'], 
                  "M":['IS', 'IV', 'DC', 'LC', 'MC', 'PC']}
    mammo_lables={"N":['N'], "B":['BC', 'BM'], "M":['CALC', 'M']}
     
    n=0
    predclsnames=[]
    #We init with 0.0 becuase we do not want an empty list due to the computation we want to do
    probmap_3labels=[]
    
    #Stores the actual and predicted probability values
    probsinfo=[]
    
    for ytrue, ypredprobs in zip(y_test, y_pred):        
        labelpred=ypredprobs #ypredprobs[n] #preidcted
        index_max = max(range(len(labelpred)), key=labelpred.__getitem__)        
        classnamepred=namedclasses[index_max]#Get the real class label
        
        labeltrue=ytrue#true
        index_max = max(range(len(labeltrue)), key=labeltrue.__getitem__)
        classnametruesample=namedclasses[index_max]#Get the real class label
        
        #check where the real class labels falls in the 3-class-label remapping
        if labeltype=='histo':
            #for 1-hot encoding
            idx=0
            for newlabel, oldlabels in zip(histo_lables.keys(), histo_lables.values()):
                #loop over the sub-labels in histo_lables
                for label in oldlabels:
                    #if the original class label of this img is same as current label
                    if classnamepred == label:
                        # keep record of int representation of class for use in 1-hot encoding later
                        predclsnames.append(idx) 
                        ##increament for the first for-loop: "for newlabel, oldlabels in zip(histo_lables.keys(), ..."
                        idx=idx+1
            
            #Remap the old labels into their 3-count new label
            newclassnamepred=''
            newclassnametruesample=''
            for newlabel, oldlabels in zip(histo_lables.keys(), histo_lables.values()):
                
                #store the remapped label prediction labels
                if classnamepred in oldlabels:
                    newclassnamepred=newlabel
                
                #store the remapped label true labels
                if classnametruesample in oldlabels:
                    newclassnametruesample=newlabel
                    
            #3-prob-label remapping indexing
            idx=0 #But this will count up to 12 since there are 12-lables in histo old probmaps arrangement
            #We init with 0.0 becuase we do not want an empty list due to the computation we want to do
            newprobs=[0.0, 0.0, 0.0, 
                      newclassnamepred,#label for predicted  
                      newclassnametruesample, #label for true
                      '' #label for fused
                     ] 
            
            #Keep the 3-probs for true probability values
            truelabelprobs=[0.0, 0.0, 0.0]
            for oldtrueprobs, oldprobs in zip(ytrue, ypredprobs): #this loops for 12-times since histo has 12-labels
                #Since we a forming a 3-label mapping from the original 12-label,
                #we need to ensure that the the probabilities are well arranged according to the new labelling
                #e.g oldprobs=[0.05, 0.25, 0.05, 0.05, 0.1, 0.05, 0.05, 0.25, 0.05, 0.05, 0.1, 0.05], then newprobs=[0.1, 0.6, 0.3]
                #    oldprobs=["N","B","IS", "IV","A","F","PT", "TA", "DC", "LC", "MC","PC"]  newprobs=['N', 'B', 'M']
                #NB: all benign probs in old were summed into benign in new, same with other classes
                oldprobsidxlabel=namedclasses[idx]
                #find the index of oldprobsidxlabel in mammo_lables
                newidx=0
                for _, dlabels in zip(histo_lables.keys(), histo_lables.values()):
                    if oldprobsidxlabel in dlabels:
                        currentproblabel=newprobs[newidx]
                        probs=currentproblabel + oldprobs
                        newprobs[newidx]=probs
                        
                        #Do same 3-prob remapping for ytrue probability values
                        truecurrentproblabel=truelabelprobs[newidx]
                        trueprobs=truecurrentproblabel + oldtrueprobs
                        truelabelprobs[newidx]=trueprobs
                    newidx=newidx+1
                ##increament for the first inner for-loop: for newlabel, oldlabels in zip(histo_lables.keys(), ...
                idx=idx+1
            
            tmpnewprobs=newprobs[:-3] #interested in only all elements except last three
            indexmax = max(range(len(tmpnewprobs)), key=tmpnewprobs.__getitem__)
            #declare that the label for this new 3-map computed is the label of the maximum probability in newprobs 
            newprobs[-1]=list(histo_lables.keys())[indexmax]
            probmap_3labels.append(newprobs)
            #create a list of two items: (1) highest probability for true, (2) highest probability for predicted
            probsinfo.append([
                              max(truelabelprobs), #Item (1)  NB: that in truelabelprobs we have only 3 items
                              max(newprobs[:predictedLabelPos]) #Item (1)  NB: that in newprobs we have more than 3 items, we need the first 3 items excluding those starting from predictedLabelPos
                             ])
            
        
        if labeltype=='mammo':            
            #for 1-hot encoding
            idx=0
            for newlabel, oldlabels in zip(mammo_lables.keys(), mammo_lables.values()):
                #loop over the sub-labels in mammo_lables
                for label in oldlabels:
                    #if the original class label of this img is same as current label
                    if classnamepred == label:
                        # keep record of int representation of class for use in 1-hot encoding later
                        predclsnames.append(idx) 
                        ##increament for the first for-loop: "for newlabel, oldlabels in zip(mammo_lables.keys(), ..."
                        idx=idx+1
            
            #Remap the old labels into their 3-count new label
            newclassnamepred=''
            newclassnametruesample=''
            for newlabel, oldlabels in zip(histo_lables.keys(), histo_lables.values()):
                
                #store the remapped label prediction labels
                if classnamepred in oldlabels:
                    newclassnamepred=newlabel
                
                #store the remapped label true labels
                if classnametruesample in oldlabels:
                    newclassnametruesample=newlabel
                    
            #3-prob-label remapping
            idx=0
            #We init with 0.0 becuase we do not want an empty list due to the computation we want to do
            newprobs=[0.0, 0.0, 0.0,
                      newclassnamepred,#label for predicted  
                      newclassnametruesample, #label for true
                      '' #label for fused
                      ]
            #Keep the 3-probs for true probability values
            truelabelprobs=[0.0, 0.0, 0.0]
            for oldtrueprobs, oldprobs in zip(ytrue, ypredprobs): #this loops for 5-times since mammo has 5-labels
                #Since we a forming a 3-label mapping from the original 5-label,
                #we need to ensure that the the probabilities are well arranged according to the new labelling
                #e.g oldprobs=[0.1, 0.5, 0.1, 0.1, 0.2], then newprobs=[0.1, 0.6, 0.3]
                #    oldprobs=['N', 'BC', 'BM', 'CALC', 'M']  newprobs=['N', 'B', 'M']
                #NB: all benign probs in old were summed into benign in new, same with other classes
                oldprobsidxlabel=namedclasses[idx]
                #find the index of oldprobsidxlabel in mammo_lables
                newidx=0
                for _, dlabels in zip(mammo_lables.keys(), mammo_lables.values()):
                    if oldprobsidxlabel in dlabels:
                        currentproblabel=newprobs[newidx]
                        probs=currentproblabel + oldprobs
                        newprobs[newidx]=probs
                        
                        #Do same 3-prob remapping for ytrue probability values
                        truecurrentproblabel=truelabelprobs[newidx]
                        trueprobs=truecurrentproblabel + oldtrueprobs
                        truelabelprobs[newidx]=trueprobs
                    newidx=newidx+1
                ##increament for the first inner for-loop: for newlabel, oldlabels in zip(mammo_lables.keys(), ...
                idx=idx+1
                
            tmpnewprobs=newprobs[:-3] #interested in only all elements except last three
            indexmax = max(range(len(tmpnewprobs)), key=tmpnewprobs.__getitem__)
            #declare that the label for this new 3-map computed is the label of the maximum prob in newprobs 
            newprobs[-1]=list(mammo_lables.keys())[indexmax]
            probmap_3labels.append(newprobs)
            #create a list of two items: (1) highest probability for true, (2) highest probability for predicted
            probsinfo.append([
                              max(truelabelprobs), #Item (1)  NB: that in truelabelprobs we have only 3 items
                              max(newprobs[:predictedLabelPos]) #Item (1)  NB: that in newprobs we have more than 3 items, we need the first 3 items excluding those starting from predictedLabelPos
                             ])
            
        #increament for the outer for-loop: for ytrue, ypredprobs in zip(y_test, y_pred)
        n=n+1   
    
    #Generate the new namedclass for use in the 1-hot encoding 
    newnamedclass=[]
    if labeltype=='mammo':
       newnamedclass=mammo_lables.keys()
    if labeltype=='histo':
        newnamedclass=histo_lables.keys()
    #Convet the list of class labels index to numpy array
    categoricaltrue=array(predclsnames)
    #Do 1-hot encoding
    categoricaltrue = to_categorical(categoricaltrue, len(newnamedclass))        
    return probmap_3labels, probsinfo, categoricaltrue

def generate_fusion_population(histoprobs, mammoprobs, histo_probsinfo, mammo_probsinfo, fusion_info_dir):
    #declare an empty list to store the population to be derived
    populationprobs=[]
    
    stored_3labels=[]
    stored_3probmap=[]
    
    hist_counter=0
    #for each item in histoprobs, we want to map it to all items with similar labels in mammoprobs
    for histo in histoprobs:
        #We use -3 becuase predicted labels are in the last position of the list
        '''
        E.g: histo is =
          [0.00011233749683015049, 0.05490428452321794, 0.9449833922844846, '', 'B', '']
        '''
        print('GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG')
        print(histo)
        print(mammoprobs[0])
        print('GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG')
        mammo_counter=0
        #iterate over the items in mammoprobs to check if labels matches
        for mammo in mammoprobs:
            #We used -3 becuase predicted labels are in the third position in the rear of the list
            #if the predicted label of histo and mammo are same for form an individual of the population with that
            if histo[actualLabelPos]==mammo[actualLabelPos]: #where mammo[-3] is the predicted label for mammograpghy
                #join two list together
                '''
                e.g 
                [0.2, 0.5, 0.3, 'N', 'N', 'N'] + [0.1, 0.4, 0.5, 'N', 'N', 'N'] 
                   =
                [0.2, 0.5, 0.3, 0.1, 0.4, 0.5, 'N', 'N', 'N', 'N', 'N', 'N']
                '''
                #we include all elements bcos last three stores the pred, true labels, and fused label
                #Note that the end index is exclusive of (predictedProbsPos=-3) in e.g histo[:-2] OR mammo[:-2]
                individual=histo[:predictedProbsPos] + mammo[:predictedProbsPos] + histo[predictedProbsPos:] + mammo[predictedProbsPos:] 
                populationprobs.append(individual)
                #else:
                #individual=histo[:predictedProbsPos] + mammo[:predictedProbsPos] + histo[predictedProbsPos:] + mammo[predictedProbsPos:] 
                #populationprobs.append(individual)
            
            #store probs and lables whether there is a match or not
            #This will store fusion result (label and prob-map) for us according to the following:
            '''
            Sample #	Actual Predicted	| Actual Predicted	| Histology	Mammography	Fused
            classdict_row={'actualhisto':, 'predhisto':, 'actualmammo':, 'predmammo':, 
                           'fusedhisto':, 'fusedmammo':, 'fused':}
            '''            
            classdict_row={'actualhisto':histo[actualLabelPos], 
                           'predhisto':histo[predictedLabelPos], 
                           'actualmammo':mammo[actualLabelPos], 
                           'predmammo':mammo[predictedLabelPos], 
                           'fusedhisto':histo[fusedLabelPos], 
                           'fusedmammo':mammo[fusedLabelPos], 
                           # We wiil take both labels (histo and mammo) since they might not match
                           'fused':histo[fusedLabelPos]+' - '+mammo[fusedLabelPos] 
                           }
            histoItem=histo_probsinfo[hist_counter]
            mammoItem=mammo_probsinfo[mammo_counter]
            #Take the average of the histo and mammo probs to obtain the fused probability
            fusedprob=histoItem[1]+mammoItem[1]/2
            probdict_row={'actualhisto':histoItem[0], #at index=0 is the actual/true probability
                          'predhisto':histoItem[1], #at index=1 is the predicted probability
                          'actualmammo':mammoItem[0], #at index=0 is the actual/true probability
                          'predmammo':mammoItem[1], #at index=0 is the predicted probability
                          'fusedhisto':str(histoItem[0])+' - '+str(histoItem[1]), 
                          'fusedmammo':str(mammoItem[0])+' - '+str(mammoItem[1]), 
                           # We wiil take both labels (histo and mammo) since they might not match
                          'fused':fusedprob
                          }
            mammo_counter=mammo_counter+1
            save_results_to_csv(classdict_row, 'classdict', fusion_info_dir)
            save_results_to_csv(probdict_row, 'probdicts', fusion_info_dir)
        
        #Increament the hist_counter variable before returning to the for-loop: "for histo in histoprobs"
        hist_counter=hist_counter+1
            
    return populationprobs

def fusion_fitness_func(hprob, mprob):
    histo_real_prob=1.0
    mammo_real_prob=1.0
    multimodal_real_prob=histo_real_prob+mammo_real_prob
    diff=multimodal_real_prob - (sum(hprob)+sum(mprob))
    return diff

def problem(lb, ub, hprob, mprob):  
    def fitness_function(solution):
        #nonlocal model
        histo_real_prob=1.0
        mammo_real_prob=1.0
        multimodal_real_prob=histo_real_prob+mammo_real_prob
        diff=multimodal_real_prob - (sum(hprob)+sum(mprob))
        return diff

    problem = {
        "fit_func": fitness_function,
        "lb": [lb, lb, lb, lb, lb],
        "ub": [ub, ub, ub, ub, ub],
        "minmax": "min",
    }
    return problem

def store_remap_individuals(allfit, featCnt, AllSol, pr, optimize_fusion_info_dir):
    #Note that AllSol represents the optimize array of arrays for 1s|0s
    #recall that population contains 3-class probability maps 
    #for say N images with 6 probs (3 for histo, 3 for mammo) so that each individual is represented as follows:
    # individual[i]=(idx, [solution, fitness, probs])
    #where idx=0, 1, 2, ... N
    #      solution=[1, 0, 1, 1, 1, 0]
    #      fitness= 0.12
    #      probs=[0.2, 0.5, 0.3, 0.1, 0.4, 0.5, 'N', 'N', 'N', 'N', 'N', 'N']
    #seperating probs will be: [0.2, 0.5, 0.3, 'N', 'N', 'N'] for histo, [0.1, 0.4, 0.5, 'N', 'N', 'N'] for mammo
    '''
       AllSol=[
               [1, 0, 1, 1, 1, 0], row #1 in population
               [1, 0, 0, 1, 0, 1], row #2 in population
               [1, 0, 1, 0, 1, 1], row #3 in population
               ...
               [0, 0, 1, 0, 1, 0], row #N in population
              ]
       Then we assume that all places where 0s are we blind the corresponding features in x_train
       while were have 1s, the feature values are left as they are.
       We need a transformation function that we rewrite x_train before passing it to the prediction phase
       
    '''
    endOfHistoProbs=-3 #actually it ends at -2, but since this is slicing, we usually specify index to exlude. So -3 is excluded
    endOfMammoProbs=-6 #actually it ends at -5, but since this is slicing, we usually specify index to exlude. So -6 is excluded
    namedclasses=['N', 'B', 'M']
    for individual in AllSol:
        idx, indv=individual
        solution=indv[EOSA_ID_POS]
        fitness=indv[EOSA_ID_FIT]
        probs=indv[EOSA_PROBS_ID]
        
        histofusedLabelPos=predictedLabelPos-1 #becuase since predictedLabelPos is at -3, then -4 will give us fuse for histo in [0.2, 0.5, 0.3, 0.1, 0.4, 0.5, 'N', 'N', 'N', 'N', 'N', 'N']
        histoactualLabelPos=histofusedLabelPos-1 #-5 will give us the actual for histo in [0.2, 0.5, 0.3, 0.1, 0.4, 0.5, 'N', 'N', 'N', 'N', 'N', 'N']
        histopredictedLabelPos=histoactualLabelPos-1 #-6 will give use the predicted for histo in [0.2, 0.5, 0.3, 0.1, 0.4, 0.5, 'N', 'N', 'N', 'N', 'N', 'N']
        
        #extracts probs components
        histoprobs=probs[:endOfHistoProbs]
        mammoprobs=probs[endOfHistoProbs:endOfMammoProbs]
        
        #extract label component
        histpredlabel, histotruelabel, histofusedlabel=probs[histopredictedLabelPos], probs[histoactualLabelPos], probs[actualLabelPos]
        mammopredlabel, mammotruelabel, mammofusedlabel=probs[predictedLabelPos], probs[actualLabelPos], probs[fusedLabelPos]
        
        #find the new labels due to optimized solution
        #solution=[1, 0, 1, 1, 1, 0]
        histosolution=probs[:endOfHistoProbs] #[1, 0, 1]
        mammosolution=probs[endOfHistoProbs:endOfMammoProbs] #[1, 1, 0]
        
        #map histoprobs to histosolution, and map mammoprobs to mammosolution
        histomap=[histoprobs[i] for i in range(histosolution) if histosolution[i]==1]
        mammomap=[mammoprobs[i] for i in range(mammosolution) if mammosolution[i]==1]
        
        #find the mx in histomap and map the label for that max prob
        histo_index_max = histomap.index(max(histomap))
        mammo_index_max = histomap.index(max(mammomap))
        
        #Get the real class label from the 3-class format
        histonewlabel=namedclasses[histo_index_max]
        mammonewlabel=namedclasses[mammo_index_max]
        
        #Store the optimized result
        classdict_row={'actualhisto':histotruelabel, 
                       'predhisto':histpredlabel, 
                       'actualmammo':mammotruelabel, 
                       'predmammo':mammopredlabel, 
                       'fusedhisto':histofusedlabel, 
                       'fusedmammo':mammofusedlabel, 
                       # We wiil take both labels (histo and mammo) since they might not match
                       'fused':histofusedlabel+' - '+mammofusedlabel 
                       }
        #Take the average of the histo and mammo probs to obtain the fused probability
        fusedprob=max(histoprobs)+max(mammoprobs)/2
        probdict_row={'actualhisto':histoprobs, 
                      'predhisto':max(histoprobs), 
                      'actualmammo':mammoprobs, 
                      'predmammo':max(mammoprobs), 
                      'fusedhisto':' - ', 
                      'fusedmammo':' - ', 
                       # We wiil take both labels (histo and mammo) since they might not match
                      'fused':fusedprob
                     }
        save_results_to_csv(classdict_row, 'classdict', optimize_fusion_info_dir)
        save_results_to_csv(probdict_row, 'probdicts', optimize_fusion_info_dir)

        #store result
        pr.save_probs_optimized_solutions(solution, fitness, probs, histoprobs, mammoprobs, 
                                          histpredlabel, histotruelabel, mammopredlabel, mammotruelabel,
                                          histosolution, mammosolution, histonewlabel, mammonewlabel)
    return None
    
    
## Run the algorithm
def probsmap_optimizer(population, model_rates, pr, optimize_fusion_info_dir):
    #lb, ub=0.0, 1.0
    #prob=problem(lb, ub, hprob, mprob)
    MaxIter=50
    pop_size=len(population)
    print('================== Population Size ======================')
    print(pop_size)
    allfit, featCnt, AllSol=beosafusion(population, pop_size, MaxIter, model_rates)
    
    #Here is the transform function
    store_remap_individuals(allfit, featCnt, AllSol, pr, optimize_fusion_info_dir)
    return None
