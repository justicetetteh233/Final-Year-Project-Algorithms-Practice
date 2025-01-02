# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 08:07:04 2022

@author: Oyelade
"""

#Number of iterations
MaxIter = 50

#Initialization of accumulative list storage for fit and cost values
method_fit_pop=None
method_cost_pop=None

#Number of runnings for each experimentation of the algorithms
number_of_runs=2

#Variouse population sizes you want to investigate with
p_size = [50, 100] 

#Store results dir
result_dir='./results/'
dataset_dir='datasets/'
runs_dir='runs/'
bests_dir='bests/'
accuracy_dir='accuracy/'
fitness_dir='fitness/'
cost_dir='cost/'
metrics_dir='metrics/'
popsize_accuracy_dir='popsizeacc/'

#Classifiers
classifiers=['knn', 'rf', 'mlp', 'dt', 'svm', 'gnb']
#All datasets to be used for experimentation
    
#All algorithms to be experimented with
algorithms=[ #Note that:  ***-NT means not to use transfer function but only threshold
            #'HBEOSA-SA', #'HBEOSA-SA-NT',
            #'HBEOSA-FFA', 'HBEOSA-FFA-NT',
            #'HBEOSA-DMO', 'HBEOSA-DMO-NT',
            #'HBEOSA-PSO', 'HBEOSA-PSO-NT',
            'BEOSA',
        ]

datasetlist = [         
                'iris.csv',
                #'Lung.csv', 
                'Prostrate.csv',
                'Colon.csv',
                'Leukemia.csv',
    
                #"BreastEW.csv", 
                #"BreastCancer.csv",
                #"CongressEW.csv", 
                #"Exactly.csv", 
                #"Exactly2.csv", 
                #"HeartEW.csv", 
    
                #"Ionosphere.csv",
                #"M-of-n.csv", 
                #"PenglungEW.csv",
                #"Sonar.csv",
                #"SpectEW.csv", 
                #"Tic-tac-toe.csv", 
                #"Lymphography.csv", 
                
                #"Vote.csv", 
                #"Wine.csv", 
                #"Zoo.csv", 
                #"KrVsKpEW.csv",
                #"WaveformEW.csv" 
              ]
    