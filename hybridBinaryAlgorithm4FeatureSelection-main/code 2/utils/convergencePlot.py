# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:40:46 2022

@author: Oyelade
"""


import matplotlib.pyplot as plt
plt.style.use('bmh')
import numpy as np
import math
from matplotlib.ticker import ScalarFormatter
#from csv_parser import get_data
from csv import reader
from numpy import concatenate, savetxt, array
from csv import DictWriter
from os import getcwd, path, makedirs
from pandas import read_csv
from binaryOptimizer.model.root import *


def save_best_fit(pathsave, filename, datasetname, method, pop_size, maxxAcc, currFeat, avgFeatureCount):
    check_directory = getcwd() + "/" + pathsave
    if not path.exists(check_directory):
        makedirs(check_directory)
    with open(pathsave + filename + ".csv", "a") as f:
        print(datasetname, method, pop_size, maxxAcc, currFeat, avgFeatureCount, sep=',', file=f)

def save_results_to_csv(item=None, filename=None, pathsave=None):
    check_directory = getcwd() + "/" + pathsave
    if not path.exists(check_directory):
        makedirs(check_directory)
    with open(pathsave + filename + ".csv", 'a') as file:
        w = DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=item.keys())
        if file.tell() == 0:
            w.writeheader()
        w.writerow(item)
        
def plot_convergence(title, ylabel, xlabel, pathsave, filename, history, iterations, algorithms):
    check_directory = getcwd() + "/" + pathsave
    if not path.exists(check_directory):
        makedirs(check_directory)
        
    plt.title(title)
    for algol in algorithms:
        plt.plot(iterations, history[algol], label=algol, marker='o')
    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(algorithms, bbox_to_anchor =(1.2, 0.9))
    plt.savefig(pathsave + filename + ".png")
    plt.show()

def plots_s_v_tranforms():
    lb=-7
    ub=7
    iterations=np.arange(lb, ub)
    stypes=['S1', 'S2', 'S1(V1)', 'S1(V2)', 'S2(V1)', 'S2(V2)'] #np.arange(
    data=[(s1(i), s2(i), s1(t1(i)), s1(t2(i)), s2(t1(i)), s2(t2(i))) for i in range(lb, ub)]
    p_1, p_2, p_3, p_4, p_5, p_6=[], [], [], [], [], []
    for s in data:
        p1, p2, p3, p4, p5, p6=s
        p_1.append(p1)
        p_2.append(p2)
        p_3.append(p3)
        p_4.append(p4)
        p_5.append(p5)
        p_6.append(p6)
    sdata={stypes[0]:p_1, stypes[1]:p_2, stypes[2]:p_3, stypes[3]:p_4, stypes[4]:p_5, stypes[5]:p_6}
    
    plt.title('S Transform functions')
    for algol in stypes[:2]:
        plt.plot(iterations, sdata[algol], label=algol, marker='.')
    plt.ylabel('T(S)')
    plt.xlabel('S')
    plt.legend(stypes[:2], bbox_to_anchor =(1.2, 0.9))
    plt.show()
    
    
    plt.title('S(V) Transform functions')
    for algol in stypes[2:]:
        plt.plot(iterations, sdata[algol], label=algol, marker='.')
    plt.ylabel('T(S)')
    plt.xlabel('S')
    plt.legend(stypes[2:], bbox_to_anchor =(1.2, 0.9))
    plt.show()
    
    
    vtypes=['V1', 'V2', 'V1(S1)', 'V1(S2)', 'V2(S1)', 'V2(S2)']
    data=[(t1(i), t2(i), t1(s1(i)), t1(s2(i)), t2(s1(i)), t2(s2(i))) for i in range(lb, ub)]
    p_1, p_2, p_3, p_4, p_5, p_6=[], [], [], [], [], []
    for v in data:
        p1, p2, p3, p4, p5, p6=v
        p_1.append(p1)
        p_2.append(p2)
        p_3.append(p3)
        p_4.append(p4)
        p_5.append(p5)
        p_6.append(p6)
    sdata={vtypes[0]:p_1, vtypes[1]:p_2, vtypes[2]:p_3, vtypes[3]:p_4, vtypes[4]:p_5, vtypes[5]:p_6}
    
    plt.title('V Transform functions')
    for algol in vtypes[:2]:
        plt.plot(iterations, sdata[algol], label=algol, marker='.')
    plt.ylabel('T(V)')
    plt.xlabel('V')
    plt.legend(vtypes[:2], bbox_to_anchor =(1.2, 0.9))
    plt.show()
    
    plt.title('V(S) Transform functions')
    for algol in vtypes[2:]:
        plt.plot(iterations, sdata[algol], label=algol, marker='.')
    plt.ylabel('T(V)')
    plt.xlabel('V')
    plt.legend(vtypes[2:], bbox_to_anchor =(1.2, 0.9))
    plt.show()
    
def plots_from_dict(title, ylabel, xlabel, pathsave, filename, history, iterations, algorithms):
    plt.title(title)
    check_directory = getcwd() + "/" + pathsave
    if not path.exists(check_directory):
        makedirs(check_directory)
        
    for algol in algorithms:
        data=[v for k, v in history[algol].items()]
        plt.plot(iterations, data, label=algol.upper(), marker='.')
    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(algorithms, bbox_to_anchor =(1.2, 0.9))
    plt.savefig(pathsave + filename + ".png")
    plt.show()
    '''
    ylabel='Acc'
    xlabel='PSize'
    pathsave='./'
    filename='testme.png'
    algorithms=['knn', 'rf', 'mlp', 'dt', 'svm', 'gnb']
    history={'knn': 
               #[0.9473684210526316, 0.9385964912280702,0.956140350877193,0.9473684210526316], 
               {'PSize:25': 0.9473684210526316, 'PSize:50': 0.9385964912280702, 'PSize:75': 0.956140350877193, 'PSize:100': 0.9473684210526316}, 
             'rf': 
               #[0.9780701754385965,0.9429824561403508, 0.9561403508771931,0.9385964912280702],
               {'PSize:25': 0.9780701754385965, 'PSize:50': 0.9429824561403508, 'PSize:75': 0.9561403508771931, 'PSize:100': 0.9385964912280702}, 
             'mlp': 
               #[0.8070175438596492, 0.8859649122807017, 0.8728070175438596, 0.9078947368421053], 
               {'PSize:25': 0.8070175438596492, 'PSize:50': 0.8859649122807017, 'PSize:75': 0.8728070175438596, 'PSize:100': 0.9078947368421053}, 
             'dt': 
               #[0.9342105263157894, 0.9166666666666667, 0.9122807017543859, 0.8903508771929824], 
               {'PSize:25': 0.9342105263157894, 'PSize:50': 0.9166666666666667, 'PSize:75': 0.9122807017543859, 'PSize:100': 0.8903508771929824}, 
             'svm': 
               #[0.9692982456140351, 0.956140350877193, 0.9429824561403508, 0.9692982456140351], 
               {'PSize:25': 0.9692982456140351, 'PSize:50': 0.956140350877193, 'PSize:75': 0.9429824561403508, 'PSize:100': 0.9692982456140351}, 
             'gnb': 
               #[0.9473684210526316, 0.9385964912280702, 0.956140350877193, 0.9473684210526316]
               {'PSize:25': 0.9605263157894737, 'PSize:50': 0.9473684210526316, 'PSize:75': 0.9342105263157895, 'PSize:100': 0.9298245614035088}
            } 
    iterations=[25, 50, 75, 100]
    
    '''
