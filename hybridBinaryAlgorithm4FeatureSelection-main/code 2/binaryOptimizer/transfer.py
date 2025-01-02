# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:42:11 2022

@author: Oyelade
"""
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import math
from matplotlib.ticker import ScalarFormatter
#from csv_parser import get_data
from csv import reader
from numpy import concatenate, savetxt, array
from csv import DictWriter
from os import getcwd, path, makedirs
from pandas import read_csv
import math
import random
import time

def z1(x): #np.sqrt(1 - np.power(2, x))
    return np.sqrt((1 - np.power(2, x))) #np.sqrt(np.power(x, 2) -1)

def z2(x): 
    return np.sqrt((1 - np.power(5, x))) 

def z3(x): 
    return np.sqrt((1 - np.power(8, x))) 

def z4(x): 
    return np.sqrt((1 - np.power(20, x))) 

def v2(x):
    return abs(np.tanh(x))

def s4(x):
    return 1/(1 + np.exp(-x/3))

def u3(x):
    epsilon=0.00001
    return epsilon * np.abs(x)

def q1(x):
    return condition(x, -1)

def q2(x):
    return condition(x, 0.5)

def q3(x):
    return condition(x, 2)

def q4(x):
    return condition(x, 3)

def condition(x, p):
    #print(gamma)
    mx=np.max(x)        
    gamma=0.5 * mx
    if x < gamma:
        nx=x/(0.5 * mx)
        if p > 0:
            return np.power(nx, p) 
        else:
            return np.abs(nx)
    if x >= gamma:
        return 1

def nested_v1(x): 
    return abs((x)/math.sqrt(2 + x * x))

def nested_v2(x):
    return abs(np.tanh(x))

def nested_s1(x):
    return 1/(1 + np.exp(-x/2))

def nested_s2(x):
    return 1 - 1/(1 + np.exp(x))

def plots_tranforms():
    lb, ub=-5, 5
    iterations=np.arange(lb, ub)
    
    stypes=['S1', 'S2'] 
    sdata={stypes[0]:[nested_s1(float(i)) for i in range(lb, ub)],   stypes[1]:[nested_s2(float(i)) for i in range(lb, ub)]}
    plt.title('S Transform functions')
    for algol in stypes:
        plt.plot(iterations, sdata[algol], label=algol, marker='o')
    plt.ylabel('S1,S2')
    plt.xlabel('x')
    plt.legend(stypes, bbox_to_anchor =(1.2, 0.9))
    plt.show()
    
    
    stypes=['V1', 'V2'] 
    sdata={stypes[0]:[nested_v1(float(i)) for i in range(lb, ub)],   stypes[1]:[nested_v2(float(i)) for i in range(lb, ub)]}
    plt.title('V Transform functions')
    for algol in stypes:
        plt.plot(iterations, sdata[algol], label=algol, marker='o')
    plt.ylabel('V1,V2')
    plt.xlabel('x')
    plt.legend(stypes, bbox_to_anchor =(1.2, 0.9))
    plt.show()
       
    
    stypes=['S1(V1)', 'S2(V1)'] 
    sdata={stypes[0]:[nested_s1(nested_v1(float(i))) for i in range(lb, ub)],   stypes[1]:[nested_s2(nested_v1(float(i))) for i in range(lb, ub)]}
    plt.title('SV1 Transform functions')
    for algol in stypes:
        plt.plot(iterations, sdata[algol], label=algol, marker='o')
    plt.ylabel('S1(V1),S2(V1)')
    plt.xlabel('x')
    plt.legend(stypes, bbox_to_anchor =(1.2, 0.9))
    plt.show()
    
    
    stypes=['S1(V2)', 'S2(V2)'] 
    sdata={stypes[0]:[nested_s1(nested_v2(float(i))) for i in range(lb, ub)],   stypes[1]:[nested_s2(nested_v2(float(i))) for i in range(lb, ub)]}
    plt.title('SV2 Transform functions')
    for algol in stypes:
        plt.plot(iterations, sdata[algol], label=algol, marker='o')
    plt.ylabel('S1(V2),S2(V2)')
    plt.xlabel('x')
    plt.legend(stypes, bbox_to_anchor =(1.2, 0.9))
    plt.show()
    
    
    stypes=['V1(S1)', 'V2(S1)'] 
    sdata={stypes[0]:[nested_v1(nested_s1(float(i))) for i in range(lb, ub)],   stypes[1]:[nested_v2(nested_s1(float(i))) for i in range(lb, ub)]}
    plt.title('VS1 Transform functions')
    for algol in stypes:
        plt.plot(iterations, sdata[algol], label=algol, marker='o')
    plt.ylabel('V1(S1),V2(S1)')
    plt.xlabel('x')
    plt.legend(stypes, bbox_to_anchor =(1.2, 0.9))
    plt.show()
    
    
    stypes=['V1(S2)', 'V2(S2)'] 
    sdata={stypes[0]:[nested_v1(nested_s2(float(i))) for i in range(lb, ub)],   stypes[1]:[nested_v2(nested_s2(float(i))) for i in range(lb, ub)]}
    plt.title('VS2 Transform functions')
    for algol in stypes:
        plt.plot(iterations, sdata[algol], label=algol, marker='o')
    plt.ylabel('V1(S2),V2(S2)')
    plt.xlabel('x')
    plt.legend(stypes, bbox_to_anchor =(1.2, 0.9))
    plt.show()

plots_tranforms()

def plots_tranforms_old():
    lb=-8
    ub=8
    iterations=np.arange(lb, ub)
    
    stypes=['S4'] 
    data=[s4(float(i)) for i in range(lb, ub)]
    p_1=[]
    for s in data:
        p_1.append(s)
    sdata={stypes[0]:p_1}
    plt.title('S Transform functions')
    for algol in stypes:
        plt.plot(iterations, sdata[algol], label=algol, marker='o')
    plt.ylabel('T(s)')
    plt.xlabel('s')
    plt.legend(stypes, bbox_to_anchor =(1.2, 0.9))
    plt.show()
    
    vtypes=['V1']
    data=[v2(i) for i in range(lb, ub)]
    p_1=[]
    for v in data:
        p_1.append(v)
    sdata={vtypes[0]:p_1}
    plt.title('V Transform functions')
    for algol in vtypes:
        plt.plot(iterations, sdata[algol], label=algol, marker='o')    
    plt.ylabel('T(v)')
    plt.xlabel('v')
    plt.legend(vtypes, bbox_to_anchor =(1.2, 0.9))
    plt.show()
    
    vtypes=['U3']
    data=[u3(i) for i in range(lb, ub)]
    p_1=[]
    for v in data:
        p_1.append(v)
    sdata={vtypes[0]:p_1}
    plt.title('U Transform functions')
    for algol in vtypes:
        plt.plot(iterations, sdata[algol], label=algol, marker='o')    
    plt.ylabel('T(u)')
    plt.xlabel('u')
    plt.legend(vtypes, bbox_to_anchor =(1.2, 0.9))
    plt.show()
    
    stypes=['Q1'] #, 'Q2', 'Q3', 'Q4'] 
    data=[(q1(i), q2(i), q3(i), q4(i)) for i in range(lb, ub)]
    p_1, p_2, p_3, p_4=[], [], [], []
    for s in data:
        s_1, s_2, s_3, s_4=s
        p_1.append(s_1)
        p_2.append(s_2)
        p_3.append(s_3)
        p_4.append(s_4)
    sdata={stypes[0]:p_1} #, stypes[1]:p_2, stypes[2]:p_3, stypes[3]:p_4}
    plt.title('Q Transform functions')
    for algol in stypes:
        plt.plot(iterations, sdata[algol], label=algol, marker='o')
    plt.ylabel('T(q)')
    plt.xlabel('q')
    plt.legend(stypes, bbox_to_anchor =(1.2, 0.9))
    plt.show()
    
    stypes=['Z1'] #, 'Z2', 'Z3', 'Z4'] 
    data=[(z1(float(i)), z2(float(i)), z3(float(i)), z4(float(i))) for i in range(lb, ub)]
    p_1, p_2, p_3, p_4=[], [], [], []
    for s in data:
        s_1, s_2, s_3, s_4=s
        p_1.append(s_1)
        p_2.append(s_2)
        p_3.append(s_3)
        p_4.append(s_4)
    sdata={stypes[0]:p_1} #, stypes[1]:p_2, stypes[2]:p_3, stypes[3]:p_4}
    plt.title('Z Transform functions')
    for algol in stypes:
        plt.plot(iterations, sdata[algol], label=algol, marker='o')
    plt.ylabel('T(z)')
    plt.xlabel('z')
    plt.legend(stypes, bbox_to_anchor =(1.2, 0.9))
    plt.show()