# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:02:04 2020

@author: Oyelade
"""

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import math
from matplotlib.ticker import ScalarFormatter
#from csv_parser import get_data


from csv import reader

def get_data(myfile=None, algorithms=None, iterations=0, benchmarkfuncs=[]):
    history=[]
    with open(myfile, 'r') as read_obj:
        csv_reader = reader(read_obj)
        index=0
        for row in csv_reader:
            funcNumber=row[0]
            if funcNumber in benchmarkfuncs:
                func=row[1]
                ABC=row[2]
                WOA=row[3]
                BOA=row[4]
                PSO=row[5]
                QSO=row[6]
                CSO=row[7]
                EOSA=row[8]
                DE=row[9]
                GA=row[10]
                HGSO=row[11] 
                BFO=row[12] 
                EOSA2=row[13] 
                
                
                algos=[ABC, WOA, #BOA, 
                       PSO, #QSO, CSO, 
                       EOSA, #DE, 
                       GA#, HGSO, BFO, EOSA2
                      ]
                algodict={}
                indxdict=0
                for raw_sols in algos:
                    raw_sols=raw_sols.split(',')
                    #print((raw_sols))
                    sols=[]
                    n=1
                    for rs in raw_sols:
                        if n in iterations:
                            point=float(((rs.strip()).replace(']', '')).replace('[', '') )
                            sols.append(point)
                        n+=1
                    algodict[algorithms[indxdict]]= sols#append()
                    indxdict+=1
                history.append(algodict)
                #solution=np.asarray(sols, dtype=np.float32)
                
            index+=1
    return history

myfile="../history/fit_cec.csv"  #convergence cec, classical
#colorama.init()
#os.system("")

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
   
iterations = [1, 50, 100, 200, 300, 400, 500]
algorithms=['ABC','WOA',#'BOA',
            'PSO',#'QSO','CSO', 
            'EOSA', #'DE',
            'GA'#,'HGSO', 'BFO', 'EOSA2'
            ]
#benchmarkfuncs=['F1', 'F2', 'F3', 'F4', 'F5', 'F7', 'F8', 'F14', 'F20', 'F27', 'F33', 'F37', 'F17', 'F26', 'F45', 'F44']
benchmarkfuncs=['CEC_F1', 'CEC_F2', 'CEC_F3', 'CEC_F4', 'CEC_F5', 'CEC_F6', 'CEC_F7', 'CEC_F8', 'CEC_F9', 'CEC_F10', 'CEC_F11', 'CEC_F12', 'C_F1', 'C_F9','C_F25', 'C_F30']
cols=4
rows=math.floor(len(benchmarkfuncs)/cols)
ylabel='Best'
obj=''

history= get_data(myfile, algorithms, iterations, benchmarkfuncs)

fig, axs = plt.subplots(rows, cols, figsize=(12,5))
fig.subplots_adjust(wspace=0.5, hspace=0.2)


n=0
idx=0
fig.tight_layout(pad=1.0)
for i in range(rows):
    for j in range(cols): #color.BOLD+    +color.END
        axs[i, j].set_title(benchmarkfuncs[n]+" "+obj)
        n+=1
        for algol in algorithms:
            #print(history[i][algol])
            axs[i, j].plot(iterations,history[idx][algol], label=algol, marker='o')
        idx+=1
        #axs[i, j].legend(bbox_to_anchor =(0.75, 1.15), ncol = len(algorithms)) #, loc="lower left"
        if i == rows -1 or j==0:
            if i == rows -1 and j==0:
                axs[i, j].set(xlabel='iterations', ylabel=ylabel)
            elif i == rows -1 and j>0:
                axs[i, j].set(xlabel='iterations', ylabel='')
            else:
                axs[i, j].set(xlabel='', ylabel=ylabel)             
        
        axs[i, j].get_yaxis().set_major_formatter(ScalarFormatter()) 
        axs[i, j].ticklabel_format(axis='y', style='sci', scilimits=[-11, 1], useOffset=True)
        handles, labels = axs[i, j].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor =(0.75, 1.09), ncol = len(algorithms))
