# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:18:53 2021

@author: Oyelade
"""

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from matplotlib.ticker import ScalarFormatter
import numpy as np
import matplotlib.pyplot as plt
import math

def get_train_time_data(myfile=None, limit=9):
    history=[]
    with open(myfile, 'r') as read_obj:
        csv_reader = reader(read_obj)
        index=0
        for row in csv_reader:
            if index >= 1 and index <= limit:
                funcNumber=row[0]
                ABC=[row[1] ]#, row[13]] 
                ABC=[round(float(i), 1) for i in ABC]
                WOA=[row[2]]#, row[14]]
                WOA=[round(float(i), 1) for i in WOA]
                #BOA=[row[3], row[15]] BOA=[float(i) for i in BOA]
                PSO=[row[4]]#, row[16]]
                PSO=[round(float(i), 1) for i in PSO]
                #QSO=[row[5], row[17]] QSO=[float(i) for i in QSO]
                #CSO=[row[6], row[18]] CSO=[float(i) for i in CSO]
                EOSA=[row[7]]#, row[19]]
                EOSA=[round(float(i), 1) for i in EOSA]
                #DE=[row[8], row[20]] DE=[float(i) for i in DE]
                GA=[row[9]]#, row[21]]
                GA=[round(float(i), 1) for i in GA]
                #HGSO=[row[10], row[22]]  HGSO=[float(i) for i in HGSO]
                #BFO=[row[11], row[23]] BFO=[float(i) for i in BFO]
                #EOSA2=[row[12], row[4]] EOSA2=[float(i) for i in EOSA2]
                
                
                algos=[ABC, WOA, #BOA, 
                       PSO, #QSO, CSO, 
                       EOSA, #DE, 
                       GA#, HGSO, BFO, EOSA2
                      ]
                history.append(algos)                
            index+=1
    return history

myfile="../history/time_classical.csv"  #convergence time_classical, time_cec
#colorama.init()
#os.system("")

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
iterations = [1, 50, 100, 200, 300, 400, 500]
algorithms=['ABC','WOA',#'BOA',
            'PSO',#'QSO','CSO', 
            'EOSA', #'DE',
            'GA'#,'HGSO', 'BFO', 'EOSA2'
            ]
times=['']
X = np.arange(1)
benchmarkfuncs=['F1', 'F2', 'F3', 'F4', 'F5', 'F7', 'F8', 'F14', 'F20', 'F27', 'F33', 'F37', 'F17', 'F26', 'F45', 'F44']
#benchmarkfuncs=['CEC_F1', 'CEC_F2', 'CEC_F3', 'CEC_F4', 'CEC_F5', 'CEC_F6', 'CEC_F7', 'CEC_F8', 'CEC_F9', 'CEC_F10', 'CEC_F11', 'CEC_F12', 'C_F1', 'C_F9','C_F25', 'C_F30']
cols=4
rows=math.floor(len(benchmarkfuncs)/cols)
ylabel='Time (sec)'
obj=''

data= get_train_time_data(myfile, len(benchmarkfuncs))

fig, axs = plt.subplots(rows, cols, figsize=(10,5))
fig.subplots_adjust(wspace=0.5, hspace=0.2)


n=0
idx=0
fig.tight_layout(pad=1.0)
for i in range(rows):
    for j in range(cols): #color.BOLD+    +color.END
        axs[i, j].set_title(benchmarkfuncs[n]+" "+obj)
        n+=1
        bar_pos=0.00
        bar_width=0.10
        k=0        
        for algol in algorithms:
            dt=data[idx][k]
            rects=axs[i, j].bar(X + bar_pos, dt, width = bar_width, label=algol)
            autolabel(rects, axs[i, j])
            #axs[i, j].set_xticks(times, minor=False)
            bar_pos+=bar_width
            k+=1  
        idx+=1
        if i == rows -1 or j==0:
            xx = np.arange(len(times))
            axs[i, j].set_xticks(xx)
            if i == rows -1 and j==0:
                axs[i, j].set(xlabel='', ylabel=ylabel)
                axs[i, j].set_xticklabels(times)
            elif i == rows -1 and j>0:
                axs[i, j].set(xlabel='', ylabel='')                
                axs[i, j].set_xticklabels(times)
            else:
                axs[i, j].set(xlabel='', ylabel=ylabel)  
                axs[i, j].set_xticklabels([])
        else:
            axs[i, j].set_xticklabels([])
        
        axs[i, j].spines['right'].set_visible(False)
        axs[i, j].spines['top'].set_visible(False)
        axs[i, j].get_yaxis().set_major_formatter(ScalarFormatter()) 
        handles, labels = axs[i, j].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor =(0.75, 1.05), ncol = len(algorithms))
