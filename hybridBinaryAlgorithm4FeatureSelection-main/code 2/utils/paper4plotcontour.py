# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:02:04 2020

@author: Oyelade
"""
import requests
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import cv2
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import math
from matplotlib.ticker import ScalarFormatter

myfile="../history/graphs/"
cmap = [(0, '#2f9599'), (0.45, '#eee124'), (1, '#8800ff')]
cmap = cm.colors.LinearSegmentedColormap.from_list('Custom', cmap, N=256)

iterations = [1, 50, 100, 200, 300, 400, 500]
algorithms=['ABC','WOA',#'BOA',
            'PSO',#'QSO','CSO', 
            'EOSA', #'DE',
            'GA'#,'HGSO', 'BFO', 'EOSA2'
            ]
#benchmarkfuncs=['CEC_F1', 'CEC_F2', 'CEC_F3']#, 'CEC_F4', 'CEC_F5', 'CEC_F6', 'CEC_F7', 'CEC_F8', 'CEC_F9', 'CEC_F10', 'CEC_F11', 'CEC_F12']
ylabel='Best'

'''
                'F1-3D', 'F1-2D', 'F1-EOSA', 'F1-ABC', 'F1-WOA', 
                'F34-3D','F34-2D', 'F34-EOSA', 'F34-PSO','F34-GA',
                'F3-3D', 'F3-3D', 'F3-EOSA',  'F3-PSO', 'F3-QSO', 
                'F32-3D', 'F32-2D', 'F32-EOSA', 'F32-BOA', 'F32-ABC', 
                
                'F28-3D', 'F28-2D', 'F28-EOSA', 'F28-PSO', 'F28-GA', 
                'F45-3D', 'F45-2D', 'F45-EOSA', 'F45-DE', 'F45-ABC', 
                'F46-3D', 'F46-2D', 'F46-EOSA', 'F46-BOA', 'F46-DE', 
                'F7-3D', 'F7-2D', 'F7-EOSA', 'F7-BOA', 'F7-GA', 
'''
'''
          'ackleyc.PNG', 'ackleyf.PNG', 'ackley_eosa.png', 'ackley_abc.png', 'ackley_woa.png',
          'spherec.PNG', 'spheref.PNG', 'sphere_eosa.png', 'sphere_pso.png', 'sphere_ga.png',
          'brownc.PNG', 'brownf.PNG',  'brown_eosa.png', 'brown_pso.png', 'brown_qso.png',
          'schwefel2_21c.PNG', 'schwefel2_21f.PNG', 'schwefel2_21_eosa.png', 'schwefel221_boa.png', 'schwefel221_abc.png',
          
          
          'rotatedhyperellipsoidc.PNG', 'rotatedhyperellipsoidf.PNG', 'rotatedhyperellipsoid_eosa.png', 'rotatedhyperellipsoid_pso.png', 'rotatedhyperellipsoid_ga.png',
          'zakharovc.PNG', 'zakharovf.PNG', 'zakharov_eosa.png', 'zakharov_de.png', 'zakharov_abc.png',
          'salomonc.PNG', 'salomonf.PNG', 'salomon_eosa.png', 'salomon_boa.png', 'salomon_de.png',
          'dixonc.PNG', 'dixonf.PNG', 'dixon_eosa.png', 'dixon_boa.png', 'dixon_ga.png'
'''
benchmarkfuncs=[
                'F1-3D', 'F1-2D', 'F1-EOSA', 'F1-ABC', 'F1-WOA', 
                'F34-3D','F34-2D', 'F34-EOSA', 'F34-PSO','F34-GA',
                'F3-3D', 'F3-3D', 'F3-EOSA',  'F3-PSO', 'F3-QSO', 
                'F32-3D', 'F32-2D', 'F32-EOSA', 'F32-BOA', 'F32-ABC',                 
                ]
history= [
          'ackleyc.PNG', 'ackleyf.PNG', 'ackley_eosa.png', 'ackley_abc.png', 'ackley_woa.png',
          'spherec.PNG', 'spheref.PNG', 'sphere_eosa.png', 'sphere_pso.png', 'sphere_ga.png',
          'brownc.PNG', 'brownf.PNG',  'brown_eosa.png', 'brown_pso.png', 'brown_qso.png',
          'schwefel2_21c.PNG', 'schwefel2_21f.PNG', 'schwefel2_21_eosa.png', 'schwefel221_boa.png', 'schwefel221_abc.png',          
         ]
cols=5
rows=math.floor(len(history)/cols) 

fig, axs = plt.subplots(rows, cols, figsize=(12,12))
fig.subplots_adjust(wspace=0.0, hspace=0.0)
n=0
idx=0
for i in range(rows):
    for j in range(cols): #color.BOLD+    +color.END
        axs[i, j].set_title(benchmarkfuncs[idx])
        n+=1
        img= myfile+history[idx]
        img=cv2.imread(img)
        axs[i, j].imshow(img)        
        axs[i, j].spines['right'].set_visible(False)
        axs[i, j].spines['top'].set_visible(False)
        axs[i, j].spines['left'].set_visible(False)
        axs[i, j].spines['bottom'].set_visible(False)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        idx+=1