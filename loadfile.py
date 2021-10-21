# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 21:40:01 2021

@author: Arthur Gengler
"""

import numpy as np
import random
import copy
import operator
import pandas as pd
import matplotlib.pyplot as plt
import time
import gzip
import tsplib95

Arthur25 = np.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/Arthur/PMX100.npy',allow_pickle=True)
Mat25 = np.load('C:/Users/Arthur_Gengler/Downloads/Matthias/TTH2_TSP_n100_1_vACO_cycle__max_nb_rounds_1000__alpha_1__beta_2__rho_0.5__Q_100__eps_0.001.npy',allow_pickle=True)
distMat25 = Mat25[:,1]
distArthur25=Arthur25[0,:]


data_a = [distMat25, distArthur25]

ticks = ['ANT', 'GA']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0, sym='', widths=0.6)
set_box_color(bpl, '#636363') # colors are from http://colorbrewer2.org/


plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.tight_layout()
