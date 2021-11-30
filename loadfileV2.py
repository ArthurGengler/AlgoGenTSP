# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 15:51:31 2021

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
import os
from operator import itemgetter

def get_box_plot_data(labels, bp):
    rows_list = []
    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = bp['whiskers'][i*2].get_ydata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
        dict1['median'] = bp['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_ydata()[1]
        rows_list.append(dict1) 
    return pd.DataFrame(rows_list), rows_list

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


number_figure = 0


Cities_25 = 'TSP_n25_1.npy'
Cities_50 = 'TSP_n50_1.npy'
Cities_100 = 'TSP_n100_1.npy'
Berlin_52 = 'berlin52.tsp'
att_48 = 'att48.tsp'

listOfCitySet = [Cities_25, Cities_50]

for citySet in listOfCitySet:
    for nameParameter in os.listdir('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/ArthurV2/'+citySet[:-4]):
        
        listfichier = os.listdir('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/ArthurV2/'+citySet[:-4]+'/'+nameParameter+'/')
        
        listData=[]
        for fichier in listfichier:
            listData.append(np.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/ArthurV2/'+citySet[:-4]+'/'+nameParameter+'/'+fichier,allow_pickle=True))
            
        dist = []
        temps = []
        ticks = []
        for data in listData:
            #print(data)
            dist.append(data[0])
            temps.append(data[1])
            ticks.append(nameParameter+ str(data[2][0].get(nameParameter)))
    
        """
        Internet_25 = np.load('Arthur/Internet25.npy',allow_pickle=True)
        modulebuild_50 = np.load('Arthur/prebuildmodule25.npy',allow_pickle=True)
        dist.append(modulebuild_50[0,:])
        dist.append(Internet_25[0,:])
        temps.append(modulebuild_50[1,:])
        temps.append(Internet_25[1,:])
        ticks.append("c1")
        ticks.append("module")
        """
        
        #rajouter sym='' dans boxplot si on veut pas les outliers
        
        plt.figure(number_figure)

        bpl = plt.boxplot(dist, positions=np.array(range(len(temps)))*2.0, labels=ticks)
        set_box_color(bpl, '#636363')
        plt.title(listData[0][2][0].get('citySet'))
        plt.ylabel("Distances")
        print(get_box_plot_data(ticks, bpl)[0])
        bpldata = get_box_plot_data(ticks, bpl)[1]
        

        plt.figure(number_figure+1)
        
        bpl = plt.boxplot(temps, positions=np.array(range(len(temps)))*2.0, labels=ticks)
        #set_box_color(bpl, '#636363')
        plt.title(listData[0][2][0].get('citySet'))
        plt.ylabel("Time[s]")
        print(get_box_plot_data(ticks, bpl)[0])
        bpldata = get_box_plot_data(ticks, bpl)[1]
        
        
        number_figure+=2
        
        
        