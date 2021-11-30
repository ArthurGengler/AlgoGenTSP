# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 14:47:00 2021

@author: Arthur Gengler
"""

import numpy as np
import random
import copy
import operator
import array
import pandas as pd
import matplotlib.pyplot as plt
import time
import gzip
import tsplib95
import cProfile
import os
import numpy as np

from tspV2 import *


index=1 #Tous les fichiers avec le même index ont les même valeur par défaut

Cities_25 = 'TSP_n25_1.npy'
Cities_50 = 'TSP_n50_1.npy'
Cities_100 = 'TSP_n100_1.npy'
Berlin_52 = 'berlin52.tsp'
att_48 = 'att48.tsp'

listOfCitySet=[Cities_25]

#listNameParameter = ['popSize', 'nBest','nRandom', 'mutationRate', 'crossoverRate', 'numberMaxOfIteration', 'crossType', 'nbrSameValue','selecType']
listNameParameter = ['crossType']

dictPara = {'popSize':np.arange(10,101,20) , 'nBest':np.arange(0,10,2), 'nRandom':np.arange(0,50,5),
        'mutationRate':np.arange(0,1,0.1), 'crossoverRate':np.arange(0,1,0.1), 'numberMaxOfIteration':np.arange(10,20000,2000),
        'crossType':['Ordered','Breed','NewPMX', 'SinglePoint'], 'nbrSameValue':np.arange(100,10000,500),'selecType':['rank']}

#for nameParameter in dictPara.keys():
for nameParameter in listNameParameter:
    
    popSize = 100
    nBest = 10
    nRandom = 10
    mutationRate = 0.1
    crossoverRate = 0.9
    numberMaxOfIteration=5000
    nbrSameValue = 5000
    crossType = 'Breed'
    selecType = 'rank'
    
    for citySet in listOfCitySet:

        print(citySet[:-4])
        
        if(nameParameter) not in os.listdir('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/ArthurV2/'+citySet[:-4]+'/'):
            os.mkdir('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/ArthurV2/'+citySet[:-4]+'/'+str(nameParameter))   
            
        
        hyperParameters = [{'citySet' : citySet[:-4], 'popSize': str(popSize), 'nBest': str(nBest), 'nRandom': str(nRandom),
                           'mutationRate': str(mutationRate), 'crossoverRate': str(crossoverRate), 'numberMaxOfIteration': str(numberMaxOfIteration),
                           'crossType':str(crossType), 'selecType':str(selecType), 'nbrSameValue':str(nbrSameValue)}]
        
        for valuePara in dictPara.get(nameParameter):

            hyperParameters[0][nameParameter]=valuePara
            finalDist = []
            computeTime = []
            
            for i in range(5):
                start = time.time()
                if(nameParameter == 'popSize'):
                    popSize = valuePara
                if(nameParameter == 'nBest'):  
                    nBest = valuePara
                if(nameParameter == 'nRandom'):
                    nRandom = valuePara
                if(nameParameter == 'mutationRate'):  
                    mutationRate = valuePara
                if(nameParameter == 'crossoverRate'):
                    crossoverRate = valuePara
                if(nameParameter == 'numberMaxOfIteration'):
                    numberMaxOfIteration = valuePara
                if(nameParameter == 'crossType'):
                    print(1/(5-i))
                    crossType = valuePara
                if(nameParameter == 'nbrSameValue'):
                    nbrSameValue = valuePara
                if(nameParameter == 'selecType'):
                    selecType = valuePara
                    
                (bestDist,lastPop)=algo(popSize, nBest, nRandom, mutationRate, crossoverRate, crossType, selecType , citySet, numberMaxOfIteration, nbrSameValue)
                finalDist.append(bestDist)
                end = time.time()
                computeTime.append(end-start)
               
            np.save('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/ArthurV2/'+citySet[:-4]+'/'+ nameParameter +'/'+str(index)+'_'+citySet[:-4] +'_'+ crossType +'_'+ selecType +'_'+
                    'popS'+str(popSize) +'_'+ "nB"+str(nBest)+'_' +"nR"+str(nRandom)+'_' +"mR"+str(mutationRate)+'_' +"cR"+str(crossoverRate)+'_' +"ite"+str(numberMaxOfIteration)
                    +'_'+"same"+str(nbrSameValue),
                    [finalDist, computeTime, hyperParameters, index])
            #print(finalDist)
            
        
        
        
        