# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 18:54:48 2021

@author: Arthur Gengler
"""
import numpy as np
import random
import copy
import operator
import pandas as pd
import matplotlib.pyplot as plt
import time

from crossoverfctpmx import *
from mutation import *
from selection2 import *

def populationPMXCrossover2(sorted_pop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList):
    
    crossed_pop=[]
    for i in range(nBest):
        crossed_pop.append(sorted_pop[i]) #garde le meilleur élément
    
    nOffspring = popSize - nBest - nRandom
    
    for i in range(0, nOffspring//2):
        
        (parent1,parent2) = rankSelection(sorted_pop)
        
        if(crossoverRate>random.random()):    
            (child1,child2) = PMX(parent1, parent2)
        else:
            (child1,child2) = (parent1, parent2)
            
        if(random.random()<mutationRate):
            child1 = mutation(child1)
        if(random.random()<mutationRate):
            child2 = mutation(child2)

        crossed_pop.append(list(child1))
        crossed_pop.append(list(child2))
    
    if(popSize %2 == 1):
        nRandom+=1
    for i in range(nRandom):
        crossed_pop.append(random.sample(cityList, len(cityList)))
    
    
    return crossed_pop

def populationSinglePointCrossover2(sorted_pop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList):
    
    
    crossed_pop=[]
    for i in range(nBest):
        crossed_pop.append(sorted_pop[i]) #garde le meilleur élément
    
    nOffspring = popSize - nBest - nRandom
    
    for i in range(0, nOffspring//2):
        
        (parent1,parent2) = rankSelection(sorted_pop)
        
        if(crossoverRate>random.random()):          
            locus1 = int(random.randrange(1, len(parent1)-1, step=1))
            child1 =singlePointcrossover(parent1, parent2, locus1)
            child2 =singlePointcrossover(parent2, parent1, locus1) 
        else:
            (child1,child2) = (parent1, parent2)
            
        if(random.random()<mutationRate):
            child1 = mutation(child1)
        if(random.random()<mutationRate):
            child2 = mutation(child2)

        crossed_pop.append(list(child1))
        crossed_pop.append(list(child2))
    
    if(popSize %2 == 1):
        nRandom+=1
    for i in range(nRandom):
        crossed_pop.append(random.sample(cityList, len(cityList)))
    
    return crossed_pop