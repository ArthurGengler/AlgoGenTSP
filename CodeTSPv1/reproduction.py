# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 18:40:18 2021

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

def populationCrossoverDeterministic(best_pop):
    
    crossed_pop=[]
    crossed_pop.append(best_pop[0]) #garde le meilleur élément
    for i in range(1,len(best_pop)):
        locus = int(random.randrange(1, len(best_pop[i])-1, step=1))
        crossed_pop.append(singlePointcrossover(best_pop[i-1], best_pop[i], locus))
 
    return crossed_pop


def populationCrossoverRandom(best_pop):
    
    crossed_pop=[]
    crossed_pop.append(best_pop[0]) #garde le meilleur élément
    var =len(best_pop)-1
    
    if(len(best_pop)%2 != 0):
        crossed_pop.append(best_pop[0])
        var=var-1
        
    for i in range(0, var):
        route1=random.sample(best_pop, 1)
        route1=route1[0]
       
        route2=random.sample(best_pop, 1)
        route2=route2[0]

        locus = int(random.randrange(1, len(route1)-1, step=1))
        crossed_pop.append(singlePointcrossover(route1, route2, locus))

    return crossed_pop

def populationCrossoverRandom2Children(best_pop):
    
    crossed_pop=[]
    crossed_pop.append(best_pop[0]) #garde le meilleur élément
    var = (len(best_pop)//2) - (1-len(best_pop)%2)
    
    for i in range(0, var):
        
        route1=random.sample(best_pop, 1)
        route1=route1[0]
        best_pop.remove(route1)
        
        route2=random.sample(best_pop, 1)
        route2=route2[0]
        best_pop.remove(route2)
        
        locus = int(random.randrange(1, len(route1)-1, step=1))

        crossed_pop.append(singlePointcrossover(route1, route2, locus))
        crossed_pop.append(singlePointcrossover(route2, route1, locus))
        
    return crossed_pop


def populationDoubleCrossover(best_pop):
    """
    No remove but double crossover
    """
    crossed_pop=[]
    crossed_pop.append(best_pop[0]) #garde le meilleur élément
    var = (len(best_pop)//2) - (1-len(best_pop)%2)

    for i in range(0, var):
        
        route1=random.sample(best_pop, 1)
        route1=route1[0]
        
        route2=random.sample(best_pop, 1)
        route2=route2[0]
        
        locus1 = int(random.randrange(1, len(route1)-1, step=1))
        locus2 = int(random.randrange(1, len(route1)-1, step=1))
        
        while(locus1>locus2 or locus1==locus2):
            locus1 = int(random.randrange(1, len(route1)-1, step=1))
            locus2 = int(random.randrange(1, len(route1)-1, step=1))
        
        child1 = twoPointCrossover(route1, route2, locus1, locus2)
        child2 = twoPointCrossover(route2, route1, locus1, locus2)
        
        crossed_pop.append(child1)
        crossed_pop.append(child2)
        
    return crossed_pop    
 
def populationPMXCrossover(best_pop):
    
    crossed_pop=[]
    crossed_pop.append(best_pop[0]) #garde le meilleur élément
    
    var = (len(best_pop)//2) - (1-len(best_pop)%2)
    
    for i in range(0, var):
        
        route1=random.sample(best_pop, 1)
        route1=route1[0]
        best_pop.remove(route1)
        
        route2=random.sample(best_pop, 1)
        route2=route2[0]
        best_pop.remove(route2)
        
        (child1,child2) = PMX(route1, route2)
       
        crossed_pop.append(list(child1))
        crossed_pop.append(list(child2))
        
 
    return crossed_pop