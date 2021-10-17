# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:22:31 2021

@author: Arthur Gengler
"""
from crossoverfctpmx import *
from tsp import *
import random

def populationPMXCrossover(best_pop):
    
    crossed_pop=[]
    crossed_pop.append(best_pop[0]) #garde le meilleur élément
    
    var = (len(best_pop)//2) - (1-len(best_pop)%2)
    
    for i in range(0, var):
        
        route1=random.sample(best_pop, 1)
        route1=route1[0]
        #print("best before",best_pop)
        #print("r1",len(route1))
        best_pop.remove(route1)
        #print("best after1",best_pop)
        
        route2=random.sample(best_pop, 1)
        route2=route2[0]
        best_pop.remove(route2)
        #print("best after2",best_pop)
        
        
        (child1,child2) = PMX(route1, route2)
        print("c1",child1)
        print("c2",child2)
        crossed_pop.append(list(child1))
        crossed_pop.append(list(child2))
        #print("crossed",crossed_pop)
 
    return crossed_pop


numberOfCities = 15

parent1 = createKnownCityList(numberOfCities)
random.shuffle(parent1) 

parent2 = createKnownCityList(numberOfCities)
random.shuffle(parent2)

parent3 = createKnownCityList(numberOfCities)
random.shuffle(parent3) 

parent4 = createKnownCityList(numberOfCities)
random.shuffle(parent4)

parent5 = createKnownCityList(numberOfCities)
random.shuffle(parent5) 

parent6 = createKnownCityList(numberOfCities)
random.shuffle(parent6)

(o1,o2)=PMX(parent1, parent2)


crossedpopppp=populationPMXCrossover([parent1, parent2, parent3, parent4, parent5, parent6])
print(crossedpopppp)

muatationRate = 0.01
for i in range(200):
    if(random.random()<muatationRate):
        print('a')
    