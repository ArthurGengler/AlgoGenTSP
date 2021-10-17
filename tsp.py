# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 19:06:11 2021

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
from reproduction import *
from mutation import *


""""------------#1 Initial population generated randomly-----------------"""


class City:
    """
    Class City each city has coordinate (x,y)
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def equal(self, city):
        if (self.x==city.x and self.y==city.y):
            return True
        return False
    
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
        
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


def createCityList(numberOfCities):
    """
    Create a random list of City of size numberOfCities
    """
    cityList = []
    for i in range(0,numberOfCities):
        cityList.append(City(x=int(random.randrange(0, 100, step=1)),
                             y=int(random.randrange(0, 100, step=1)))) 
    return cityList

def createKnownCityList(numberOfCities):
    """
    Create a known list of City of size numberOfCities
    """
    cityList = []
    for i in range(0,numberOfCities):
        cityList.append(City(x= i, y= 1)) 
    return cityList


def createCityListFromFile(data):
    cityList = []
    for i in range(0,len(data[0])):
            cityList.append(City(x=data[0][i],y=data[1][i]))
    return cityList


def createRoute(cityList, numberOfCities):
    """
    Creat a random route which go through each city once
    input : list of City
    output : list of City
    """
    #Return a k length list of UNIQUE elements chosen from the population sequence or set.
    route = random.sample(cityList, numberOfCities)
    return route


def createPopulation(popSize, cityList):
    """
    Create a population which is a list of route
    input : popSize = size of the wanted population (int) 
            cityList = list of City
    output : list of list of city
    """
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList, len(cityList)))
    return population


""""---------#2 Evaluate the objective function for each individual--------"""


def computeDistanceRoute(route):
    """
    Compute the distance to travel for a route
    imput : route = list of City
    output : total distance (float)
    """
    distance_tot = 0
    for i in range(len(route)-1):
        distance_tot = distance_tot + route[i].distance(route[i+1])
    distance_tot = distance_tot + route[len(route)-1].distance(route[0])
    return distance_tot


def sortPopulation(population):
    """
    Sort a population according to the total distance to travel
    imput : population : list of list of City
    output : population : list of list of City
    """
    #On veut la plus petite distance possible donc on classe du plus petit au plus grand
    return sorted(population, key = computeDistanceRoute, reverse = False)







""""-------------------#3 Selection of individual--------------------------"""

def selectionBestRoutes(sorted_pop, sizeSelected):
    """
    Keep the best half of a population
    imput : sorted_pop = population of size popSize sorted from best to worst 
    output : population of size popSize/2 sorted from best to worst
    """
    best_pop=[]
    for i in range(0,sizeSelected):
        best_pop.append(sorted_pop[i])
    return best_pop






"""-----------------------------#4 Crossover------------------------------ """




    
"""---------------------Reproduction-------------------------------------- """


"""-----------------------------#5 Mutation------------------------------ """



"""----------------------------#6 New gen ---------------------------------"""

def newGen(offspring, sizePop, cityList):
    """
    Create a new population of size sizePop based on a smaller population
    and complete which random route
    input : offspring = population 
    output : return a population of size = sizePop
    """
    new_pop=[]
    for i in range(0,len(offspring)):
        new_pop.append(offspring[i])
    for i in range(len(offspring), sizePop):
        new_pop.append(random.sample(cityList, len(cityList)))
    
    return new_pop



def algoGen(cityList, numberOfCities, sizePop, numberMaxOfIteration, 
              nbrSameValue, mutationRate, typeOfCrossover,plotBool):
    
    
    pop = createPopulation(sizePop, cityList)
    initialPopulation = copy.deepcopy(pop)
    listOfBestDistance =[computeDistanceRoute(pop[0])]
    i = 1
    count = 0
    while(count<nbrSameValue and i<numberMaxOfIteration):
        
        sortedPop = sortPopulation(pop)   
        listOfBestDistance.append(computeDistanceRoute(sortedPop[0]))
        selectedPop = selectionBestRoutes(sortedPop, sizePop-sizePop//5)
        
        #Crossover
        if(typeOfCrossover == "Random"):
            offspring = populationCrossoverRandom(selectedPop)
        
        if(typeOfCrossover == "Deterministic"):
            offspring = populationCrossoverDeterministic(selectedPop) 
            
        if(typeOfCrossover == "2Children"):
            offspring = populationCrossoverRandom2Children(selectedPop)
        
        if(typeOfCrossover == "Double"):
            offspring = populationDoubleCrossover(selectedPop)  
            
        if(typeOfCrossover == "PMX"):
            offspring = populationPMXCrossover(selectedPop)        
   
        #Mutation    
        if(random.random()<mutationRate):
            offspring = populationMutation(offspring)
            
        pop = newGen(offspring, sizePop, cityList)
        #Stopping criterion with delta
        if((listOfBestDistance[i-1]-listOfBestDistance[i])<0.001):
            count+=1
        else:
            count=0
        
        #Stopping criterion with nbr of total iteration
        i+=1
        
        
    #First pop
    xCitiesInit=[]
    yCitiesInit=[]
    for i in range(0,len(initialPopulation[0])):
        xCitiesInit.append(initialPopulation[0][i].getX())
        
    for i in range(0,len(initialPopulation[0])):
        yCitiesInit.append(initialPopulation[0][i].getY())
        

    
    #last pop
    xCities=[]
    yCities=[]  
    for i in range(0,len(pop[0])):
        xCities.append(pop[0][i].getX())
    xCities.append(pop[0][0].getX())
    for i in range(0,len(pop[0])):
        yCities.append(pop[0][i].getY())
    yCities.append(pop[0][0].getY())    

    if(plotBool==True):
        figure, axes = plt.subplots(1,4,figsize=(20,5))
    
        axes[0].plot(xCitiesInit, yCitiesInit,'ro')
        axes[0].set_xlabel('x coordinate')
        axes[0].set_ylabel('y coordinate')
        axes[0].title.set_text('Position of {} cities'.format(numberOfCities))
        
        axes[1].plot(xCitiesInit, yCitiesInit)
        axes[1].plot(xCitiesInit, yCitiesInit,'ro')
        axes[1].set_xlabel('x coordinate')
        axes[1].set_ylabel('y coordinate')
        axes[1].title.set_text('Initial route')
        
        axes[2].plot(xCities, yCities)
        axes[2].plot(xCities, yCities,'ro')
        axes[2].set_xlabel('x coordinate')
        axes[2].set_ylabel('y coordinate')
        axes[2].title.set_text("Route after {} iterations".format(numberMaxOfIteration))
        
        axes[3].plot(listOfBestDistance)
        axes[3].set_xlabel('iteration')
        axes[3].set_ylabel('distances')   
        axes[3].title.set_text('Evolution of the distance of the best route')
        
        figure.tight_layout()
    
    print(typeOfCrossover, listOfBestDistance[-1])
    return listOfBestDistance[-1]


""""-------------------------------Main----------------------------------- """


def main():
    
    #plotgraph(cityList, numberOfCities, sizePop, numberMaxOfIteration, mutationRate, plotBool)
    
    data = np.load('TSP_n25_1.npy')
    numberOfCities = len(data[0])
    cityList = createCityListFromFile(data)
    random.shuffle(cityList) 
    popSize = 30
    maxIt = 10**4
    maxEpsilon = 300
    mutationRate = 0.01 #10^-3
    
    
    for j in range(0,1):
        distanceFinalPMX1 = []
        for i in range(1):
            start = time.time()
            distanceFinalPMX1.append(algoGen(cityList, numberOfCities, popSize+30*j, maxIt, maxEpsilon, mutationRate, "Double", True))
            end = time.time()
            print("time",end-start)
            
    """     
    for j in range(1,4):
        distanceFinalPMX2 = []
        for i in range(5):
            start = time.time()
            distanceFinalPMX2.append(algoGen(cityList, numberOfCities, popSize, maxIt, maxEpsilon+300*j, mutationRate, "PMX", False))
            end = time.time()
            print("time",end-start)     
        plt.figure(j+3)
        plt.title("epsilon = {}".format(maxEpsilon+300*j))
        plt.boxplot(distanceFinalPMX2)
        plt.scatter([1,1,1,1,1],distanceFinalPMX2)   
        plt.show()    
    
    for j in range(1,4):
        distanceFinalPMX3 = []
        for i in range(5):
            start = time.time()
            distanceFinalPMX3.append(algoGen(cityList, numberOfCities, popSize, maxIt, maxEpsilon, mutationRate+0.12*j, "PMX", False))
            end = time.time()
            print("time",end-start) 
        plt.figure(j+6)
        plt.title("Mutation rate = {}".format(mutationRate+0.12*j))
        plt.boxplot(distanceFinalPMX3)
        plt.scatter([1,1,1,1,1],distanceFinalPMX3)   
        plt.show() 


    
    distanceFinalPMX = []
    for i in range(5):
        start = time.time()
        distanceFinalPMX.append(algoGen(cityList, numberOfCities, popSize, maxIt, maxEpsilon, mutationRate, "PMX", False))
        end = time.time()
        print("time",end-start)  
    
    distanceFinalRandom= []
    for i in range(5):
        start = time.time()
        distanceFinalRandom.append(algoGen(cityList, numberOfCities, popSize, maxIt, maxEpsilon, mutationRate, "Random", False))
        end = time.time()
        print("time",end-start)        
    distanceFinal2Children = []
    for i in range(5):
        start = time.time()
        distanceFinal2Children.append(algoGen(cityList, numberOfCities, popSize, maxIt, maxEpsilon, mutationRate, "2Children", False))
        end = time.time()
        print("time",end-start)   
    distanceFinalDouble=[]
    for i in range(5):
        start = time.time()    
        distanceFinalDouble.append(algoGen(cityList, numberOfCities, popSize, maxIt, maxEpsilon, mutationRate, "Double", False))
        end = time.time()
        print("time",end-start)  
    distanceFinalDeterministic = []
    for i in range(5):
        distanceFinalDeterministic.append(algoGen(cityList, numberOfCities, popSize, maxIt, maxEpsilon, mutationRate, "Deterministic", False))
    
    
    
    
    plt.figure(11)
    plt.title("PMX")
    plt.boxplot(distanceFinalPMX)
    plt.scatter([1,1,1,1,1],distanceFinalPMX)
    
    plt.figure(12)
    plt.title("Random")
    plt.boxplot(distanceFinalRandom)
    plt.scatter([1,1,1,1,1],distanceFinalRandom)
    
    plt.figure(13)
    plt.title("2Children")
    plt.boxplot(distanceFinal2Children)
    plt.scatter([1,1,1,1,1],distanceFinal2Children)
    
    plt.figure(14)
    plt.title("DoubleCross")
    plt.boxplot(distanceFinalDouble)
    plt.scatter([1,1,1,1,1],distanceFinalDouble)

    plt.figure(15)
    plt.title("Deterministic")
    plt.boxplot(distanceFinalDeterministic)
    plt.scatter([1,1,1,1,1],distanceFinalDeterministic)
    
    plt.show()
    """
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    