# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 19:06:11 2021

@author: Arthur Gengler
"""

import numpy as np, random, copy, operator, pandas as pd, matplotlib.pyplot as plt




"""-----------------------------Parameters---------------------------------"""

numberOfCities = 10
sizePop=50
cityList = []





""""------------#1 Initial population generated randomly-----------------"""
class City:
    """
    Class city each city has coordinate (x,y)
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
        
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
    
def isEqual(City1,City2):
    """
    Check if two cities are equal
    input : Objects City
    output : Boolean 
    """
    if (City1.getX()==City2.getX() and City1.getY()==City2.getY()):
        return True
    return False

def createRoute(cityList):
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
        population.append(createRoute(cityList))
    return population






""""---------#2 Evaluate the objective function for each individual--------"""


def computeDistanceRoute(route):
    """
    Compute the distance to travel for a route
    imput : route = list of City
    output : total distance (float)
    """
    distance_tot = 0
    for i in range(numberOfCities-1):
        distance_tot = distance_tot + route[i].distance(route[i+1])
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
def selectionBestRoutes(sorted_pop):
    """
    Keep the best half of a population
    imput : sorted_pop = population of size popSize sorted from best to worst 
    output : population of size popSize/2 sorted from best to worst
    """
    best_pop=[]
    for i in range(0,sizePop//2):
        best_pop.append(sorted_pop[i])
    return best_pop






"""-----------------------------#4 Crossover------------------------------ """
def crossover(route1,route2):
    """
    Single crossover at the half 1234 and 2412 give 1243 
    input : two routes (list of City) 
    output : retrun a route which results from the crossover
    """
    new_route=[]
    for i in range(0,numberOfCities//2):                #Prend la 1er moitier de parent 1
        new_route.append(route1[i])
        
    for i in range(numberOfCities//2,numberOfCities):   #Prend la 2eme moitier du parent 2
        taken=False
        for j in range(0,numberOfCities//2):
            if (isEqual(route2[i],route1[j])):          #Vérifie que aucune ville du parent 2 n'a déjà été prisent via parent 1
                taken=True
        if (taken==False):
            new_route.append(route2[i])            
        else:
            new_route.append(City(x=-1, y=-1))          #Si une ville a déjà été prise on met un -1 à la place
    for i in range(numberOfCities//2,numberOfCities):   #S'occupe des endroit où il y a un 0 pour lui trouver une ville par laquelle on passe pas encore
        if (isEqual(new_route[i],City(x=-1, y=-1))):
            new_route[i]=notTaken(new_route,route2)
    return new_route


def notTaken(liste1, liste2):    
    """
    Find a City among liste2 which is not in liste1
    We suppose that there exist one
    input : liste1, liste2 = list of City (route)
    output : retrun a City 
    """
    for i in range(len(liste2)):
        taken = False
        for j in range(len(liste1)):
            if isEqual(liste2[i],liste1[j]):
                taken = True
        if taken == False:
            return liste2[i]
    

def populationCrossover(best_pop):
    """
    Do a crossover to all element of a population except the first one which
    is kept identical for the rest crossover the i and i+1
    input : population
    output : population
    """
    
    crossed_pop=[]
    crossed_pop.append(best_pop[0]) #garde le meilleur élément
    for i in range(1,len(best_pop)):
        crossed_pop.append(crossover(best_pop[i-1],best_pop[i]))
    return crossed_pop






"""----------------------------#5 New gen ---------------------------------"""
def newGen(offspring):
    """
    Create a new population of size sizePop based on a population of size 
    sizePop/2 and complete which random route
    input : offspring = population (sizePop/2)
    output : return a population of size=sizePop
    """
    new_pop=[]
    for i in range(0,len(offspring)):
        new_pop.append(offspring[i])
        
    for i in range(len(offspring), sizePop):
        new_pop.append(random.sample(cityList, numberOfCities))
    
    return new_pop






""""-------------------------------Main----------------------------------- """

def main():
    
    #Generate a list of City 
    for i in range(0,numberOfCities):
        cityList.append(City(x=int(random.randrange(0, 100, step=1)),
                             y=int(random.randrange(0, 100, step=1))))
        
    
    listOfBestDistance =[]
    pop = createPopulation(sizePop, cityList)
    initialPopulation = copy.deepcopy(pop)
    
      
    for i in range (0,50):
        sortedPop = sortPopulation(pop)                           #Pop trié 
        listOfBestDistance.append(computeDistanceRoute(sortedPop[0]))
        selectedPop = selectionBestRoutes(sortedPop)        #Garde que les meilleurs
        crossedPop = populationCrossover(selectedPop)      #Crossover parmi les meilleurs
        pop = newGen(crossedPop)                           #Recré un population
    
    """Plot intial route"""    
    xCitiesInit=[]
    yCitiesInit=[]
    for i in range(0,len(initialPopulation[0])):
        xCitiesInit.append(initialPopulation[0][i].getX())
        
    for i in range(0,len(initialPopulation[0])):
        yCitiesInit.append(initialPopulation[0][i].getY())
        
    
    plt.figure(1)
    plt.plot(xCitiesInit,yCitiesInit)
    plt.plot(xCitiesInit,yCitiesInit,'ro')
    
    
    """Plot final route"""
    xCities=[]
    yCities=[]
    
    for i in range(0,len(pop[0])):
        xCities.append(pop[0][i].getX())
        
    for i in range(0,len(pop[0])):
        yCities.append(pop[0][i].getY())
        
    plt.figure(2)
    plt.plot(xCities,yCities)
    plt.plot(xCities,yCities,'ro')
    
    """Plot evolution of distances"""
    plt.figure(3)
    plt.plot(listOfBestDistance)
    
    plt.show()
    
if __name__ == "__main__":
    main()