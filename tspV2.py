# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:53:13 2021

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

"""---------------------------Global variales-------------------------------"""
GLOBMATRIX =[]

"""----------------------------#1 Initial pop---------------------------------"""
class City:
    """
    Class City each city has coordinate (x,y)
    """
    
    def __init__(self, x, y, node):
        self.x = x
        self.y = y
        self.node = node
        
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
    
    def getCity(self):
        return self.node 
    
    def getY(self):
        return self.y
        
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

def createKnownCityList(numberOfCities):
    """
    Create a known list of City of size numberOfCities
    """
    cityList = []
    for i in range(0,numberOfCities):
        cityList.append(City(x= i, y= 1)) 
    return cityList

def createCityListBasedOnList(ListOfCoordonne):
    """
    Create a list of City based on list of coordonne
    """
    cityList = []
    for i in range(0,len(ListOfCoordonne)):
        cityList.append(City(x= ListOfCoordonne[i][0], y=ListOfCoordonne[i][1], node=i )) 
    return cityList

def createCityListFromFile(filename):
    """
    Create a list of City based on Matthias file
    """
    data = np.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/SetCities/'+filename)
    cityList = []
    for i in range(0,len(data[0])):
            cityList.append(City(x=data[0][i],y=data[1][i], node=i))
    return cityList

def createCityListFromTSPLIB(filename):
    """
    Create a list of city based onf TSPLIB file
    """
    data = tsplib95.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/SetCities/'+filename)
    cityList=[]
    for i in range(1,len(data.node_coords)+1):
        cityList.append(City(x=data.node_coords[i][0],y=data.node_coords[i][1], node =i-1))
    return cityList

def createRoute(cityList, numberOfCities):
    """
    Creat a random route which go through each city once
    input : list of City
    output : list of City
    """
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



"""----------------------------#2 Sort pop---------------------------------"""


GLOBMATRIX =[]

def matriceDistance(cityList):
    global GLOBMATRIX
    GLOBMATRIX = np.zeros((len(cityList),len(cityList)))
    for i in range (len(GLOBMATRIX)):
        for j in range (len(GLOBMATRIX)):
            GLOBMATRIX[cityList[i].node][cityList[j].node] = cityList[i].distance(cityList[j])
            
def computeFitness(route):
    """
    Compute the distance to travel for a route
    imput : route = list of City
    output : total distance (float)
    """
    distance_tot = 0
    for i in range(len(route)-1):
        city1 = route[i].node
        city2 = route[i+1].node
        distance_tot = distance_tot + GLOBMATRIX[city1][city2]
    city1 = route[len(route)-1].node
    city2 = route[0].node
    distance_tot = distance_tot + GLOBMATRIX[city1][city2]
    return distance_tot

def sortPopulation(population):
    """
    Sort a population according to the total distance to travel
    imput : population : list of list of City
    output : population : list of list of City
    """
    return sorted(population, key = computeFitness, reverse = False)




"""--------------------#3 Selection of individual--------------------------"""

def rankSelection(sorted_population):
    n = len(sorted_population)
    probList = np.zeros(n)
    sumIndex = n*(n+1)/2  
    listIndex = np.zeros(n)
    for i in range(n):
        listIndex[i] = int(i)
    
    for i in range(1,n+1):
        probList[i-1] = (n-i+1)/sumIndex
    
    indexParent1 = np.random.choice(listIndex, 1, p=probList)
    indexParent2 = np.random.choice(listIndex, 1, p=probList)
    
    parent1 = sorted_population[int(indexParent1[0])]
    parent2 = sorted_population[int(indexParent2[0])]
    
    return parent1,parent2

"""-----------------------------#4 Crossover------------------------------ """
def cxPartialyMatched(ind1, ind2):
    """
    ind1, ind2 sont une liste d'indice chaque indice faisant référence à une city
    """
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size
    
    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    cxpoint1 = 3
    cxpoint2 = 6
    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched value
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    return ind1, ind2

def cxOrdered(ind1, ind2):
    """Executes an ordered crossover (OX) on the input
    individuals. The two individuals are modified in place. This crossover
    expects :term:`sequence` individuals of indices, the result for any other
    type of individuals is unpredictable.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    Moreover, this crossover generates holes in the input
    individuals. A hole is created when an attribute of an individual is
    between the two crossover points of the other individual. Then it rotates
    the element so that all holes are between the crossover points and fills
    them with the removed elements in order. For more details see
    [Goldberg1989]_.

    This function uses the :func:`~random.sample` function from the python base
    :mod:`random` module.

    .. [Goldberg1989] Goldberg. Genetic algorithms in search,
       optimization and machine learning. Addison Wesley, 1989
    """
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2

def fromRouteToNode(route):
    node =[]
    for i in range(len(route)):
        node.append(route[i].node)
    return node

def fromNodeToRoute(nodeList, cityList):
    route = []
    for i in range(len(nodeList)):
        for j in range(len(cityList)):
            if(cityList[j].node == nodeList[i]):
                route.append(City(cityList[j].x,cityList[j].y,nodeList[i]))
    return route

def PMXMine(parent1, parent2):
    
    nodep1 = fromRouteToNode(parent1)
    nodep2 = fromRouteToNode(parent2)
    
    nodec1, nodec2 = cxOrdered(nodep1, nodep2)
    
    child1 = fromNodeToRoute(nodec1, parent1)
    child2 = fromNodeToRoute(nodec2, parent1)
    
    return child1, child2
    #return child1
    
def oxMine(parent1,parent2):
    nodep1 = fromRouteToNode(parent1)
    nodep2 = fromRouteToNode(parent2)
    
    nodec1, nodec2 = cxPartialyMatched(nodep1, nodep2)
    
    child1 = fromNodeToRoute(nodec1, parent1)
    child2 = fromNodeToRoute(nodec2, parent1)
    
    return child1, child2    
def PMX(parent1,parent2):
    """
    return 2 routes
    """

    firstCrossPoint = np.random.randint(0,len(parent1)-2)
    secondCrossPoint = np.random.randint(firstCrossPoint+1,len(parent1)-1)

    parent1MiddleCross = parent1[firstCrossPoint:secondCrossPoint]
    parent2MiddleCross = parent2[firstCrossPoint:secondCrossPoint]

    temp_child1 = parent1[:firstCrossPoint] + parent2MiddleCross + parent1[secondCrossPoint:]
    temp_child2 = parent2[:firstCrossPoint] + parent1MiddleCross + parent2[secondCrossPoint:]

    relations = []
    for i in range(len(parent1MiddleCross)):
        relations.append([parent2MiddleCross[i], parent1MiddleCross[i]])

    child1=recursion1WithCities(temp_child1,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross, relations)
    child2=recursion2WithCities(temp_child2,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross, relations)
    
    return child1,child2



def recursion1WithCities (temp_child , firstCrossPoint , secondCrossPoint , parent1MiddleCross , parent2MiddleCross, relations) :
    child = np.array([City(0,0,-1) for i in range(len(temp_child))])
    
    for i,j in enumerate(temp_child[:firstCrossPoint]):#i=count , j=value of what is in paranthese
        c=0
        for x in relations:
            if j.equal(x[0]):
                child[i]=x[1]
                c=1
                break
        if c==0:
            child[i]=j
    
    j=0
    for i in range(firstCrossPoint,secondCrossPoint):
        child[i]=parent2MiddleCross[j]
        j+=1
    
    for i,j in enumerate(temp_child[secondCrossPoint:]):
        c=0
        for x in relations:
            if j.equal(x[0]):
                child[i+secondCrossPoint]=x[1]
                c=1
                break
        if c==0:
            child[i+secondCrossPoint]=j
    count=0
    for i in range (len(child)):
        for j in range (len(child)):
            if(i!=j):
                if(child[i].equal(child[j])):
                    count+=1
            
    if(count>1):
        child=recursion1WithCities(child,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross, relations)
    return(child)

def recursion2WithCities(temp_child,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross, relations):
    child = np.array([City(0,0,-1) for i in range(len(temp_child))])
    for i,j in enumerate(temp_child[:firstCrossPoint]):
        c=0
        for x in relations:
            if j.equal(x[1]):
                child[i]=x[0]
                c=1
                break
        if c==0:
            child[i]=j
    j=0
    for i in range(firstCrossPoint,secondCrossPoint):
        child[i]=parent1MiddleCross[j]
        j+=1

    for i,j in enumerate(temp_child[secondCrossPoint:]):
        c=0
        for x in relations:
            if j.equal(x[1]):
                child[i+secondCrossPoint]=x[0]
                c=1
                break
        if c==0:
            child[i+secondCrossPoint]=j
    count=0
    for i in range (len(child)):
        for j in range (len(child)):
            if(i!=j):
                if(child[i].equal(child[j])):
                    count+=1
            
    if(count>=1):
        child=recursion2WithCities(child,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross, relations)
    return(child)

def breed(parent1, parent2):
    """
    startGene 3
    endGene 8
    p1 [(3,2), (2,2), (1,1), (2,1), (1,2), (4,2), (3,1), (5,2), (5,1), (4,1)]
    p2 [(3,2), (2,2), (1,1), (2,1), (1,2), (4,2), (3,1), (5,2), (5,1), (4,1)]
    c [(2,1), (1,2), (4,2), (3,1), (5,2), (3,2), (2,2), (1,1), (5,1), (4,1)]
    """
    
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    
    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]
    child = childP1 + childP2
    
    return child

def singlePointcrossover(parent1, parent2):
    """
    Single crossover at the locus example locus=half: 1234 and 2412 give 1243 
    input : two routes (list of City) 
    output : retrun 1 route which results from the crossover
    """
    locus = np.random.randint(0,len(parent1)-1)
    new_route=[]
    for i in range(0, locus): 
        new_route.append(parent1[i])
        
    for i in range(locus,len(parent2)): 
        taken=False
        for j in range(0,locus):
            if parent2[i].equal(parent1[j]): 
                taken=True
        if (taken==False):
            new_route.append(parent2[i])            
        else:
            new_route.append(City(x=-1, y=-1, node=-1)) 
    for i in range(locus, len(parent2)):  
        if(new_route[i].equal(City(x=-1, y=-1,node =-1))):
            new_route[i]=notTaken(new_route,parent2)
            
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
            if(liste2[i].equal(liste1[j])):
                taken = True
        if taken == False:
            return liste2[i]

"""-----------------------------#5 Mutation------------------------------- """

def mutation(route):
    
    c1=int(random.randrange(0, len(route), step=1))
    c2=int(random.randrange(0, len(route), step=1))
    mutedRoute = copy.copy(route)
    mutedRoute[c1],mutedRoute[c2] = mutedRoute[c2], mutedRoute[c1]
    
    return mutedRoute

"""-----------------------#6 Reproduction----------------------------------"""
def populationPMXCrossover(selectionType, sorted_pop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList):
    
    crossed_pop=[]
    for i in range(nBest):
        crossed_pop.append(sorted_pop[i]) #garde le meilleur élément

    nOffspring = popSize - nBest - nRandom
    
    for i in range(0, nOffspring//2):
        #sorted_pop[:-5]
        if(selectionType=="rank"):
            (parent1,parent2) = rankSelection(sorted_pop)
        if(selectionType=="before"):
            (parent1,parent2) = rankSelection(sorted_pop[:-20])
        
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
    
    if(popSize%2 == 1):
        nRandom+=1
    for i in range(nRandom):
        crossed_pop.append(random.sample(cityList, len(cityList)))
    
    return crossed_pop

def populationPMXCrossoverMine(selectionType, sorted_pop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList):
    
    crossed_pop=[]
    for i in range(nBest):
        crossed_pop.append(sorted_pop[i]) #garde le meilleur élément

    nOffspring = popSize - nBest - nRandom
    
    for i in range(0, nOffspring//2):
        #sorted_pop[:-5]
        if(selectionType=="rank"):
            (parent1,parent2) = rankSelection(sorted_pop)
        if(selectionType=="before"):
            (parent1,parent2) = rankSelection(sorted_pop[:-20])
        
        
        if(crossoverRate>random.random()):    
            (child1,child2) = PMXMine(parent1, parent2)
        else:
            (child1,child2) = (parent1, parent2)
        if(random.random()<mutationRate):
            child1 = mutation(child1)
        if(random.random()<mutationRate):
            child2 = mutation(child2)
          
        crossed_pop.append(list(child1))
        crossed_pop.append(list(child2))
    
    if(popSize%2 == 1):
        nRandom+=1
    for i in range(nRandom):
        crossed_pop.append(random.sample(cityList, len(cityList)))
    
    return crossed_pop


def populationOXCrossoverMine(selectionType, sorted_pop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList):
    
    crossed_pop=[]
    for i in range(nBest):
        crossed_pop.append(sorted_pop[i]) #garde le meilleur élément

    nOffspring = popSize - nBest - nRandom
    
    for i in range(0, nOffspring//2):
        #sorted_pop[:-5]
        if(selectionType=="rank"):
            (parent1,parent2) = rankSelection(sorted_pop)
        if(selectionType=="before"):
            (parent1,parent2) = rankSelection(sorted_pop[:-20])
        
        if(crossoverRate>random.random()):    
            (child1,child2) = oxMine(parent1, parent2)
        else:
            (child1,child2) = (parent1, parent2)
        if(random.random()<mutationRate):
            child1 = mutation(child1)
        if(random.random()<mutationRate):
            child2 = mutation(child2)
          
        crossed_pop.append(list(child1))
        crossed_pop.append(list(child2))
    
    if(popSize%2 == 1):
        nRandom+=1
    for i in range(nRandom):
        crossed_pop.append(random.sample(cityList, len(cityList)))
    
    return crossed_pop

def populationSinglePoint(sorted_pop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList):
    
    crossed_pop=[]
    for i in range(nBest):
        crossed_pop.append(sorted_pop[i]) #garde le meilleur élément

    nOffspring = popSize - nBest - nRandom
    
    for i in range(0, nOffspring//2):
        
        (parent1,parent2) = rankSelection(sorted_pop)
        
        if(crossoverRate>random.random()):    
            child1 = singlePointcrossover(parent1, parent2)
            child2 = singlePointcrossover(parent2, parent1)
        else:
            (child1,child2) = (parent1, parent2)
           
        if(random.random()<mutationRate):
            child1 = mutation(child1)
        if(random.random()<mutationRate):
            child2 = mutation(child2)
          
        crossed_pop.append(list(child1))
        crossed_pop.append(list(child2))
    
    if(popSize%2 == 1):
        nRandom+=1
    for i in range(nRandom):
        crossed_pop.append(random.sample(cityList, len(cityList)))
    
    return crossed_pop

def populationBreed(sorted_pop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList):
    
    crossed_pop=[]
    for i in range(nBest):
        crossed_pop.append(sorted_pop[i]) #garde le meilleur élément

    nOffspring = popSize - nBest - nRandom
    
    for i in range(0, nOffspring//2):
        
        (parent1,parent2) = rankSelection(sorted_pop)
        
        if(crossoverRate>random.random()):    
            child1 = breed(parent1, parent2)
            child2 = breed(parent2, parent1)
        else:
            (child1,child2) = (parent1, parent2)
           
        if(random.random()<mutationRate):
            child1 = mutation(child1)
        if(random.random()<mutationRate):
            child2 = mutation(child2)
          
        crossed_pop.append(list(child1))
        crossed_pop.append(list(child2))
    
    if(popSize%2 == 1):
        nRandom+=1
    for i in range(nRandom):
        crossed_pop.append(random.sample(cityList, len(cityList)))
    
    return crossed_pop


"""-----------------------#Plot graph----------------------------------"""
def plotGraph(listOfBestDistance, pop):

    
    xCities=[]
    yCities=[]  
    for i in range(0,len(pop[0])):
        xCities.append(pop[0][i].getX())
    xCities.append(pop[0][0].getX())
    for i in range(0,len(pop[0])):
        yCities.append(pop[0][i].getY())
    yCities.append(pop[0][0].getY())    

    
    figure, axes = plt.subplots(1,3,figsize=(20,5))

    axes[0].plot(xCities, yCities,'ro')
    axes[0].set_xlabel('x coordinate')
    axes[0].set_ylabel('y coordinate')
    axes[0].title.set_text('Position of {} cities'.format(len(xCities)))
        
    axes[1].plot(xCities, yCities)
    axes[1].plot(xCities, yCities,'ro')
    axes[1].set_xlabel('x coordinate')
    axes[1].set_ylabel('y coordinate')
    axes[1].title.set_text("Route after {} iterations".format(len(listOfBestDistance)))
    
    axes[2].plot(listOfBestDistance)
    axes[2].set_xlabel('iteration')
    axes[2].set_ylabel('distances')   
    axes[2].title.set_text('Evolution of the distance of the best route')
    
    figure.tight_layout()


def algo(popSize, nBest, nRandom, mutationRate, crossoverRate, crossoverType, selectionType, filename, numberMaxOfIteration, nbrSameValue):
    
    if(filename[-3:] == 'npy'):
        cityList = createCityListFromFile(filename)
    
    if(filename[-3:] == 'tsp'):
        cityList = createCityListFromTSPLIB(filename)
    

    
    mutationRateInit = mutationRate
    

    
    pop = createPopulation(popSize, cityList)
    
    matriceDistance(cityList)
    
    listOfBestDistance =[computeFitness(pop[0])]
    
    i = 1
    count=0
    while(count<nbrSameValue and i<numberMaxOfIteration): 
        
        sortedPop = sortPopulation(pop)   
        listOfBestDistance.append(computeFitness(sortedPop[0]))
        
        if(crossoverType == "SinglePoint"):#d'office rank selection dans mon code
            pop = populationSinglePoint(sortedPop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList)
        if(crossoverType == "PMX"):
            pop = populationPMXCrossover(selectionType, sortedPop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList)   
        if(crossoverType == "Breed"):#d'office rank selection dans mon code
            pop = populationBreed(sortedPop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList)
        if(crossoverType == "NewPMX"):
            pop = populationPMXCrossoverMine(selectionType, sortedPop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList)
        if(crossoverType == "Ordered"):
            pop = populationOXCrossoverMine(selectionType, sortedPop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList)
        if (listOfBestDistance[i-1]<listOfBestDistance[i]):
            print("wtf")
        if(listOfBestDistance[i-1]-listOfBestDistance[i]<0.001):
            count+=1
            mutationRate = 0.2
        else:
            count=0
            mutationRate = mutationRateInit
        i+=1
    
    #print(crossoverType, ":", listOfBestDistance[-1])
    #plotGraph(listOfBestDistance, pop)
    return (listOfBestDistance[-1],pop)

"""----------------------------------------------------------------------"""



