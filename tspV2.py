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
    cityList = []
    for i in range(0,len(ListOfCoordonne)):
        cityList.append(City(x= ListOfCoordonne[i][0], y=ListOfCoordonne[i][1], node=i )) 
    return cityList

def createCityListFromFile(data):
    cityList = []
    for i in range(0,len(data[0])):
            cityList.append(City(x=data[0][i],y=data[1][i], node=i))
    return cityList

def createCityListFromTSPLIB(problem):
    
    cityList=[]
    for i in range(1,len(problem.node_coords)+1):
        cityList.append(City(x=problem.node_coords[i][0],y=problem.node_coords[i][1], node =i-1))
        
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
    
    nodec1, nodec2 = cxPartialyMatched(nodep1, nodep2)
    
    child1 = fromNodeToRoute(nodec1, parent1)
    child2 = fromNodeToRoute(nodec2, parent1)
    
    return child1, child2
    #return child1

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
"""
problem = tsplib95.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/berlin52.tsp')
cityList = createCityListFromTSPLIB(problem)
"""

def algo(crossoverType,selectionType,elitism, data):
    
    
    
    """
    ListOfCoordonne = data
    cityList = createCityListBasedOnList(ListOfCoordonne)
    
    
    
    Create from file(Matthias)
    """
    cityList = createCityListFromFile(data)
    
    """
    problem = data
    cityList = createCityListFromTSPLIB(problem)
    """
    popSize = 200
    nBest = elitism
    nRandom = 1
    mutationRate = 0.1
    crossoverRate = 0.9
    
    mutationRateInit = mutationRate
    
    numberMaxOfIteration = 5000
    nbrSameValue = 6000
    
    pop = createPopulation(popSize, cityList)
    
    matriceDistance(cityList)
    
    listOfBestDistance =[computeFitness(pop[0])]
    
    i = 1
    count=0
    while(count<nbrSameValue and i<numberMaxOfIteration): 
        
        sortedPop = sortPopulation(pop)   
        listOfBestDistance.append(computeFitness(sortedPop[0]))
        
        if(crossoverType == "SinglePoint"):
            pop = populationSinglePoint(sortedPop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList)
        if(crossoverType == "PMX"):
            pop = populationPMXCrossover(selectionType, sortedPop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList)   
        if(crossoverType == "Breed"):
            pop = populationBreed(sortedPop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList)
        if(crossoverType == "NewPMX"):
            pop = populationPMXCrossoverMine(selectionType, sortedPop, popSize, nBest, nRandom, mutationRate, crossoverRate, cityList)
        if (listOfBestDistance[i-1]<listOfBestDistance[i]):
            print("wtf")
        if(listOfBestDistance[i-1]-listOfBestDistance[i]<0.001):
            count+=1
            mutationRate = 0.2
            
        else:
            count=0
            mutationRate = mutationRateInit
        i+=1
    
    print(crossoverType, ":", listOfBestDistance[-1])
    plotGraph(listOfBestDistance, pop)
    return listOfBestDistance[-1]
"""----------------------------------------------------------------------"""
#ListOfCoordonne = [(13, 2), (1, 12), (12, 5), (19, 6), (2, 10), (15, 15), (5, 11), (17, 9),
#             (10, 18), (17, 5), (13, 12), (1, 17), (2, 6), (7, 16), (19, 2), (3, 7),
#             (10, 9), (5, 19), (1, 2), (9, 2)]


problem = tsplib95.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/att48.tsp')
"""
for i in range(5):
    algo("Breed","rank",5 , data1)
#cProfile.run('algo("SinglePoint","rank",1 , data1)')

"""

data1 = np.load('TSP_n100_1.npy')
finalDist = []
computeTime = []
for i in range(5):
    start = time.time()
    finalDist.append(algo("Breed","rank",10 , data1))
    end = time.time()
    computeTime.append(end-start)
np.save('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/Arthur/newordered100', [finalDist,computeTime])
print(finalDist)


