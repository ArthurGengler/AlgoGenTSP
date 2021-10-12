# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 19:06:11 2021

@author: Arthur Gengler
"""

import numpy as np, random, copy, operator, pandas as pd, matplotlib.pyplot as plt
import time



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



def crossover(route1, route2, locus):
    """
    Single crossover at the locus example locus=half: 1234 and 2412 give 1243 
    input : two routes (list of City) 
    output : retrun a route which results from the crossover
    """
    
    new_route=[]
    for i in range(0, locus): #Prend la 1er partie de parent 1
        new_route.append(route1[i])
        
    for i in range(locus,len(route2)): #Prend la 2eme partie du parent 2
        taken=False
        for j in range(0,locus):
            if route2[i].equal(route1[j]): #Vérifie que aucune ville du parent 2 n'a déjà été prisent via parent 1
                taken=True
        if (taken==False):
            new_route.append(route2[i])            
        else:
            new_route.append(City(x=-1, y=-1)) #Si une ville a déjà été prise on met un -1 à la place
    for i in range(locus, len(route2)):   #S'occupe des endroit où il y a un 0 pour lui trouver une ville par laquelle on passe pas encore
        if(new_route[i].equal(City(x=-1, y=-1))):
            new_route[i]=notTaken(new_route,route2)
    return new_route


def doubleCrossover(route1, route2, locus1, locus2):
    
    new_route=[]
    for i in range(0, locus1): 
        new_route.append(route1[i])
    for i in range(locus2,len(route1)):
        new_route.append(route1[i])
    for i in range(locus1,locus2): 
        taken=False
        for j in range(0,locus1):
            if route2[i].equal(route1[j]): 
                taken=True
        for j in range(locus2,len(route1)):
            if(route2[i].equal(route1[j])):
                taken=True
        if (taken==False):
            new_route.append(route2[i])            
        else:
            new_route.append(City(x=-1, y=-1)) 
            
    for i in range(locus1, locus2):   
        if(new_route[i].equal(City(x=-1, y=-1))):
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
            if(liste2[i].equal(liste1[j])):
                taken = True
        if taken == False:
            return liste2[i]
    

def populationCrossoverDeterministic(best_pop):
    
    crossed_pop=[]
    crossed_pop.append(best_pop[0]) #garde le meilleur élément
    for i in range(1,len(best_pop)):
        locus = int(random.randrange(1, len(best_pop[i])-1, step=1))
        crossed_pop.append(crossover(best_pop[i-1], best_pop[i], locus))
 
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
        crossed_pop.append(crossover(route1, route2, locus))

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
        
        crossed_pop.append(crossover(route1, route2, locus))
        crossed_pop.append(crossover(route2, route1, locus))
        
 
    return crossed_pop

"""-----------------------------#5 Mutation------------------------------ """
def mutation(route):
    c1=int(random.randrange(0, len(route), step=1))
    c2=int(random.randrange(0, len(route), step=1))
    route[c1],route[c2] = route[c2], route[c1]
    return route

def populationMutation(crossed_pop):
    muted_pop=[]
    muted_pop.append(crossed_pop[0])
    for i in range(1,len(crossed_pop)):
        muted_pop.append(mutation(crossed_pop[i]))    
    return muted_pop


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



def plotgraph(cityList, numberOfCities, sizePop, numberMaxOfIteration, 
              nbrSameValue, mutationRate, typeOfCrossover):
    
    
    pop = createPopulation(sizePop, cityList)
    initialPopulation = copy.deepcopy(pop)
    listOfBestDistance =[computeDistanceRoute(pop[0])]
    i = 1
    count = 0
    while(count<nbrSameValue and i<numberMaxOfIteration):
        sortedPop = sortPopulation(pop)              
        listOfBestDistance.append(computeDistanceRoute(sortedPop[0]))
        selectedPop = selectionBestRoutes(sortedPop, sizePop-sizePop//3)
        
        #Crossover
        if(typeOfCrossover == "Random"):
            offspring = populationCrossoverRandom(selectedPop)
        
        if(typeOfCrossover == "Deterministic"):
            offspring = populationCrossoverDeterministic(selectedPop) 
            
        if(typeOfCrossover == "2Children"):
            offspring = populationCrossoverRandom2Children(selectedPop)
            
        #Mutation    
        if(int(random.randrange(1, 1/mutationRate, step=1))==2):
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
        
    for i in range(0,len(pop[0])):
        yCities.append(pop[0][i].getY())
        

    """
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
    """
    print(typeOfCrossover, listOfBestDistance[-1])



""""-------------------------------Main----------------------------------- """


def main():
    
    #plotgraph(cityList, numberOfCities, sizePop, numberMaxOfIteration, mutationRate)
    



        
    numberOfCities = 18
    cityList = createKnownCityList(numberOfCities)
    random.shuffle(cityList)
    """
    for i in range(5):
        plotgraph(cityList, numberOfCities, 12, 10**5, 40**4, 0.01, "Random")
        
    for i in range(5):
        plotgraph(cityList, numberOfCities, 12, 10**5, 40**4, 0.01, "Deterministic")
    """
    

    for i in range(5):
        start = time.time()
        plotgraph(cityList, numberOfCities, 12, 10**5, 40**4, 0.01, "2Children")    
    
        end = time.time()
        print(end-start)
    
if __name__ == "__main__":
    main()