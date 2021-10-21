import numpy as np
import copy
import random

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


def singlePointcrossover(route1, route2, locus):
    """
    Single crossover at the locus example locus=half: 1234 and 2412 give 1243 
    input : two routes (list of City) 
    output : retrun 1 route which results from the crossover
    """
    
    new_route=[]
    for i in range(0, locus): 
        new_route.append(route1[i])
        
    for i in range(locus,len(route2)): 
        taken=False
        for j in range(0,locus):
            if route2[i].equal(route1[j]): 
                taken=True
        if (taken==False):
            new_route.append(route2[i])            
        else:
            new_route.append(City(x=-1, y=-1)) 
    for i in range(locus, len(route2)):  
        if(new_route[i].equal(City(x=-1, y=-1))):
            new_route[i]=notTaken(new_route,route2)
    return new_route


def twoPointCrossover(route1, route2, locus1, locus2):
    """
    return 1 route
    """

    new_route=[]
    for i in range(0, locus1): 
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
    for i in range(locus2,len(route1)):
        new_route.append(route1[i])
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
    child = np.array([City(0,0) for i in range(len(temp_child))])
    
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
    child = np.array([City(0,0) for i in range(len(temp_child))])
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


