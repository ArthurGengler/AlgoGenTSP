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


def createKnownCityList(numberOfCities):
    """
    Create a known list of City of size numberOfCities
    """
    cityList = []
    for i in range(0,numberOfCities):
        cityList.append(City(x= i, y= 1)) 
    return cityList




"""
def recursion1 (temp_child , firstCrossPoint , secondCrossPoint , parent1MiddleCross , parent2MiddleCross, relations) :
    child = np.array([0 for i in range(len(parent1))])
    
    for i,j in enumerate(temp_child[:firstCrossPoint]):#i=count , j=value of what is in paranthese
        c=0
        for x in relations:
            if j == x[0]:
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
            if j == x[0]:
                child[i+secondCrossPoint]=x[1]
                c=1
                break
        if c==0:
            child[i+secondCrossPoint]=j
    
    child_unique=np.unique(child)
    if len(child)>len(child_unique):
        child=recursion1(child,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross, relations)
    return(child)

def recursion2(temp_child,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross, relations):
    child = np.array([0 for i in range(len(parent1))])
    for i,j in enumerate(temp_child[:firstCrossPoint]):
        c=0
        for x in relations:
            if j == x[1]:
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
            if j == x[1]:
                child[i+secondCrossPoint]=x[0]
                c=1
                break
        if c==0:
            child[i+secondCrossPoint]=j
    child_unique=np.unique(child)
    if len(child)>len(child_unique):
        child=recursion2(child,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross, relations)
    return(child)

def createChildren(parent1,parent2):
    firstCrossPoint = 4
    secondCrossPoint = 7

    parent1MiddleCross = parent1[firstCrossPoint:secondCrossPoint]
    parent2MiddleCross = parent2[firstCrossPoint:secondCrossPoint]

    temp_child1 = parent1[:firstCrossPoint] + parent2MiddleCross + parent1[secondCrossPoint:]
    temp_child2 = parent2[:firstCrossPoint] + parent1MiddleCross + parent2[secondCrossPoint:]

    relations = []
    for i in range(len(parent1MiddleCross)):
        relations.append([parent2MiddleCross[i], parent1MiddleCross[i]])
    child1=recursion1(temp_child1,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross, relations)
    child2=recursion2(temp_child2,firstCrossPoint,secondCrossPoint,parent1MiddleCross,parent2MiddleCross, relations)
    
    return child1,child2
"""
def PMX(parent1,parent2):

    firstCrossPoint = np.random.randint(0,len(parent1)-2)
    secondCrossPoint = np.random.randint(firstCrossPoint+1,len(parent1)-1)
    print(firstCrossPoint)
    print(secondCrossPoint)
    parent1MiddleCross = parent1[firstCrossPoint:secondCrossPoint]
    parent2MiddleCross = parent2[firstCrossPoint:secondCrossPoint]

    temp_child1 = parent1[:firstCrossPoint] + parent2MiddleCross + parent1[secondCrossPoint:]
    temp_child2 = parent2[:firstCrossPoint] + parent1MiddleCross + parent2[secondCrossPoint:]

    relations = []
    for i in range(len(parent1MiddleCross)):
        relations.append([parent2MiddleCross[i], parent1MiddleCross[i]])
    #print("Middle",parent1MiddleCross)
    #print("temp",temp_child1)
    #print("relations",relations)
    
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
"""
parent1 = [1,2,3,4,5,6,7,8,9]
parent2 = [5,4,6,1,2,7,3,9,8]


firstCrossPoint = np.random.randint(0,len(parent1)-2)
secondCrossPoint = np.random.randint(firstCrossPoint+1,len(parent1)-1)

(c1,c2)=createChildren(parent1,parent2)

print("",c1)
print("",c2)

liste1=createKnownCityList(5)
liste2=copy.deepcopy(liste1)
random.shuffle(liste2) 
print(liste1)
print(liste2)

(c1,c2)=createChildrenWithCities(liste1,liste2)

print("",c1)
print("",c2)
"""