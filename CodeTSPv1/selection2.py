# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 18:54:50 2021

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


