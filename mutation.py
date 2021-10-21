# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 18:43:08 2021

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

def mutation(route):
    c1=int(random.randrange(0, len(route), step=1))
    c2=int(random.randrange(0, len(route), step=1))
    mutedRoute = copy.copy(route)
    mutedRoute[c1],mutedRoute[c2] = mutedRoute[c2], mutedRoute[c1]
    
    return mutedRoute

def populationMutation(crossed_pop):
    muted_pop=[]
    muted_pop.append(crossed_pop[0])
    for i in range(1,len(crossed_pop)):
        muted_pop.append(mutation(crossed_pop[i]))    
    return muted_pop