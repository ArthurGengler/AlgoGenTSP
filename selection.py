# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 18:49:48 2021

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