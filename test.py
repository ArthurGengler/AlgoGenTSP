# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:39:46 2021

@author: Arthur Gengler
"""

numberOfCities = 15

cityList = createKnownCityList(numberOfCities)
random.shuffle(cityList) 

distanceFinal = []

for i in range(5):
    start = time.time()
    distanceFinal.append(algoGen(cityList, numberOfCities, 12, 10**5, 10**4, 0.01, "PMX", False))
    end = time.time()
    print("time",end-start)      
print(distanceFinal)

for i in range(1):
    start = time.time()
    algoGen(cityList, numberOfCities, 12, 10**5, 10**4, 0.01, "Random", False)
    end = time.time()
    print("time",end-start)        
    
for i in range(1):
    start = time.time()
    algoGen(cityList, numberOfCities, 12, 10**5, 10**4, 0.01, "2Children", False)
    end = time.time()
    print("time",end-start)   

for i in range(1):
    start = time.time()    
    algoGen(cityList, numberOfCities, 12, 10**5, 10**4, 0.01, "Double", False)
    end = time.time()
    print("time",end-start)  

for i in range(1):
    algoGen(cityList, numberOfCities, 12, 10**5, 10**4, 0.01, "Deterministic", False)

numberOfCities = 15

parent1 = createKnownCityList(numberOfCities)
random.shuffle(parent1) 

parent2 = createKnownCityList(numberOfCities)
random.shuffle(parent2)

locus1 = 1
locus2 = 5
   
while(locus1>locus2 or locus1==locus2):
    locus1 = int(random.randrange(1, len(parent1)-1, step=1))
    locus2 = int(random.randrange(1, len(parent1)-1, step=1))    
    print("fr")
doubleCrossover(r1,r2,locus1,locus2)

(o1,o2)=PMX(parent1, parent2)
print("p1",parent1)
print("p2",parent2)
print("o1",o1)
print("o2",o2)
