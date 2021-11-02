# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 21:40:01 2021

@author: Arthur Gengler
"""

import numpy as np
import random
import copy
import operator
import pandas as pd
import matplotlib.pyplot as plt
import time
import gzip
import tsplib95

Arthur1_25 = np.load('Arthur/SinglePoint25.npy',allow_pickle=True)
Arthur2_25 = np.load('Arthur/PMX25.npy',allow_pickle=True)
Arthur3_25 = np.load('Arthur/Breed25.npy',allow_pickle=True)
Arthur4_25 = np.load('Arthur/NewPMX25.npy',allow_pickle=True)


Internet_25 = np.load('Arthur/Internet25.npy',allow_pickle=True)
modulebuild_50 = np.load('Arthur/prebuildmodule25.npy',allow_pickle=True)



distArthur1_25=Arthur1_25[0,:]
distArthur2_25=Arthur2_25[0,:]
distArthur3_25=Arthur3_25[0,:]
distArthur4_25=Arthur4_25[0,:]
distmodulebuild_50=modulebuild_50[0,:]
distInternet_25 =Internet_25[0,:]
print(distArthur3_25)
distArthur3_25 =[4.719620806163776, 4.748024960914867, 5.127808602910909, 4.862621553945511, 4.719620806163776, 5.077385317230955, 4.719620806163776, 4.861156640648169, 4.719620806163776, 4.870116374778456, 4.788188600125276, 4.788188600125276, 4.719620806163776, 4.7881886001252765, 4.7881886001252765, 4.9062064388332045, 4.788188600125276, 4.788188600125276, 4.897305289780557, 4.963337749583405, 4.801548580816956, 4.748024960914867, 4.861156640648169, 4.855872872189074, 4.855872872189074, 4.929724434609669, 5.218512129390732, 4.719620806163776, 4.748024960914867, 4.862621553945511, 4.748024960914867, 4.870116374778455, 4.855872872189074, 4.748024960914867, 4.801548580816956, 4.870116374778455, 4.788188600125276, 4.817447318755696, 4.788188600125276, 4.719620806163776, 4.9062064388332045, 4.929724434609669, 4.748024960914868, 4.855872872189074, 4.855872872189074, 4.748024960914867, 4.788188600125276, 4.719620806163776, 4.801548580816956, 4.719620806163776, 4.788188600125276, 4.855872872189074, 4.855872872189074, 4.870116374778456, 4.855872872189074, 4.935819455030013, 4.748024960914867, 4.788188600125276, 4.719620806163776, 4.719620806163776, 4.719620806163776, 4.9062064388332045, 4.870116374778455, 4.788188600125276, 4.788188600125276, 4.855872872189074, 4.861156640648169, 4.862621553945511, 4.788188600125276, 4.870116374778456, 4.890139848498903, 4.801548580816956, 4.748024960914867, 4.7196208061637765, 4.855872872189074, 4.748024960914867, 4.855872872189074, 4.9062064388332045, 4.963337749583405, 4.935819455030013, 4.788188600125276, 4.861156640648169, 4.935819455030013, 4.862621553945511, 4.748024960914867, 4.935819455030012, 4.997408706673467, 4.981982240284881, 4.719620806163776, 4.870116374778455, 4.748024960914867, 4.861156640648169, 4.870116374778456, 4.929724434609669, 4.801548580816956, 4.870116374778456, 4.9062064388332045, 4.937800646842254, 4.719620806163776, 4.870116374778455]

data_a = [distArthur1_25, distArthur2_25, distArthur3_25,distInternet_25,distmodulebuild_50]

ticks = ['Single', 'PMX', 'OrederedModified', 'CodeFound1','CodeFound2']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure(1)

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0, sym='', widths=0.6)
set_box_color(bpl, '#636363') # colors are from http://colorbrewer2.org/


plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.tight_layout()


Arthur1_50 = np.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/Arthur/SinglePoint50.npy',allow_pickle=True)
Arthur2_50 = np.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/Arthur/PMX50.npy',allow_pickle=True)
Arthur3_50 = np.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/Arthur/Breed50.npy',allow_pickle=True)
Arthur4_50 = np.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/Arthur/NewPMX50.npy',allow_pickle=True)


Internet_50 = np.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/Arthur/Internet50.npy',allow_pickle=True)
modulebuild_50 = np.load('Arthur/prebuildmodule50.npy',allow_pickle=True)

distArthur1_50=Arthur1_50[0,:]
distArthur2_50=Arthur2_50[0,:]
distArthur3_50=Arthur3_50[0,:]
distArthur4_50=Arthur4_50[0,:]
distInternet_50 =Internet_50[0,:]
distmodulebuild_50=modulebuild_50[0,:]
data_a = [ distArthur3_50,distInternet_50,distmodulebuild_50]

ticks = ['OrederedModified','CodeFound1','CodeFound2']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure(2)

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0, sym='', widths=0.6)
set_box_color(bpl, '#636363') # colors are from http://colorbrewer2.org/

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.tight_layout()




Arthur3_100 = np.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/Arthur/newordered100.npy',allow_pickle=True)
modulebuild_100 = np.load('Arthur/prebuildmodule100.npy',allow_pickle=True)


Mat100 = np.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/Matthias/TTH2_TSP_n100_1_vACO_cycle__max_nb_rounds_1000__alpha_1__beta_2__rho_0.5__Q_100__eps_0.001.npy',allow_pickle=True)

distMat100 = Mat100[:,1]
distArthur3_100=Arthur3_100[0,:]
distmodulebuild_100=modulebuild_100[0,:]
data_a = [distArthur3_100, distmodulebuild_100]

ticks = ['OrederedModified','CodeFound2' ]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure(3)

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0, sym='', widths=0.6)
set_box_color(bpl, '#636363') # colors are from http://colorbrewer2.org/

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.tight_layout()



Mat25 = np.load('Matthias/TTH2_TSP_n25_1_vACO_cycle__max_nb_rounds_1000__alpha_1__beta_2__rho_0.5__Q_100__eps_0.001.npy',allow_pickle=True)
Matthias50 = np.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/Matthias/TTH2_TSP_n50_1_vACO_cycle__max_nb_rounds_1000__alpha_1__beta_2__rho_0.5__Q_100__eps_0.001.npy',allow_pickle=True)
Matthias100 = np.load('C:/Users/Arthur_Gengler/Documents/GitHub/AlgoGenTSP/Matthias/TTH2_TSP_n100_1_vACO_cycle__max_nb_rounds_1000__alpha_1__beta_2__rho_0.5__Q_100__eps_0.001.npy',allow_pickle=True)

print(Mat25[:,1])
data_a = [Mat25[:,1], distArthur3_25, Matthias50[:,1], distArthur3_50,Matthias100[:,1], distArthur3_100]

ticks = ['Matthias25','Arthur25','Matthias50','Arthur50','Matthias50','Arthur50' ]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure(4)
bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0, sym='', widths=0.6)
set_box_color(bpl, '#636363') # colors are from http://colorbrewer2.org/

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.tight_layout()



