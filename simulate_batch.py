#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:13:01 2024

@author: gert
"""


import swyft
import os
import sys
import numpy as np

# ALP_file_dir = os.path.dirname(os.getcwd())+"/analysis_scripts/ALP_sim"  

# if ALP_file_dir: sys.path.append(ALP_file_dir)   #!!! Change path to location of differential_counts.py and ALP_sim.py
from ALP_quick_sim import ALP_sim
from alp_swyft_simulator import ALP_SWYFT_Simulator

import scipy.stats as scist

import copy


import pickle


if __name__ == "__main__":
    
    with open('config_objects.pickle', 'rb') as file:
        config_objects = pickle.load(file)
        
    for key in config_objects.keys():
        locals()[key] = config_objects[key]
    
    
    with open('sim_objects.pickle', 'rb') as file:
        sim_objects = pickle.load(file)
        
    for key in sim_objects.keys():
        locals()[key] = sim_objects[key]
    
    
    
    
    store = swyft.ZarrStore(os.getcwd() + "/" + store_name+ "/store")
    
    
    store.simulate(sim, batch_size=simulation_batch_size)

















