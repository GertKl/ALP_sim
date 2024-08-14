#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:41:58 2024

@author: gert
"""



import swyft
import os
import sys
import numpy as np
import pickle
import argparse
import subprocess
import itertools
import copy




   
filename_variables = "config_variables.pickle"
filename_phys = "physics_variables.pickle"
filename_truncation_record = "truncation_record.pickle"


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-path", type=str)
    parser.add_argument("-argsstr", type=str)
    parser.add_argument("-trial", type=str)
    args = parser.parse_args()
    

    # loading config parameters
    with open(args.path+'/' +filename_variables, 'rb') as file:
        config_dict = pickle.load(file)
    for key in config_dict.keys():
        locals()[key] = config_dict[key]
    
    # loading physics parameters
    with open(args.path+'/' +filename_phys, 'rb') as file:
        config_phys_dict = pickle.load(file)
    for key in config_phys_dict.keys():
        locals()[key] = config_phys_dict[key]
    
    # loading information on previous truncations, if they exist
    if os.path.exists(args.path+'/' + filename_truncation_record):
        with open(args.path+'/' +filename_truncation_record, 'rb') as file:
            truncation_dict = pickle.load(file)
        for key in truncation_dict.keys():
            locals()[key] = truncation_dict[key]
    else:
        # Initializing rounds-specific variables
        truncation_dict = {}
        truncation_dict['bounds_rounds'] = [[np.array(bounds)]]
        truncation_dict['logratios_rounds'] = [[]]

    
    which_grid_point = copy.copy(start_grid_test_at_count)-1
    
    # print("Which truncation: " + str(len( truncation_dict["logratios_rounds"][which_grid_point] )-1)) 
    
    for ci,combo in enumerate(itertools.product(*hyperparams.values())):
        
        which_grid_point += 1
        
        if which_grid_point in range(start_grid_test_at_count): continue
        
        if train and len(list(itertools.product(*hyperparams.values())))>1:
            print(flush=True)
            print(flush=True)
            print("Starting grid test, point " + str(which_grid_point), flush=True)
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", flush=True)
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", flush=True)
            print()
            for i, key in enumerate(hyperparams.keys()):
                print(key + ": " + str(combo[i]),flush=True)
            print("train_batch_size: "+str(train_batch_size_1d),flush=True)
            print()

    
        # print("Which truncation: " + str(len( truncation_dict["logratios_rounds"][which_grid_point] )-1)) 

        # Intervening to prevent re-truncating the original store for every grid point. 
        if os.path.exists(results_dir+'/'+filename_truncation_record):
            # loading truncation record
            with open(args.path+'/' +filename_truncation_record, 'rb') as file:
                truncation_dict = pickle.load(file)
            
        # print("Which truncation: " + str(len( truncation_dict["logratios_rounds"][which_grid_point] )-1)) 


        # Adding a grid point slolt in the record, if not already in place
        if len(truncation_dict['logratios_rounds']) < which_grid_point+1:  
            truncation_dict['bounds_rounds'].append([bounds,truncation_dict['bounds_rounds'][0][1]])
            truncation_dict['logratios_rounds'].append([truncation_dict['logratios_rounds'][0][0]])
            
            
        # print("Which truncation: " + str(len( truncation_dict["logratios_rounds"][which_grid_point] )-1))    
            
        which_truncation = len( truncation_dict["logratios_rounds"][which_grid_point] )-1
        if which_truncation > n_truncations:
            print()
            print('NOTE: which_truncation was set to ' + str(which_truncation) + 'which is larger than n_truncations, which is ' + str(n_truncations) + ". Setting which_truncation to " + str(n_truncations) + ".")
            print()
            which_truncation=n_truncations
        
        truncation_dict['which_grid_point'] = which_grid_point
        
        truncated_anything = False
    
        # print('truncation ' + str(which_truncation))
    
        truncations_to_do = range(which_truncation, n_truncations) if which_truncation != n_truncations else range(retrain_last_round)
        
        for i in truncations_to_do:
            
            # print('truncation ' + str(which_truncation))
            
            which_truncation = which_truncation+1 if which_truncation != n_truncations else n_truncations
            
            # print('truncation ' + str(which_truncation))
          
            truncation_dict['which_truncation'] = which_truncation
            
            with open(results_dir+'/'+filename_truncation_record,'wb') as file:
                pickle.dump(truncation_dict, file)
             
            
            truncation_result = subprocess.run([args.path+'/run_truncation_round.sh',args.argsstr, str(which_truncation), str(which_grid_point), args.trial], capture_output = False, text=True)
        
            
            with open(args.path+'/' +filename_truncation_record, 'rb') as file:
                truncation_dict = pickle.load(file)
        
        
            truncated_anything = True  
                  
        
        if not truncated_anything: print("Nothing to do \n" )
    
        
    

        
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            




