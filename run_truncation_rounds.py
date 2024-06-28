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





   
filename_variables = "config_variables.pickle"
filename_phys = "physics_variables.pickle"


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-path", type=str)
    parser.add_argument("-argsstr", type=str)
    args = parser.parse_args()
    

    # loading config parameters
    with open(args.path+'/' +filename_variables, 'rb') as file:
        config_dict = pickle.load(file)
    for key in config_dict.keys():
        locals()[key] = config_dict[key]
        

    which_truncation = -1
    for i in range(n_truncations+1):
        
        which_truncation+=1
        
        config_dict['which_truncation'] = which_truncation
        
        with open(results_dir+'/'+filename_variables,'wb') as file:
            pickle.dump(config_dict, file)
        
        truncation_result = subprocess.run([args.path+'/run_truncation_round.sh',args.argsstr, str(i)], capture_output = False, text=True)
        
        
    
        
    

        
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            




