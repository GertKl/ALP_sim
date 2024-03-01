#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:18:47 2024

@author: gert
"""


import argparse
import re
import numpy as np
import pickle

print("Importing ALP_sim... ", end="", flush=True)
from ALP_quick_sim import ALP_sim
print("done.")
    
if __name__ == "__main__":
   
    # Parsing results directory path and arguments-string.  
    
    print("Parsing arguments-string to set_parameters.py... ", end="", flush=True)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-args", type=str)
    args = parser.parse_args()
    print("done.")
    
    
    # Making basic dict of configuration arguments from raw arguments string. 

    print("Converting arguments-string to variables... ", end="", flush=True)
    config_dict = {}
    config_list = args.args.split(";")

    for item in config_list:
        
        # Converting argument strings into lists.
        
        config_argument_name_and_value = re.sub(r'\s','',item).split("=")
     
        is_item_1d=True
        is_row_1d=True
        
        if len(config_argument_name_and_value)>1:
            
            if len(config_argument_name_and_value)>3:
                raise ValueError('Argument name and value defined incorrectly \
                                 in configuration script: ' + item)
            
            config_dict_item = config_argument_name_and_value[1].split(",")
            if len(config_dict_item)>1: is_item_1d=False
            
            for row, item_row in enumerate(config_dict_item):
                
                config_dict_item_row = item_row.split("|")
                if len(config_dict_item_row)>1: is_row_1d=False
                config_dict_item[row] = config_dict_item_row[0] if is_row_1d else config_dict_item_row
                
            config_dict_item = config_dict_item[0] if is_item_1d else config_dict_item
                
            # Converting elements of argument into specified type, if given. 
            if len(config_argument_name_and_value)==3:
                config_dict_item = np.array(config_dict_item).astype(config_argument_name_and_value[2])
                if is_item_1d==0 or is_row_1d==0:
                    config_dict_item = list(config_dict_item)
                else:
                    config_dict_item = getattr(__builtins__, config_argument_name_and_value[2])(config_dict_item)
            

            config_dict[config_argument_name_and_value[0]] = config_dict_item
            
            # print( {config_argument_name_and_value[0]:config_dict[config_argument_name_and_value[0]]} )
            # print()
    print("done.")
    
    
    
    # Formatting selected arguments
    
    
    # Formatting model parameter information. 
    
    model_params = np.array(config_dict['model_params'])
    
    sim_params = list(model_params[:,0])
    bounds = []
    for isp, sim_param in enumerate(sim_params):
        sim_params[isp] = eval(sim_param.replace(':',','))
        if isinstance(sim_params[isp], list):
            bounds.append(sim_params[isp])
        elif not isinstance(sim_params[isp], (int, float)):
            raise TypeError("A parameter argument (" + str(sim_params[isp]) + ") was evaluated to an unexpected \
                            type ("+str(type(sim_params[isp]))+")")
        
    obs_params = list(model_params[:,1].astype(float))
    null_params = list(model_params[:,2].astype(float))
    is_log = list(model_params[:,3].astype(int))
    all_param_names = list(model_params[:,4].astype(str))
    all_param_units = list(model_params[:,5].astype(str))
    
    param_names = []
    param_units = []
    for j, val_j in enumerate(sim_params):
        if isinstance(val_j,list): 
            param_names.append(all_param_names[j])
            param_units.append(all_param_units[j])
    
    
    config_dict['sim_params'] = sim_params
    config_dict['obs_params'] = obs_params
    config_dict['param_names'] = param_names
    config_dict['param_units'] = param_units
    config_dict['bounds'] = bounds

    

    # Creating and formatting ALP_sim object. 
    
    
    print("Initializing new ALP_sim object... ", end='', flush=True)
    A = ALP_sim(set_null=0, set_obs=0)
    print("done.")
    print()

    A.configure_model(
        model=config_dict['model'],
        noise="poisson",
        log_params=is_log,
        null_params=null_params,
        param_names=param_names,
        param_units=param_units,
        ALP_seed=eval(config_dict['ALP_seed']),
        floor=float(config_dict['floor_exp']),
        floor_obs=float(config_dict['floor_obs']), # not reflected in training set of all_larger_bounds
        logcounts=True,
        residuals=True
    )
    
    print("Configuring observations geometry... ", end='', flush=True)
    A.configure_obs(
        nbins = int(config_dict['nbins']),
        nbins_etrue = int(config_dict['nbins_etrue']),
        emin = float(config_dict['emin']),
        emax = float(config_dict['emax']),
        livetime = float(config_dict['livetime']),
        irf_file = config_dict['IRF_file'],
    )
    print("done.")
    
    A.generate_null()
    print()
    
    config_dict['A'] = A
 
    
    
    
    # Printing and saving variables to files
    
    results_dir = config_dict['results_dir']
    
    file_control_name = "check_variables.txt"
    file_control = open(results_dir + "/" + file_control_name, "w")
    for key in config_dict.keys():    
        file_control.write(str(key) +" : " + str(config_dict[key])+"\n")
    file_control.close()
    print("Printed configuration variables to check_variables.txt")
    
    with open(results_dir+'/config_variables.pickle','wb') as file:
        pickle.dump(config_dict, file)
    print("Saved all configuration variables to "+file_control_name)
    print()
    

    

    
    




