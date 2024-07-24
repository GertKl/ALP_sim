#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:18:47 2024

@author: gert
"""


import argparse
import re
import os
import sys
import numpy as np
import pickle
import subprocess
import importlib
from types import ModuleType
import swyft
from alp_swyft_simulator import ALP_SWYFT_Simulator
import random
import scipy.stats as sps


def save_variables(variables: dict, path: str) -> None:  
    '''
    Saves variables (stored in a dict) to a pickle file. 
    Input:
        -  path:            Path of the pickle file. 
    '''
    with open(path,'wb') as file:
        pickle.dump(variables, file)

filename_variables = "config_variables.pickle"
filename_phys = "physics_variables.pickle"
filename_control = "check_variables.txt"
filename_true_obs = "true_obs.pickle"


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
                try:
                    config_dict_item = np.array(config_dict_item).astype(config_argument_name_and_value[2])
                except Exception as Err:
                    print("Encountered problems when setting type of raw argument: " + item)
                    print()
                    raise Err
                    
                if is_item_1d==0 or is_row_1d==0:
                    config_dict_item = list(config_dict_item)
                else:
                    config_dict_item = getattr(__builtins__, config_argument_name_and_value[2])(config_dict_item)

            config_dict[config_argument_name_and_value[0]] = config_dict_item
            
    print("done.")
    print()
    
   
    # Defining central paths
    results_dir = config_dict['results_dir']
    scripts_dir = config_dict['scripts_dir']
    
    # Importing useful functions
    sys.path.append(scripts_dir)
    from utils import*
    
    # Extracting store 
    use_old_sims=config_dict['use_old_sims']
    if os.path.exists(use_old_sims):
        store_name = use_old_sims.split("/")[-1]
    else:
        store_name = 'store'
    config_dict['store_name'] = store_name
    

    # Formatting hyperparameters
    if not isinstance(config_dict['hyperparams'],str):
        hyperparams = ",".join(config_dict['hyperparams']).split('--')[1:]
    else:
        hyperparams = config_dict['hyperparams'].split('--')[1:]
    hyperparams_dict = {}
    for hyperparam in hyperparams:
        hyperparam_list = hyperparam.split(':')
        hyperparam_info = hyperparam_list[0].split("(")
        hyperparam_name = hyperparam_info[0]
        hyperparam_type = hyperparam_info[1].split(")")[0]
        hyperparam_vals = np.array(hyperparam_list[1].split(","))
        if not hyperparam_type == "":
            if len(hyperparam_vals)>1:
                hyperparam_vals = hyperparam_vals.astype(hyperparam_type)
            else:
                hyperparam_vals = [getattr(__builtins__, hyperparam_type)(hyperparam_vals)]
        hyperparams_dict[hyperparam_name] = hyperparam_vals
    config_dict['hyperparams'] = hyperparams_dict
    
    
    
    # Printing variables to file for double-checking 
    file_control = open(results_dir + "/" + filename_control, "w")
    for key in config_dict.keys():    
        file_control.write(str(key) +" : " + str(config_dict[key])+"\n")
    file_control.close()
    print("Printed configuration variables to "+results_dir + "/" + filename_control)
    print()
    
    # Saving config variabls to file
    save_variables(config_dict, results_dir+'/'+filename_variables)
    print("Saved all configuration variables to "+results_dir + "/" + filename_variables)
    print()
    
    
    # Formatting selected physics arguments
    
    if config_dict['update_physics']: 
        config_phys_dict = {}
        
        # print(config_dict['model_params'])
        
        # Formatting model parameter information. 
        model_params = np.array( [ innerel.split('|') for innerel in ','.join( ['|'.join(el) if isinstance(el,list) else el for el in config_dict['model_params']] ).split('+') ]  )
        
        # print(model_params)
        
        sim_params = list(model_params[:,0])
        bounds = []
        
        for isp, sim_param in enumerate(sim_params):
            sim_params[isp] = eval(sim_param.replace(':',','))
            if isinstance(sim_params[isp], list):
                bounds.append(sim_params[isp])
            elif not isinstance(sim_params[isp], (int, float)):
                raise TypeError("A parameter argument (" + str(sim_params[isp]) + ") was evaluated to an unexpected \
                                type ("+str(type(sim_params[isp]))+")")
                                
        obs_params_full = list(model_params[:,1].astype(float))
        null_params_full = list(model_params[:,2].astype(float))
        is_log = list(model_params[:,3].astype(int))
        all_prior_funcs = list(model_params[:,4].astype(str))
        all_param_names = list(model_params[:,5].astype(str))
        all_param_units = list(model_params[:,6].astype(str))
        
        obs_params = []
        null_params = []
        param_names = []
        param_units = []
        log_params = []
        prior_funcs = []
        
        for j, val_j in enumerate(sim_params):
            if isinstance(val_j,list):
                obs_params.append(obs_params_full[j])
                null_params.append(null_params_full[j]) 
                param_names.append(all_param_names[j])
                param_units.append(all_param_units[j])
                log_params.append(is_log[j])
                prior_funcs.append(all_prior_funcs[j])
            else:
                sim_params[j] = val_j if not is_log[j] else 10**val_j

        
        config_phys_dict['sim_params'] = sim_params
        config_phys_dict['obs_params'] = obs_params
        config_phys_dict['param_names'] = param_names
        config_phys_dict['param_units'] = param_units
        config_phys_dict['log_params'] = log_params
        config_phys_dict['bounds'] = bounds
        config_phys_dict['prior_funcs'] = prior_funcs
        

            
        # Saving variables so far, in order to be able to write parametr-extension function.
        save_variables(config_phys_dict,results_dir+'/'+filename_phys)
    
        # Creating file defining parameter-extension function
        print("Writing param_function.py... ", end='', flush=True)
        config_pois_result = subprocess.run(['python', results_dir+'/config_pois.py','-path',results_dir], capture_output = True, text=True)
        print("done.")
        
        # print()
        # print(config_pois_result)
        # print()
        
        # importing parameter function
        from param_function import param_function
    
        # Creating and formatting ALP_sim object. 
        print("Importing ALP_sim... ", end="", flush=True)
        from ALP_quick_sim import ALP_sim
        print("done.")
        print("Initializing new ALP_sim object... ", end='', flush=True)
        A = ALP_sim(set_null=0, set_obs=0)
        print("done.")
        print()
    
        A.full_param_vec = param_function
    
        A.configure_model(
            model=config_dict['model'],
            noise="poisson",
            log_params=log_params,
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
        
    
        # Making sure POI_indices (marginals) is a list. 
        POI_indices=config_dict['POI_indices']
        if isinstance(POI_indices,int): POI_indices = [POI_indices]
    
        
        # Writing remaining physics variables to config_phys_dict, and deleting coresponding
        # variablses from config_dict to avoid confusion. 
        config_phys_dict['A'] = A
        config_phys_dict['POI_indices'] = POI_indices
        
        del config_dict['model_params']
        del config_dict['model']
        del config_dict['ALP_seed']
        del config_dict['floor_exp']
        del config_dict['floor_obs']
        del config_dict['nbins']
        del config_dict['nbins_etrue']
        del config_dict['emin']
        del config_dict['emax']
        del config_dict['livetime']
        del config_dict['IRF_file']
    
        
        # Saving physics variables to files
        save_variables(config_phys_dict, results_dir+'/'+filename_phys)
        print("Saved all physics-related variables to "+filename_phys)
        print()
    
    
        
        # Simulating the mock true observation, on which to base truncation
        bounds_for_true_obs = [[obs_param,obs_param] for obs_param in obs_params] 
        sim_for_true_obs = ALP_SWYFT_Simulator(A, bounds_for_true_obs)
        true_obs = sim_for_true_obs.sample(1, progress_bar=False)[0]
        true_obs_dict = dict(true_obs=true_obs)


        # Saving mock true observation to file
        save_variables(true_obs_dict, results_dir+'/'+filename_true_obs)
        print("Saved mock true observation to "+filename_true_obs)
        print()

    

    
    




