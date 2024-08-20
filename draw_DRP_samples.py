#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:58:38 2024

@author: gert
"""



import numpy as np
import torch
from torch.multiprocessing import Pool


from DRP_test import draw_DRP_samples_new
import pickle

import argparse

import swyft

import itertools
import os


torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(28)

filename_variables = "config_variables.pickle"
filename_phys = "physics_variables.pickle"
filename_truncation_record = "truncation_record.pickle"



if __name__=='__main__': 
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-path", type=str)
    args = parser.parse_args()
    
    # loading config parameters
    with open(args.path+'/' +filename_variables, 'rb') as file:
        config_dict = pickle.load(file)
    for key in config_dict.keys():
        locals()[key] = config_dict[key]
        
    print("Importing ALP_sim... ", end="", flush=True)
    from ALP_quick_sim import ALP_sim
    print("done.")
    
    # loading physics parameters
    with open(args.path+'/' +filename_phys, 'rb') as file:
        config_dict = pickle.load(file)
    for key in config_dict.keys():
        locals()[key] = config_dict[key]
        
    # loading information on previous truncations
    with open(args.path+'/' +filename_truncation_record, 'rb') as file:
        truncation_dict = pickle.load(file)
    for key in truncation_dict.keys():
        locals()[key] = truncation_dict[key]
    
    
    
    DEVICE = 'cpu' if not gpus else 'cuda'




    grid_point_str = "_gridpoint_"+str(which_grid_point) if which_truncation > 0 else ""
    truncation_round_str = "_round_" + str(which_truncation) if which_truncation > 0 else ""
    store_path = args.path + "/sim_output/store/" + store_name + truncation_round_str + grid_point_str
    store = swyft.ZarrStore(store_path)
    if len(store) == 0:
        raise ValueError("Store is empty!")
        

    all_samples = store.get_sample_store()
    samples = all_samples[-n_sim_coverage:]


    print("Store length: " + str(len(samples)))


    which_net = 0

    
    # print()
    # print("Posterior draws per coverage sim: " + str(n_draws))
    # print("Coverage sims: " + str(n_coverage))
    # print("First coverage sim: " + str(n_sim_coverage_start))
    # print("Grid point (model): " + str(which_net))
    # print("Number of processes: " + str(n_processes_cov))
    # print()
    
    DRP_path = results_dir+"/train_output/net/DRP_draws.pickle"
    if os.path.exists(DRP_path):
        
        print()
        print("Draws already exist for these specifications.")
        
    else:
    
        grid_point = which_net
        
        count = 0
        for combo in itertools.product(*hyperparams.values()):
            if count == grid_point:
                hyperparams_point = {}
                for i, key in enumerate(hyperparams.keys()):
                    hyperparams_point[key]=combo[i]
            count +=1
        
        
        
        print(len(samples[:min(DRP_coverage_parameters[1],n_sim_coverage)]))
        print(DRP_coverage_parameters[0])
        
        draws1d,draws2d = draw_DRP_samples_new(
            (
                A,
                bounds_rounds[which_grid_point][which_truncation],
                prior_funcs,
                samples[:min(DRP_coverage_parameters[1],n_sim_coverage)],
                DRP_coverage_parameters[0],
                results_dir+"/train_output/net/trained_network_round_"+str(which_truncation)+"_gridpoint_"+str(grid_point)+'.pt',
                DEVICE,
                A.nbins,
                POI_indices,
                A.param_names,
                8,
                1000,
                n_jobs_sim,
                hyperparams_point,
                True,
            )
        )
        

        
        with open(DRP_path,'wb') as file:
                pickle.dump(dict(
                    draws1d=draws1d,
                    draws2d=draws2d,
                ), file)


















