#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:58:38 2024

@author: gert
"""



import numpy as np
import torch
from torch.multiprocessing import Pool


from DRP_test import draw_DRP_samples
import pickle

import argparse

import swyft

import itertools
import os


torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(28)

filename_variables = "config_variables.pickle"
filename_phys = "physics_variables.pickle"



if __name__ == "__main__":
    
    
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


    store = swyft.ZarrStore(args.path + "/sim_output/store/" + store_name)
    if len(store) == 0:
        raise ValueError("Store is empty!")
        
    
    all_samples = store.get_sample_store()
    samples = all_samples[-n_sim_coverage:]
    
    
    print("Store length: " + str(len(samples)))

    # module_name = 'architecture'
    # spec = importlib.util.spec_from_file_location(module_name, results_dir+"/train_output/net/network.py")
    # net = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(net)

    DEVICE = 'cpu' if not gpus else 'cuda'
    

    if __name__=='__main__':

        for DRPCP in DRP_coverage_parameters:
            
            n_draws = int(DRPCP[0])
            n_coverage = int(DRPCP[1])
            n_sim_coverage_start = int(DRPCP[2])
            which_net = int(DRPCP[3])
            n_processes_cov = int(DRPCP[4])
            
            
            print()
            print("Posterior draws per coverage sim: " + str(n_draws))
            print("Coverage sims: " + str(n_coverage))
            print("First coverage sim: " + str(n_sim_coverage_start))
            print("Grid point (model): " + str(which_net))
            print("Number of processes: " + str(n_processes_cov))
            print()
            
            DRP_path = results_dir+"/train_output/net/DRP_draws_"+"_".join([str(i) for i in np.array(DRPCP)[:-1]])+".pickle"
            if os.path.exists(DRP_path):
                
                print()
                print("Draws already exist for these specifications.")
                
            else:
            
                grid_point = which_net
                
                count = 1
                for combo in itertools.product(*hyperparams.values()):
                    if count == grid_point:
                        hyperparams_point = {}
                        for i, key in enumerate(hyperparams.keys()):
                            hyperparams_point[key]=combo[i]
                    count +=1
                
                
                iterable = list(np.zeros(n_processes_cov))
                extra_step = 1 if not n_coverage%n_processes_cov==0 else 0
                for itb in range(len(iterable)):
                    range_start = n_sim_coverage_start + itb*(int(n_coverage/n_processes_cov)+extra_step)
                    range_stop = range_start + int(n_coverage/n_processes_cov)+extra_step
                    range_stop = range_stop if range_stop < n_sim_coverage_start+n_coverage else n_sim_coverage_start+n_coverage
                    iterable[itb] = (
                                        samples[range_start:range_stop],   
                                           n_draws,
                                           results_dir+"/train_output/net/trained_network_"+str(which_net)+".pt",
                                           DEVICE,
                                           bounds,
                                           A.nbins,
                                           POI_indices,
                                           A.param_names,
                                           8,
                                           1000,
                                           n_processes_cov,
                                           hyperparams_point,
                                       )
                
                with Pool(n_processes_cov) as pool:
                    try:
                        res = pool.map(draw_DRP_samples,iterable,chunksize = 1,)
                    except Exception as err:
                        pool.terminate()
                        print(err)
                
                pool.terminate()
                pool.close()
            
            
                draws1d = { key : np.concatenate([ res[i][0][key] for i in range(len(res))],axis=1) for key in res[0][0].keys() }
                draws2d = { key : np.concatenate([ res[i][1][key] for i in range(len(res))],axis=1) for key in res[0][1].keys() }
                
                with open(DRP_path,'wb') as file:
                        pickle.dump(dict(
                            draws1d=draws1d,
                            draws2d=draws2d,
                        ), file)


















