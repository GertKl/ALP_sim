#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:13:01 2024

@author: gert
"""


import swyft
from ALP_quick_sim import ALP_sim
from alp_swyft_simulator import ALP_SWYFT_Simulator
import pickle
import argparse
import numpy as np
import time
import os
from multiprocessing import Process
import copy
import shutil
import random


class Timer():
    
    def __init__(self):
        self.start_time = None
        self.stop_time = None
    
    def start(self):
        self.start_time = time.time()
        
    def stop(self,what="Elapsed time"):
        self.stop_time = time.time()
        h,m,s = Timer.process_time(self.stop_time-self.start_time)
        print(what + ": "+str(h)+" h, "+str(m)+" min, "+str(s)+" sec.")
    
    @staticmethod
    def process_time(s):
        h = int(s/3600)
        m = int((s-3600*h)/60)
        s = int(s-3600*h-60*m)
        return h, m, s


filename_variables = "config_variables.pickle"
filename_phys = "physics_variables.pickle"
filename_truncation_record = "truncation_record.pickle"

T = Timer()

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
        
    # loading information on previous truncations
    with open(args.path+'/' +filename_truncation_record, 'rb') as file:
        truncation_dict = pickle.load(file)
    for key in truncation_dict.keys():
        locals()[key] = truncation_dict[key]
    
    
    bounds_explim = copy.copy(bounds_rounds[which_grid_point][-1])
    prior_funcs_explim = copy.copy(prior_funcs)
    for poi in POI_indices: 
        bounds_explim[poi] = [obs_params[poi],obs_params[poi]]
        prior_funcs_explim[poi] = 'U'
    

    sim = ALP_SWYFT_Simulator(A, bounds_rounds[which_grid_point][-1], prior_funcs)
    sim_explim = ALP_SWYFT_Simulator(A, bounds_explim, prior_funcs_explim)
    
    
    if isinstance(n_sim_train,int):
        n_sim_round = copy.copy(n_sim_train)+n_sim_coverage
    else:
        n_sim_round = copy.copy(n_sim_train[min(which_truncation,len(n_sim_train)-1)])+n_sim_coverage
    
    n_sim_prior = n_prior_samples

    chunk_size = n_sim_round/n_jobs_sim
    n_chunks = 1
    while chunk_size > 1117:
        chunk_size = chunk_size/2
        n_chunks *= 2
    chunk_size = int(np.ceil(chunk_size))
    
    chunk_size_explim = n_sim_explim/n_jobs_sim
    n_chunks_explim = 1
    while chunk_size_explim > 1117:
        chunk_size_explim = chunk_size_explim/2
        n_chunks_explim *= 2
    chunk_size_explim = int(np.ceil(chunk_size_explim))
    
    chunk_size_prior = n_sim_prior/n_jobs_sim
    n_chunks_prior = 1
    while chunk_size_prior> 1117:
        chunk_size_prior = chunk_size_prior/2
        n_chunks_prior *= 2
    chunk_size_prior = int(np.ceil(chunk_size_prior))
    
    
    if which_truncation > 0:
        grid_point_str = "_gridpoint_"+str(which_grid_point)
        truncation_round_str = "_round_" + str(which_truncation)
        store_path = args.path + "/sim_output/store/" + store_name + truncation_round_str + grid_point_str
        store_explim_path = args.path + "/sim_output/store/" + store_name + "_explim" + truncation_round_str + grid_point_str
        store_prior_path = args.path + "/sim_output/store/" + store_name + "_prior" + truncation_round_str + grid_point_str
        if os.path.exists(store_path) and not use_old_truncations:
            shutil.rmtree(store_path)
            shutil.rmtree(store_explim_path)
            shutil.rmtree(store_prior_path)
    else:
        store_path = args.path + "/sim_output/store/" + store_name
        store_explim_path = args.path + "/sim_output/store/" + store_name + "_explim"
        store_prior_path = args.path + "/sim_output/store/" + store_name + "_prior"
    
    
    store = swyft.ZarrStore(store_path)
    sands = sim.get_shapes_and_dtypes()
    if len(store) == 0:
        store.init(
        N = n_sim_round,
        chunk_size=chunk_size,
        shapes=sands[0],
        dtypes=sands[1],
        )
    
    store_explim = swyft.ZarrStore(store_explim_path)
    if len(store_explim) == 0:
        store_explim.init(
        N = n_sim_explim,
        chunk_size=chunk_size_explim,
        shapes=sands[0],
        dtypes=sands[1],
        )
        
    store_prior = swyft.ZarrStore(store_prior_path)
    if len(store_prior) == 0:
        store_prior.init(
        N = n_sim_prior,
        chunk_size=chunk_size_prior,
        shapes={k:sands[0][k] for k in list(sands[0])[:1]},
        dtypes={k:sands[1][k] for k in list(sands[1])[:1]},
        )
        
        
    print()
    print("Unfinished simulations: " + str(store.sims_required+store_explim.sims_required))
    
    if on_cluster in ["fox"]:
        np.random.seed(None)
        random.seed(None)
        store.simulate(sim, batch_size=chunk_size, max_sims=chunk_size*n_chunks)
        store_explim.simulate(sim_explim, batch_size=chunk_size_explim, max_sims=chunk_size_explim*n_chunks_explim)
        store_prior.simulate(sim_explim, batch_size=chunk_size_explim, max_sims=chunk_size_explim*n_chunks_explim,targets=['params'])
    
    elif on_cluster in ["hepp", "local"]:
        
        
        processes = list(np.zeros(n_jobs_sim))
        
        def run_simulations(store_path, sim_obj,batch_size,n_chunks,print_progress,targets):
            np.random.seed(None)
            random.seed(None)
            sim_obj_loc = copy.deepcopy(sim_obj)
            store_obj = swyft.ZarrStore(store_path)  
            store_obj.simulate(sim_obj_loc, batch_size=batch_size, max_sims=batch_size*n_chunks,progress_bar=print_progress,targets=targets)
            
        
        T.start()
        print()
        print("Simulating training- and coverage samples. ")
        print("Running " +str(n_jobs_sim)+ " parallel simulation jobs, with "+str(n_chunks*chunk_size)+" simulations each, split into (at most) "+str(n_chunks)+" chunks of "+str(chunk_size)+".")
        for pi in range(int(n_jobs_sim)):
            sim_obj = copy.deepcopy(sim)
            processes[pi] = Process(target=run_simulations,args=(store_path,sim_obj,chunk_size,n_chunks,not int(n_jobs_sim-pi-1),[]))
            processes[pi].start()
        
        for pi in range(n_jobs_sim):
            processes[pi].join()
        
        if n_sim_explim:
            print()
            print("Simulating samples for expected limits. ")
            print("Running " +str(n_jobs_sim)+ " parallel simulation jobs, with "+str(n_chunks_explim*chunk_size_explim)+" simulations each, split into (at most) "+str(n_chunks_explim)+" chunks of "+str(chunk_size_explim)+".")
            for pi in range(int(n_jobs_sim)):
                sim_obj = copy.deepcopy(sim_explim)
                processes[pi] = Process(target=run_simulations,args=(store_explim_path,sim_obj,chunk_size_explim,n_chunks_explim,not int(n_jobs_sim-pi-1),[]))
                processes[pi].start()
            
            for pi in range(n_jobs_sim):
                processes[pi].join()
        
        if n_prior_samples:
            print()
            print("Drawing prior samples. ")
            print("Running " +str(n_jobs_sim)+ " parallel draws, with "+str(n_chunks_prior*chunk_size_prior)+" draws each, split into (at most) "+str(n_chunks_prior)+" chunks of "+str(chunk_size_prior)+".")
            for pi in range(int(n_jobs_sim)):
                sim_obj = copy.deepcopy(sim)
                processes[pi] = Process(target=run_simulations,args=(store_prior_path,sim_obj,chunk_size_prior,n_chunks_prior,not int(n_jobs_sim-pi-1),['params']))
                processes[pi].start()
            
            for pi in range(n_jobs_sim):
                processes[pi].join()
              
        print()  
        T.stop("Time spent simulating")
        print("Length of store: " + str(len(store)))
        print("Unfinished simulations: " + str(store.sims_required)) 
    else:
        raise ValueError("Cluster \""+on_cluster+"\" not recognized")



            































