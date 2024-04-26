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
import sys
import os
import torch
from multiprocessing import Process


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
    
    
    sim = ALP_SWYFT_Simulator(A, bounds)
    
    # time.sleep(60)
    
    chunk_size = n_sim/n_jobs_sim
    n_chunks = 1
    while chunk_size > 1117:
        chunk_size = chunk_size/2
        n_chunks *= 2
    chunk_size = int(np.ceil(chunk_size))
    
    store = swyft.ZarrStore(args.path + "/sim_output/store/" + store_name)
    if len(store) == 0:
        store.init(
        N = n_sim,
        chunk_size=chunk_size,
        shapes=sim.get_shapes_and_dtypes()[0],
        dtypes=sim.get_shapes_and_dtypes()[1],
        )
    
    print()
    print("Running " +str(n_jobs_sim)+ " parallel simulation jobs, with "+str(n_chunks*chunk_size)+" simulations each, split into "+str(n_chunks)+" chunks of "+str(chunk_size)+".")
    
    if on_cluster in ["fox"]:
        store.simulate(sim, batch_size=chunk_size, max_sims=chunk_size*n_chunks)
    
    elif on_cluster in ["hepp", "local"]:
        
        print("Suppressing progress bar for all but last job.")
        
       
        processes = list(np.zeros(n_jobs_sim))
        
        def run_simulations(sim_obj,batch_size,n_chunks,print_progress):
            store.simulate(sim, batch_size=batch_size, max_sims=batch_size*n_chunks,progress_bar=print_progress)

        
        T.start()
        for pi in range(int(n_jobs_sim)):
            processes[pi] = Process(target=run_simulations,args=(sim,chunk_size,n_chunks,not (n_jobs_sim-pi-1)))
            processes[pi].start()
        
        for pi in range(n_jobs_sim):
            processes[pi].join()

        T.stop("Time spent simulating")
        print("Length of store: " + str(len(store)))
        print("Unfinished simulations: " + str(store.sims_required)) 
    else:
        raise ValueError("Cluster \""+on_cluster+"\" not recognized")








        # def run_simulations(sim_obj,batch_size,pi):
        #     store.simulate(sim, batch_size=batch_size)
        
        # for pi in range(n_jobs_sim):
        #     processes[pi] = Process(target=run_simulations,args=(sim,chunk_size,pi))
        #     processes[pi].start()

        # for pi in range(n_jobs_sim):
        #     processes[pi].join()
            































