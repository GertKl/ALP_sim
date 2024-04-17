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
from multiprocessing import Process
import numpy as np
import time


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
    
    
    chunk_size = int(n_sim/n_jobs_sim)
    while chunk_size > 1117:
        chunk_size = int(chunk_size/2)
    
    store = swyft.ZarrStore(args.path + "/sim_output/store/" + store_name)
    if len(store) == 0:
        store.init(
        N = n_sim,
        chunk_size=chunk_size,
        shapes=sim.get_shapes_and_dtypes()[0],
        dtypes=sim.get_shapes_and_dtypes()[1],
        )
    
     
    
    if on_cluster in ["fox"]:
        store.simulate(sim, batch_size=chunk_size)
    
    elif on_cluster in ["hepp", "local"]:
        
        processes = list(np.zeros(n_jobs_sim))
        run_simulations = lambda sim_obj,batch_size: store.simulate(sim, batch_size=batch_size)
        
        T.start()
        for pi in range(n_jobs_sim):
            processes[pi] = Process(target=run_simulations,args=(sim,chunk_size))
            processes[pi].start()
            # print("Started "+str(pi+1)+"/"+str(n_jobs_sim)+" jobs.")
        for pi in range(n_jobs_sim):
            processes[pi].join()
            print("Finished "+str(pi+1)+"/"+str(n_jobs_sim)+" jobs.")
        T.stop("Time spent simulating")
        # print("Length of store: " + str(len(store)))
    else:
        raise ValueError("Cluster \""+on_cluster+"\" not recognized")












