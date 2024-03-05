#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:13:01 2024

@author: gert
"""


import swyft
import os
from ALP_quick_sim import ALP_sim
from alp_swyft_simulator import ALP_SWYFT_Simulator
import pickle
import argparse

import time


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-path", type=str)
    args = parser.parse_args()
    
    with open(args.path +'/config_variables.pickle', 'rb') as file:
        config_dict = pickle.load(file)
    for key in config_dict.keys():
        locals()[key] = config_dict[key]
    
    
    sim = ALP_SWYFT_Simulator(A, bounds)
    
    # time.sleep(60)
    
    store = swyft.ZarrStore(args.path + "/sim_output/store/" + store_name)
    if len(store) == 0:
        store.init(
        N = n_sim,
        chunk_size=int(n_sim/n_jobs_sim),
        shapes=sim.get_shapes_and_dtypes()[0],
        dtypes=sim.get_shapes_and_dtypes()[1],
        )
    
    
    store.simulate(sim, batch_size=int(n_sim/n_jobs_sim))

















