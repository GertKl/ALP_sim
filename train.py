#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:41:58 2024

@author: gert
"""



import numpy as np
import random
import time

import sys
import os

import swyft
import torch
from pytorch_lightning.loggers import WandbLogger
import wandb

import matplotlib
import matplotlib.pyplot as plt

import pickle

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




if __name__ == "__main__":
    
    T = Timer()
    
    # files_name = os.getcwd().split('/')[-1]
    # parent_dir = (os.getcwd()+"/").split('/notebooks/', 1)[0]
    # start_dir = os.getcwd()
    # files_dir = os.getcwd()
    # store_name = files_name.split("_files")[0] + "_store"
    # store_dir = os.getcwd() + "/" + files_name + "/" + store_name
    # sys.path.append(files_dir)
    
    from ALP_quick_sim import ALP_sim
    from network import Network
    
    
    with open('config_objects.pickle', 'rb') as file:
        config_objects = pickle.load(file)
    for key in config_objects.keys():
        locals()[key] = config_objects[key]
    
    with open('obs_objects.pickle', 'rb') as file:
        obs_objects = pickle.load(file) 
    for key in obs_objects.keys():
        locals()[key] = obs_objects[key]
    
    with open('sim_objects.pickle', 'rb') as file:
        sim_objects = pickle.load(file)
    for key in sim_objects.keys():
        locals()[key] = sim_objects[key]
        
        
    store = swyft.ZarrStore(slurm_dir_on_cluster + "/" + files_name + "/" + store_name + "/store")
    samples = store.get_sample_store()
    
    print("Store length: " + str(len(samples)))
          
    print("Infs in store: " + str(np.where(np.isinf(samples['data']))))
    
    print("nans in store: " + str(np.where(np.isinf(samples['data']))))
    

    network = Network(nbins=A.nbins, marginals=marginals, param_names=A.param_names)
    
    wandb_logger = WandbLogger(log_model='all')
    
    DEVICE = 'cpu' if not gpus else 'cuda'
    
    trainer = swyft.SwyftTrainer(
        accelerator = DEVICE, precision = 64, logger=wandb_logger #, min_epochs =25, 
    )
    
    dm = swyft.SwyftDataModule(samples)
    
    T.start()
    trainer.fit(network, dm)
    T.stop("Time spent training")
    
    wandb.finish()
    
    
    torch.save(network.state_dict(),"trained_network.pt")
    
    torch.save(network.state_dict(),'trained_network_'+files_name.split('_files')[0]+'_slurm.pt')
    
    
    prior_samples = sim.sample(100_000, targets=['params'])
    
    
    for j in range(len(truths)):
        logratios = trainer.infer(
                                network,
                                observations[j],
                                prior_samples
                                )
        
        fig = swyft.plot_posterior(logratios, A.param_names[0], truth={A.param_names[i]:truths[j][i] for i in range(1)},color_truth=colors[j])
        plt.savefig('posterior_'+str(truths[j]))




