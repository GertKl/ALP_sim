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
from ALP_quick_sim import ALP_sim
from alp_swyft_simulator import ALP_SWYFT_Simulator
import pickle
import argparse
import importlib
import time
import datetime

import torch
torch.set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy('file_system')
print('set matmul precision') 
from pytorch_lightning.loggers import WandbLogger
import wandb



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
    
    T = Timer()
    
    
    store = swyft.ZarrStore(args.path + "/sim_output/store/" + store_name)
    if len(store) == 0:
        raise ValueError("Store is empty!")
        
    all_samples = store.get_sample_store()
    samples = all_samples[:n_sim_train]
    
    
    print("Store length: " + str(len(samples)))
    print("Infs in store: " + str(np.where(np.isinf(samples['data']))))
    print("nans in store: " + str(np.where(np.isinf(samples['data']))))
    
    module_name = 'architecture'
    spec = importlib.util.spec_from_file_location(module_name, results_dir+"/train_output/net/network.py")
    net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(net)
    
    
    learning_rates = [5e-2, 5e-3]
    patiences = [5, 15]
    n_featureses = [64, 128]
    depths = [2,3]
    dropouts = [0,0.1,0.2]
    
    count = 0
    for learning_rate in learning_rates:
        for depth in depths:
            for patience_temp in patiences:
                for dropout in dropouts: 
                    for n_features in n_featureses:
                        
                        count = count +1
                        
                        print()
                        print()
                        print(count)
                        print("Learning_rate")
                        print(learning_rate)
                        print("Depth")
                        print(depth)
                        
                        patience = patience_temp if n_features==64 else int(patience_temp*4)
                        
                        print("Patience")
                        print(patience)
                        print("Dropout")
                        print(dropout)
                        print("n_features")
                        print(n_features)
                        
                        train_batch_size = train_batch_size_1d if n_features==64 else int(train_batch_size_1d/4)
                        
                        print("train_batch_size")
                        print(train_batch_size_1d)
    
    
                        if restricted_posterior == 1 or len(POI_indices) == 1:
                            network = net.Network1D(nbins=A.nbins, marginals=POI_indices, param_names=A.param_names)
                        elif restricted_posterior == 2:
                            network = net.Network2D(nbins=A.nbins, marginals=POI_indices, param_names=A.param_names)
                        else:
                            # network = net.NetworkCorner(nbins=A.nbins, marginals=POI_indices, param_names=A.param_names)
                            network = net.NetworkCorner(nbins=A.nbins, marginals=POI_indices, param_names=A.param_names, 
                                                        lr=learning_rate, stopping_patience=patience,features=n_features,blocks=depth,dropout=dropout)
                        
                        wandb_logger = WandbLogger(log_model='all')
                        
                        DEVICE = 'cpu' if not gpus else 'cuda'
                        
                        trainer = swyft.SwyftTrainer(
                            accelerator = DEVICE, precision = 64, logger=wandb_logger, 
                            enable_progress_bar=not on_cluster, max_epochs=max_epochs, 
                            log_every_n_steps=50,
                        )
                        
                        num_workers = 4 if on_cluster in ["hepp"] and gpus else 2
                        dm = swyft.SwyftDataModule(samples, num_workers = num_workers, batch_size=int(train_batch_size), 
                                               on_after_load_sample = sim.get_resampler(targets = ['data']),)
                        
                          
                        print()
                        print("Current time and date: " + str(datetime.datetime.now()).split(".")[0])
                        print()
                        
                        T.start()
                        
                        try:
                            trainer.fit(network, dm)
                        except Exception as Err:
                            print()
                            print("*******TRAINING FAILED*********")
                            print(Err)
                        
                        print()
                        print("Current time and date: " + str(datetime.datetime.now()).split(".")[0])
                        print()
                        T.stop("Time spent training")
                        print()
                        
                        wandb.finish()
                                                
                        
                        try:
                            torch.save(network.state_dict(), results_dir+"/train_output/net/trained_network_"+str(count)+".pt")
                            print("Network state dict saved as "+results_dir+"/train_output/net/trained_network_"+str(count)+".pt")
                        except Exception as Err2:
                            print(Err2)
                            
                        # print("Shutting down workers...", end="", flush=True)
                        # trainer.train_dataloader._iter._shutdown_workers()
                        # print(" done.")
                            
                        # prior_samples = sim.sample(100_000, targets=['params'])
                        
                        
                        # for j in range(len(truths)):
                        #     logratios 
                        #                             network,
                        #                             observations[j],
                        #                             prior_samples
                        #                             )
                            
                        #     fig = swyft.plot_posterior(logratios, A.param_names[0], truth={A.param_names[i]:truths[j][i] for i in range(1)},color_truth=colors[j])
                        #     plt.savefig('posterior_'+str(truths[j]))
                    



