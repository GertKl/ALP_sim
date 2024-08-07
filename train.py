#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:41:58 2024

@author: gert
"""



import swyft
import numpy as np
from ALP_quick_sim import ALP_sim
from alp_swyft_simulator import ALP_SWYFT_Simulator
import pickle
import argparse
import importlib
import time
import datetime
import itertools
import copy
import os

import torch
torch.set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy('file_system')
print('set matmul precision') 
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
import wandb



def convert_pair_to_index(pair,n_indices):
    pair = sorted(pair)
    return int((pair[0]+1)*(n_indices-1+n_indices-pair[0]-1)/2 - n_indices + pair[1])


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
filename_true_obs = "true_obs.pickle"
filename_truncation_record = "truncation_record.pickle"
filename_explim_predictions = "explim_predictions.pickle"


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
        config_phys_dict = pickle.load(file)
    for key in config_phys_dict.keys():
        locals()[key] = config_phys_dict[key]
    
    # loading information on previous truncations
    with open(args.path+'/' +filename_truncation_record, 'rb') as file:
        truncation_dict = pickle.load(file)
    for key in truncation_dict.keys():
        locals()[key] = truncation_dict[key]
        
    # loading mock true observation
    with open(args.path+'/' +filename_true_obs, 'rb') as file:
        true_obs_dict = pickle.load(file)
    for key in true_obs_dict.keys():
        locals()[key] = true_obs_dict[key]
    
    network_file = results_dir+"/train_output/net/trained_network"+"_round_"+str(which_truncation)+"_gridpoint_"+str(which_grid_point)+".pt"
    
    sim = ALP_SWYFT_Simulator(A, bounds_rounds[which_grid_point][which_truncation], prior_funcs)
    
    if isinstance(n_sim_train,int):
        n_sim_round = copy.copy(n_sim_train)
    else:
        n_sim_round = copy.copy(n_sim_train[min(which_truncation,len(n_sim_train)-1)])
    
    # if which_truncation == n_truncations:
    #     n_sim_round += n_sim_coverage
    
    
    T = Timer()
    
    grid_point_str = "_gridpoint_"+str(which_grid_point) if which_truncation > 0 else ""
    truncation_round_str = "_round_" + str(which_truncation) if which_truncation > 0 else ""
    store_path = args.path + "/sim_output/store/" + store_name + truncation_round_str + grid_point_str
    store = swyft.ZarrStore(store_path)
    if len(store) == 0:
        raise ValueError("Store is empty!")
    all_samples = store.get_sample_store()
    samples = all_samples[:n_sim_round]
    
    print("Training on " + str(len(samples)) + " samples" )
    print("Store length: " + str(len(samples)))

    module_name = 'architecture'
    spec = importlib.util.spec_from_file_location(module_name, results_dir+"/train_output/net/network.py")
    net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(net)
    
    count = -1
    for combo in itertools.product(*hyperparams.values()):
        count +=1
        if count == which_grid_point:
            hyperparams_point = {}
            for i, key in enumerate(hyperparams.keys()):
                hyperparams_point[key]=combo[i]
            break
        

    if restricted_posterior == 1 or len(POI_indices) == 1:
        network = net.Network1D(nbins=A.nbins, marginals=POI_indices, param_names=A.param_names)
    elif restricted_posterior == 2:
        network = net.Network2D(nbins=A.nbins, marginals=POI_indices, param_names=A.param_names)
    else:
        network = net.NetworkCorner(A.nbins, POI_indices, A.param_names, **hyperparams_point)


    train_net = 1
    if os.path.exists(use_old_net):
        network.load_state_dict(torch.load(use_old_net))
        print('Loaded old network state')
        if not continue_training: train_net=0
    else:
        if eval(use_old_net):
            if os.path.exists(network_file):
                network.load_state_dict(torch.load(network_file))
                print('Loaded old network state')
                if not continue_training: train_net=0
                
            

    if train_net: wandb_logger = WandbLogger(log_model='all')
    
    DEVICE = 'cpu' if not gpus else 'cuda'
    
    trainer = swyft.SwyftTrainer(
        accelerator = DEVICE, precision = 64, logger=wandb_logger if train_net else None, 
        enable_progress_bar=not on_cluster, max_epochs=max_epochs, 
        log_every_n_steps=50,
    )
    
    num_workers = 4 if on_cluster in ["hepp"] and gpus else 2
    

    dm = swyft.SwyftDataModule(samples, num_workers = num_workers, batch_size=int(train_batch_size_1d), 
                           on_after_load_sample = sim.get_resampler(targets = ['data','power']),)
    
    
    print()
    print("Current time and date: " + str(datetime.datetime.now()).split(".")[0])
    print()
    
    T.start()
    
    if train_net:
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
        torch.save(network.state_dict(), network_file)
        print("Network state dict saved as "+ network_file)
    except Exception as Err2:
        print(Err2)
    print()
        


    # Truncating priors based on temporary posterior
    print("Truncating... \n", end="", flush=True)
    store_prior_path = args.path + "/sim_output/store/" + store_name + "_prior" + truncation_round_str + grid_point_str
    store_prior = swyft.ZarrStore(store_prior_path)
    if len(store_prior) == 0:
        raise ValueError("Prior store is empty!")
    prior_samples = store_prior.get_sample_store()
    
    logratios_round = trainer.infer(network, true_obs, prior_samples)
    if len(logratios_rounds[which_grid_point]) < which_truncation+1:
        logratios_rounds[which_grid_point].append(logratios_round)
    else:
        logratios_rounds[which_grid_point][which_truncation] = logratios_round
        
    
    bounds_truncated = swyft.lightning.bounds.get_rect_bounds(logratios_rounds[which_grid_point][which_truncation][0], threshold=1e-6).bounds[:,0,:]
    bounds_round = np.array(bounds).copy()
    for bi in range(len(bounds_truncated)): bounds_round[POI_indices[bi]] = np.array(bounds_truncated[bi])
    if len(bounds_rounds[which_grid_point]) < which_truncation+2:    
        bounds_rounds[which_grid_point].append(np.array(bounds_round))
    else:
        bounds_rounds[which_grid_point][which_truncation+1] = bounds_round
    # print(bounds_rounds)

    truncation_dict['logratios_rounds'] = logratios_rounds
    truncation_dict['bounds_rounds'] = bounds_rounds
    # print('yoop')
    # print(len(truncation_dict['logratios_rounds'][which_grid_point]))
    with open(results_dir+'/'+filename_truncation_record,'wb') as file:
        pickle.dump(truncation_dict, file)
    print("Done truncating.")
    print("Parameter space reductions compared to previous round (to original bounds):")
    print()
    for poi in POI_indices:
        reduction = 1-(bounds_rounds[which_grid_point][which_truncation+1][poi][1]-bounds_rounds[which_grid_point][which_truncation+1][poi][0])/(bounds_rounds[which_grid_point][which_truncation][poi][1]-bounds_rounds[which_grid_point][which_truncation][poi][0])
        reduction_orig = 1-(bounds_rounds[which_grid_point][which_truncation+1][poi][1]-bounds_rounds[which_grid_point][which_truncation+1][poi][0])/(bounds_rounds[which_grid_point][0][poi][1]-bounds_rounds[which_grid_point][0][poi][0])
        print("  "+A.param_names[poi]+f":{reduction*100: .0f}% ({reduction_orig*100: .0f}% ) ")
    print()
    
    
    # Computing expected limit predictions: 
    if which_truncation == n_truncations and n_sim_explim:
        
        print('Making predictions for expected limits...')
        print()
        
        store_explim_path = args.path + "/sim_output/store/" + store_name + "_explim" + truncation_round_str + grid_point_str
        store_explim = swyft.ZarrStore(store_explim_path)
        if len(store_explim) == 0:
            raise ValueError("Expected limits-store is empty!")
        samples_explim = store_explim.get_sample_store()
        
        
        n_limits = len(samples_explim)
        n_prior_samples = len(prior_samples)
        
        
        batch_size = 1024
        repeat = n_prior_samples // batch_size + (n_prior_samples % batch_size > 0)
        
        
        trainer = swyft.SwyftTrainer(
            accelerator = DEVICE, precision = 64, 
            enable_progress_bar=True, max_epochs=max_epochs, 
            log_every_n_steps=50,callbacks=[TQDMProgressBar(refresh_rate=100)]
        )
        
        T.start()
        predictions = trainer.infer(
            network,
            samples_explim.get_dataloader(batch_size=1,repeat=repeat),
            prior_samples.get_dataloader(batch_size=batch_size)
        )
        T.stop('Time spent making predictions for expected limits')
        print()
        
        #reducing the size of the predictions before saving
        predictions[0] = None
        prediction_indices = []
        for pair in prediction_pairs:
            prediction_indices.append(convert_pair_to_index(pair, len(POI_indices)))
        predictions[1].logratios = predictions[1].logratios[:,prediction_indices].to(torch.float32)
        predictions[1].params = predictions[1].params[:,prediction_indices,:].to(torch.float32)
        
        
        with open(results_dir+'/'+filename_explim_predictions,'wb') as file:
            pickle.dump(predictions, file)
            
        print("Expected limit predictions saved as "+ filename_explim_predictions)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            




