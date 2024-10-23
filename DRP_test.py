#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:52:46 2023

@author: gert
"""


'''

A function to perform the DPR coverage test, written by Pablo Lemos et. al. 

'''


from typing import Tuple, Union

import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse

import swyft

import os
import copy

import torch
import importlib

torch.set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy('file_system')
device_notebook = "cuda" if torch.cuda.is_available() else "cpu"

from tqdm.auto import tqdm
import logging
logging.getLogger('pytorch_lightning').setLevel(0)
logging.getLogger('lightning.pytorch').setLevel(0)

from pytorch_lightning.callbacks import TQDMProgressBar

logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(0)
import warnings
warnings.simplefilter("ignore")

from multiprocessing import Pool

from alp_swyft_simulator import ALP_SWYFT_Simulator
from ALP_quick_sim import ALP_sim

import time

# import network 




torch.set_num_threads(28)

__all__ = ("get_drp_coverage",)

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
  
stdnorm = norm()
    
def draw_prior_samples(tup):
    A = tup[0]
    bounds = tup[1]
    prior_funcs = tup[2]
    n_samples = tup[3]
    enable_progress_bar = tup[4]
    sim = ALP_SWYFT_Simulator(A,bounds,prior_funcs)
    return sim.sample(n_samples, targets=['params'],progress_bar=enable_progress_bar)


def rejection_sampling_new(A,bounds,prior_funcs,trainer,network,observations,n_samps,excess_factor=5, max_iters=1000, sampling_device='cuda',n_processes=1,progress_bar=True):


    
    successes_min=0
    iters=0
    
    samps1d = {}
    samps2d = {}
    
    # bounds_gpu = bounds
    
    batch_size = 2**15
    
    observations_unfinished = list(np.arange(len(observations),dtype=int))
    
    if n_processes <= 1: sim = ALP_SWYFT_Simulator(A,bounds,prior_funcs)
    
    T=Timer()

    while successes_min<n_samps and iters<max_iters :
           
        T.start()
        
        iters += 1
        
        if n_processes>1:
            
            iterable = [(A,
                         bounds,
                         prior_funcs,
                         int((excess_factor*n_samps)/n_processes)+(1-min(1,(excess_factor*n_samps)//n_processes))*((excess_factor*n_samps)%n_processes),
                         1-min(si,1) if progress_bar else 0,
                        ) for si in range(n_processes) ]
            
            with Pool(n_processes) as pool:
                try:
                    res = pool.map(draw_prior_samples,iterable,chunksize = 1,)
                    pool.terminate()
                except Exception as err:
                    pool.terminate()
                    print(err)
            pool.close()
            
            prior_samples = swyft.Samples({'params': np.concatenate(np.array([r['params'] for r in res]))})
            
        else:
            prior_samples = sim.sample(int(n_samps*excess_factor), targets=['params'])

        # prior_samples_manual = torch.rand((n_samps*excess_factor,len(bounds_gpu)), device=sampling_device)*(bounds_gpu[:,1]-bounds_gpu[:,0]) + bounds_gpu[:,0]
        
        # prior_samps = swyft.Samples(params=np.array(prior_samples_manual.to('cpu')))
        
        # prior_samps_dl = prior_samps.get_dataloader(batch_size=int(min(batch_size,n_samps*excess_factor)/n_processes), num_workers=0, pin_memory=True)
        
        # logratios1d, logratios2d = trainer.infer(network,observ,prior_samps_dl)


        repeat = int(n_samps*excess_factor) // batch_size + (int(n_samps*excess_factor) % batch_size > 0)
        
        observations_reduced = swyft.Samples({key : val[observations_unfinished] for key,val in observations.items() })
        logratios1d, logratios2d = trainer.infer(
            network,
            observations_reduced.get_dataloader(batch_size=1,repeat=repeat,num_workers=0,pin_memory=False),
            prior_samples.get_dataloader(batch_size=batch_size,num_workers=0,pin_memory=False)
        )


        if iters == 1:
            samps1d = {name[0]: np.zeros(shape=(n_samps,len(observations),1)) for name in logratios1d.parnames} 
            samps2d = {"["+names[0]+","+names[1]+"]": np.zeros(shape=(n_samps,len(observations),2)) for names in logratios2d.parnames}
            successes1d = np.array([ np.zeros(len(observations),dtype=int) for pi in range(len(logratios1d.parnames))])
            successes2d = np.array([ np.zeros(len(observations),dtype=int) for pi in range(len(logratios2d.parnames))])
            # successes2d = np.zeros(len(logratios2d.parnames),dtype=int)
            # successes_previous = np.zeros(len(logratios1d.parnames)+len(logratios2d.parnames),dtype=int)
  
        log_ratios_1d = logratios1d.logratios.to(sampling_device)
        param_values_1d = logratios1d.params.to(sampling_device)

        log_ratios_2d = logratios2d.logratios.to(sampling_device)
        param_values_2d = logratios2d.params.to(sampling_device)
        
        
        for i, name in enumerate(samps1d.keys()):
            if np.min(successes1d[i][observations_unfinished]) < n_samps:
                samples_temp = param_values_1d[:,i]
                samples_temp[torch.rand(log_ratios_1d[:,i].flatten().shape).to(sampling_device)>torch.exp(log_ratios_1d[:,i]-torch.max(log_ratios_1d[:,i]))] = np.inf
                samples_temp = samples_temp.flatten().to('cpu')
                
                samples = [ samples_temp[obi*int(excess_factor*n_samps):(obi+1)*int(excess_factor*n_samps)]  for obi in range(len(observations_unfinished))  ]
                samples = [ samples[obi][samples[obi] != np.inf] for obi in range(len(observations_unfinished))   ]
                sample_successes = [len(samples[obi]) for obi in range(len(observations_unfinished))]
                
                needed = [int(n_samps-successes1d[i][observations_unfinished[obi]]) for obi in range(len(observations_unfinished))]
                new_successes = [ min(needed[obi],sample_successes[obi]) for obi in range(len(observations_unfinished)) ]
                
                for obi in range(len(observations_unfinished)):
                    # print()
                    # print(needed[obi])
                    # print(new_successes[obi])
                    # print(successes1d[i][observations_unfinished[obi]])
                    if new_successes[obi]: samps1d[name][:,observations_unfinished[obi],:][int(successes1d[i][observations_unfinished[obi]]):int(successes1d[i][observations_unfinished[obi]]+new_successes[obi])] = samples[obi][:int(new_successes[obi]),np.newaxis]
                    successes1d[i][observations_unfinished[obi]] += new_successes[obi]
                # samps1d[name] = np.array(samps1d[name])
                
                # successes1d[i] = np.array([ len(samps1d[name][obi]) for obi in range(len(observations)) ])
                # successes_previous[i] = len(samples_temp)
             
                
                
        for i, names in enumerate(samps2d.keys()):
            if np.min(successes2d[i][observations_unfinished]) < n_samps:
                samples_temp = param_values_2d[:,i]
                samples_temp[torch.rand(log_ratios_2d[:,i].shape).to(sampling_device)>torch.exp(log_ratios_2d[:,i]-torch.max(log_ratios_2d[:,i]))] = np.inf
                samples_temp = samples_temp.to('cpu')
                
                samples = [ samples_temp[obi*int(excess_factor*n_samps):(obi+1)*int(excess_factor*n_samps)]  for obi in range(len(observations_unfinished))  ]
                samples = [ samples[obi][samples[obi][:,0] != np.inf] for obi in range(len(observations_unfinished))   ]
                sample_successes = [len(samples[obi]) for obi in range(len(observations_unfinished))]
                
                needed = [int(n_samps-successes2d[i][observations_unfinished[obi]]) for obi in range(len(observations_unfinished))]
                new_successes = [ min(needed[obi],sample_successes[obi]) for obi in range(len(observations_unfinished)) ]
                
                for obi in range(len(observations_unfinished)): 
                    if new_successes[obi]: 
                        samps2d[names][:,observations_unfinished[obi],:][int(successes2d[i][observations_unfinished[obi]]):int(successes2d[i][observations_unfinished[obi]]+new_successes[obi])] = samples[obi][:int(new_successes[obi])]
                        successes2d[i][observations_unfinished[obi]] += new_successes[obi]
                # samps1d[name] = np.array(samps1d[name])
                
                # successes2d[i] = np.array([ len(samps2d[names][obi]) for obi in range(len(observations)) ])
                # successes_previous[i+len(logratios1d.parnames)] = len(samples_temp)
                
        successes_min = min(np.min(successes1d),np.min(successes2d))
        successes_max = np.max(np.min(np.array([np.max(successes1d,axis=0),np.min(successes2d,axis=0)]),axis=0))
        
        observations_unfinished = np.arange(len(observations),dtype=int)[np.logical_or( np.min(successes1d,axis=0)<n_samps, np.min(successes2d,axis=0)<n_samps )]
        
        # print(observations_unfinished)
        # print(successes1d)
        # print(successes2d)
        # print(np.min(successes1d,axis=0))
        # print(np.min(successes2d,axis=0))
        # print(np.logical_or( np.min(successes1d,axis=0)<n_samps, np.min(successes2d,axis=0)<n_samps ))
        # print(successes_max)
        
        print('----------------------------------------------------------', flush=True)
        print('Progress: ' + str(round(100*successes_min/n_samps)) + "%")
        print('Observations finished: '+str(round((1-len(observations_unfinished)/len(observations))*100)) + "%")
        if not 1-len(100*observations_unfinished)/len(observations) > 0: print('  Closest : ' + str(round(100*successes_max/n_samps)) + "%")
        T.stop('Time spent on this round')
        print('----------------------------------------------------------', flush=True)
        
    if iters >= max_iters: raise ValueError("Maximum iterations reached!")

    return samps1d, samps2d



def draw_DRP_samples_new(tup, ):#observations,n_draws,net_path,device,bounds,nbins,POI_indices,param_names,excess_factor,max_iter):
        
    A=tup[0]
    bounds=tup[1]
    prior_funcs=tup[2]
    observations=tup[3]
    n_draws=tup[4]
    net_path=tup[5]
    device=tup[6]
    nbins=tup[7]
    POI_indices=tup[8]
    param_names=tup[9]
    excess_factor=tup[10]
    max_iter=tup[11]
    n_processes=tup[12]
    hyperparams_point = tup[13]
    enable_progress_bar=tup[14]
    
    net_dir = net_path.split('net')[0] + "net"
    
    
    draws1d = {}
    draws2d = {}
    
    module_name = 'architecture'
    spec = importlib.util.spec_from_file_location(module_name, net_dir+"/network.py")
    net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(net)
    NetworkCorner = net.NetworkCorner
    
    net = NetworkCorner(nbins=nbins, marginals=POI_indices, param_names=param_names, **hyperparams_point)

    try:
        # print(net.load_state_dict())
        net.load_state_dict(torch.load(net_path))
    except Exception as err:
        print(err)

    trainer = swyft.SwyftTrainer(accelerator = device,enable_progress_bar=enable_progress_bar,callbacks=[TQDMProgressBar(refresh_rate=100)] if enable_progress_bar else None)
      
    draws1d, draws2d = rejection_sampling_new(A,bounds,prior_funcs,trainer,net,observations,n_draws,excess_factor=excess_factor, max_iters=max_iter, sampling_device = device, n_processes=n_processes,progress_bar=enable_progress_bar)
    
    
    return draws1d, draws2d










def rejection_sampling(trainer, network,observ,n_samps,bounds,excess_factor=5, max_iters=1000, sampling_device='cuda',n_processes=1):


    
    successes_min=0
    iters=0
    
    samps1d = {}
    samps2d = {}
    
    bounds_gpu = bounds
    
    
    while successes_min<n_samps and iters<max_iters :
            
        iters += 1

        prior_samples_manual = torch.rand((n_samps*excess_factor,len(bounds_gpu)), device=sampling_device)*(bounds_gpu[:,1]-bounds_gpu[:,0]) + bounds_gpu[:,0]
        prior_samps = swyft.Samples(params=np.array(prior_samples_manual.to('cpu')).astype(np.float64))
        
        # prior_samps_dl = prior_samps.get_dataloader(batch_size=int(min(1024*32,n_samps*excess_factor)/n_processes), num_workers=0, pin_memory=True)
        
        # print(prior_samps)
        # print()
        # print('##############################################################################################################3')
        # print()
        # print(observ)
        
        logratios1d, logratios2d = trainer.infer(network,observ,prior_samps, batch_size=int(min(1024*32,n_samps*excess_factor)/n_processes))

        if iters == 1:
            samps1d = {name[0]: np.zeros(n_samps) for name in logratios1d.parnames}
            samps2d = {"["+names[0]+","+names[1]+"]": np.zeros(shape=(n_samps,2)) for names in logratios2d.parnames}
            successes1d = np.zeros(len(logratios1d.parnames),dtype=int)
            successes2d = np.zeros(len(logratios2d.parnames),dtype=int)
            successes_previous = np.zeros(len(logratios1d.parnames)+len(logratios2d.parnames),dtype=int)
  
        log_ratios_1d = logratios1d.logratios.to(sampling_device)
        param_values_1d = logratios1d.params.to(sampling_device)

        log_ratios_2d = logratios2d.logratios.to(sampling_device)
        param_values_2d = logratios2d.params.to(sampling_device)
        
        
        for i, name in enumerate(samps1d.keys()):
            if successes1d[i] < n_samps:
                samples_temp = param_values_1d[:,i][torch.rand(log_ratios_1d[:,i].flatten().shape).to(sampling_device)<torch.exp(log_ratios_1d[:,i]-torch.max(log_ratios_1d[:,i]))].flatten().to('cpu')
                needed = int(n_samps-successes1d[i])
                new_successes = min(needed,len(samples_temp))
                samps1d[name][successes1d[i]:successes1d[i]+new_successes] = samples_temp[:new_successes]
                successes1d[i] += len(samples_temp)
                successes_previous[i] = len(samples_temp)
        for i, names in enumerate(samps2d.keys()):
            if successes2d[i] < n_samps:
                samples_temp = param_values_2d[:,i][torch.rand(log_ratios_2d[:,i].shape).to(sampling_device)<torch.exp(log_ratios_2d[:,i]-torch.max(log_ratios_2d[:,i]))].to('cpu')
                needed = int(n_samps-successes2d[i])
                new_successes = min(needed,len(samples_temp))
                samps2d[names][successes2d[i]:successes2d[i]+new_successes] = samples_temp[:new_successes]
                successes2d[i] += len(samples_temp)
                successes_previous[i+len(logratios1d.parnames)] = len(samples_temp)
                
        successes_min = min(min(successes1d),min(successes2d))
 
        
    if iters >= max_iters: raise ValueError("Maximum iterations reached!")

    
    return samps1d, samps2d



def draw_DRP_samples(tup, ):#observations,n_draws,net_path,device,bounds,nbins,POI_indices,param_names,excess_factor,max_iter):
        
    
    observations=tup[0]
    n_draws=tup[1]
    net_path=tup[2]
    device=tup[3]
    bounds=tup[4]
    nbins=tup[5]
    POI_indices=tup[6]
    param_names=tup[7]
    excess_factor=tup[8]
    max_iter=tup[9]
    n_processes=tup[10]
    hyperparams_point = tup[11]
    
    net_dir = net_path.split('net')[0] + "net"
    
    
    draws1d = {}
    draws2d = {}
    
    n_obs= len(observations)
    
    module_name = 'architecture'
    spec = importlib.util.spec_from_file_location(module_name, net_dir+"/network.py")
    net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(net)
    NetworkCorner = net.NetworkCorner
    
    

    net = NetworkCorner(nbins=nbins, marginals=POI_indices, param_names=param_names, **hyperparams_point)

    try:
        # print(net.load_state_dict())
        net.load_state_dict(torch.load(net_path))
    except Exception as err:
        print(err)

    trainer = swyft.SwyftTrainer(accelerator = device,enable_progress_bar=False,precision=64)
    
    bounds_gpu = torch.tensor(bounds).to(device)
    
    obs_range = tqdm(np.arange(-n_obs,0,dtype=int))
    for obs_i in obs_range:
        obs_range.refresh()
        

        draws1d_raw, draws2d_raw = rejection_sampling(trainer,net,observations[int(obs_i)],n_draws,bounds=bounds_gpu,excess_factor=excess_factor, max_iters=max_iter, sampling_device = device, n_processes=n_processes)
        
        for i, names in enumerate(draws1d_raw.keys()):
            if obs_i == -n_obs: 
                draws1d[names] = np.zeros(shape=(n_draws,n_obs,1))
            draws1d[names][:,obs_i,0] = draws1d_raw[names]
    
        for i, names in enumerate(draws2d_raw.keys()):
            if obs_i == -n_obs: 
                draws2d[names] = np.zeros(shape=(n_draws,n_obs,2))
            draws2d[names][:,obs_i,:] = draws2d_raw[names]
    
    obs_range.close()
 
    return draws1d, draws2d, n_obs


def draw_DRP_samples_fast(
        net,
        trainer,
        samples,
        prior_samples,
        batch_size=1024,
        ):
    
        repeat = len(prior_samples) // batch_size + (len(prior_samples) % batch_size > 0)
        logratios1d, logratios2d = trainer.infer(
            net,
            samples.get_dataloader(batch_size=1,repeat=repeat),
            prior_samples.get_dataloader(batch_size=batch_size),
        )
        
        draws1d = {}
        draws2d = {}
        weights1d = {}
        weights2d = {}
    
        for i, name in enumerate(logratios1d.parnames):
                draws1d[name[0]] = np.zeros(shape=(len(prior_samples),len(samples),1))
                for si in range(len(samples)):
                    draws1d[name[0]][:,si,:] = np.array(logratios1d.params[si*len(prior_samples):(si+1)*len(prior_samples),i,:])
                # print('yay')
                weights1d[name[0]] = np.array([ np.exp(np.array(logratios1d.logratios[si*len(prior_samples):(si+1)*len(prior_samples),i]))  for si in range(len(samples))  ]).T
                weights1d[name[0]] = weights1d[name[0]].shape[0]*weights1d[name[0]]/np.sum(weights1d[name[0]],axis=0)
                
        for i, names in enumerate(logratios2d.parnames):
                draws2d[str(names)] = np.zeros(shape=(len(prior_samples),len(samples),2))
                for si in range(len(samples)):
                    draws2d[str(names)][:,si,:] = np.array(logratios2d.params[si*len(prior_samples):(si+1)*len(prior_samples),i,:])
                weights2d[str(names)] = np.array([ np.exp(np.array(logratios2d.logratios[si*len(prior_samples):(si+1)*len(prior_samples),i]))  for si in range(len(samples))  ]).T
                weights2d[str(names)] = weights2d[str(names)].shape[0]*weights2d[str(names)]/np.sum(weights2d[str(names)],axis=0)
    
        return draws1d, draws2d, weights1d, weights2d
    



def get_drp_coverage(
    samples: np.ndarray,
    theta: np.ndarray,
    references: Union[str, np.ndarray] = "random",
    weights = None,
    bounds: Union[list[float],np.ndarray] = [],
    theta_names: list[str]=[],
    axes: Union[int,list] = 9,
    metric: str = "euclidean",
    intermediate_figures=False,
    maxy = 1,
    markersize=70,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates coverage with the distance to random point method.

    Reference: `Lemos, Coogan et al 2023 <https://arxiv.org/abs/2302.03026>`_

    Args:
        samples: the samples to compute the coverage of, with shape ``(n_samples, n_sims, n_dims)``.
        theta: the true parameter values for each samples, with shape ``(n_sims, n_dims)``.
        references: the reference points to use for the DRP regions, with shape
            ``(n_references, n_sims)``, or the string ``"random"``. If the later, then
            the reference points are chosen randomly from the unit hypercube over
            the parameter space.
        metric: the metric to use when computing the distance. Can be ``"euclidean"`` or
            ``"manhattan"``.

    Returns:
        Credibility values (``alpha``) and expected coverage probability (``ecp``).
    """


       
    
    # Check that shapes are correct
    if samples.ndim != 3:
        raise ValueError("samples must be a 3D array")

    if theta.ndim != 2:
        raise ValueError("theta must be a 2D array")

    num_samples = samples.shape[0]
    num_sims = samples.shape[1]
    num_dims = samples.shape[2]

    if theta.shape[0] != num_sims:
        raise ValueError("theta must have the same number of rows as samples")

    if theta.shape[1] != num_dims:
        raise ValueError("theta must have the same number of columns as samples")

    # Reshape theta
    theta = theta[np.newaxis, :, :]

    
    if len(bounds) == 0:
        low = np.min(samples, axis=(0,1), keepdims=True)
        high = np.max(samples, axis=(0,1), keepdims=True)
        
    else:
        bounds_temp = np.array(bounds)[np.newaxis,:,:]
        low = bounds_temp[np.newaxis,:,:,0]
        high = bounds_temp[np.newaxis,:,:,1]
    
    
    # Normalize
    # low = np.min(theta, axis=1, keepdims=True)
    # high = np.max(theta, axis=1, keepdims=True)
    samples = (samples - low) / (high - low)
    theta = (theta - low) / (high - low)

    # Generate reference points
    if isinstance(references, str) and references == "random":
        references = np.random.uniform(low=0, high=1, size=(num_sims, num_dims))     # ORIGINAL
        # references = np.random.uniform(low=-1, high=2, size=(num_sims, num_dims))
    else:
        assert isinstance(references, np.ndarray)  # to quiet pyright
        if references.ndim != 2:
            raise ValueError("references must be a 2D array")

        if references.shape[0] != num_sims:
            raise ValueError("references must have the same number of rows as samples")

        if references.shape[1] != num_dims:
            raise ValueError(
                "references must have the same number of columns as samples"
            )

    # Compute distances
    if metric == "euclidean":
        samples_distances = np.sqrt(
            np.sum((references[np.newaxis] - samples) ** 2, axis=-1)
        )
        theta_distances = np.sqrt(np.sum((references - theta) ** 2, axis=-1))
        
    elif metric == "manhattan":
        samples_distances = np.sum(np.abs(references[np.newaxis] - samples), axis=-1)
        theta_distances = np.sum(np.abs(references - theta), axis=-1)
    else:
        raise ValueError("metric must be either 'euclidean' or 'manhattan'")

    

    if not weights is None: cmap = cm.get_cmap('turbo')
    theta_plot = theta.copy()
    references_plot = references.copy()
    theta_distances_plot = np.array([theta_distances.copy(),theta_distances.copy()])
    theta_plot[0,:,0] = theta[0,:,0]*(bounds[0][1]-bounds[0][0]) + bounds[0][0]
    references_plot[:,0] = references[:,0]*(bounds[0][1]-bounds[0][0]) + bounds[0][0]
    theta_distances_plot[0,:] = theta_distances[0,:]*(bounds[0][1]-bounds[0][0])
    if num_dims > 1:
        theta_plot[0,:,1] = theta[0,:,1]*(bounds[1][1]-bounds[1][0]) + bounds[1][0]
        references_plot[:,1] = references[:,1]*(bounds[1][1]-bounds[1][0]) + bounds[1][0]
        theta_distances_plot[1,:] = theta_distances[0,:]*(bounds[1][1]-bounds[1][0])

    figures = None
    if intermediate_figures:
        
        if isinstance(axes,int): 
            n_figs = min(axes,samples.shape[1])                          
            figures = list(np.zeros(n_figs))
            axes = list(np.zeros(n_figs))
            for i in range(len(figures)):
                figures[i], axes[i] = plt.subplots()
                axes[i].set_xlim([0,1])
                axes[i].set_ylim([0,1])
        else:                                         #could use some error handling
            n_figs = min(len(axes),samples.shape[1])

        samples_plot = samples[:,:n_figs,:]
        
        zorder=1
        for i in range(n_figs):
            n_inside=0
            samples_plot_inside = samples_plot[  samples_distances[:,i] < theta_distances[0,i] , i, : ]
            samples_plot_outside = samples_plot[  samples_distances[:,i] >= theta_distances[0,i], i, :  ]
            
            
            
            if weights is None:
                n_inside = len(samples_plot_inside[:,0])
            else:
                weights_inside = weights[samples_distances[:,i] < theta_distances[0,i],i]
                weights_outside = weights[samples_distances[:,i] >= theta_distances[0,i],i]
                n_inside = np.sum(np.ones(len(samples_plot_inside[:,0]))*weights_inside)
    
    
            if num_dims == 1:
                
                n_bins_1d = max(10,num_samples//500)
                bin_edges = np.linspace(0,1,n_bins_1d)
    
                maxweight = np.max(weights[:,i])
                minweight = np.max(weights[:,i])
    
                for ed in range(len(bin_edges)-1):
                    samples_plot_inside_temp = samples_plot_inside[bin_edges[ed]<=samples_plot_inside[:,0]]
                    samples_plot_outside_temp = samples_plot_outside[bin_edges[ed]<=samples_plot_outside[:,0]]
                    samples_plot_inside_bin = samples_plot_inside_temp[bin_edges[ed+1]>samples_plot_inside_temp[:,0]]
                    samples_plot_outside_bin = samples_plot_outside_temp[ bin_edges[ed+1]>samples_plot_outside_temp[:,0]]
    
                    if not weights is None:
                        weights_inside_temp = weights_inside[bin_edges[ed]<=samples_plot_inside[:,0]]
                        weights_outside_temp = weights_outside[bin_edges[ed]<=samples_plot_outside[:,0]]
                        weights_in_bin = weights_inside_temp[bin_edges[ed+1]>samples_plot_inside_temp[:,0]]
                        weights_not_in_bin = weights_outside_temp[ bin_edges[ed+1]>samples_plot_outside_temp[:,0]]
                    
                    lenin = len(samples_plot_inside_bin[:,0])
                    lenout = len(samples_plot_outside_bin[:,0])
                    
                    num_samps_bin = lenin+lenout
    
                    samples_plot_inside_bin[:,0] = samples_plot_inside_bin[:,0]*(bounds[0][1]-bounds[0][0]) + bounds[0][0]
                    samples_plot_outside_bin[:,0] = samples_plot_outside_bin[:,0]*(bounds[0][1]-bounds[0][0]) + bounds[0][0]
                    
                    if weights is None:
                        axes[i].scatter(samples_plot_inside_bin[:,0],np.random.uniform(num_samps_bin/num_samples,size=lenin),marker='x',color='c',zorder=zorder)
                        axes[i].scatter(samples_plot_outside_bin[:,0],np.random.uniform(num_samps_bin/num_samples,size=lenout),marker='x',color='b',zorder=zorder)
                    else:
                        markersizes_inside = markersize*weights_in_bin**0.5
                        markersizes_outside = markersize*weights_not_in_bin**0.5
                        axes[i].scatter(samples_plot_inside_bin[:,0],np.random.uniform(0,maxy*weights_in_bin/(maxweight)),s=markersizes_inside,marker='.',color='c',zorder=zorder)
                        axes[i].scatter(samples_plot_outside_bin[:,0],np.random.uniform(0,maxy*weights_not_in_bin/(maxweight)),s=markersizes_outside,marker='.',color='b',zorder=zorder)
                        # axes[i].scatter(samples_plot_inside_bin[:,0],np.random.uniform(0,maxy,size=len(weights_in_bin)),s=markersizes_inside,marker='.',color='c',zorder=2)
                        # axes[i].scatter(samples_plot_outside_bin[:,0],np.random.uniform(0,maxy,size=len(weights_not_in_bin)),s=markersizes_outside,marker='.',color='b',zorder=2)
                  
                    
                axes[i].axvline(theta_plot[0,i,0],color='r')
                axes[i].axvline(references_plot[i,0],color='g')
                axes[i].axvline(references_plot[i,0]-theta_distances_plot[0,i],color='r',linestyle=":")
                axes[i].axvline(references_plot[i,0]+theta_distances_plot[0,i],color='r',linestyle=":")
                if len(theta_names) >= 1:
                    axes[i].set_xlabel(theta_names[0])
                axes[i].set_title('Samples in $\Theta_{{DRP}}$: {:.1f}/{:.0f} = {:.1f}%'.format(n_inside,num_samples, 100*n_inside/num_samples))
                axes[i].set_title('Samples in $\Theta_{{DRP}}$: {:.1f}%'.format(100*n_inside/num_samples))
                axes[i].set_yticks([])
    
    
            
            if num_dims == 2:
                
                # print(references)
                # print(samples_plot_inside)
                samples_plot_inside[:,0] = samples_plot_inside[:,0]*(bounds[0][1]-bounds[0][0]) + bounds[0][0]
                samples_plot_inside[:,1] = samples_plot_inside[:,1]*(bounds[1][1]-bounds[1][0]) + bounds[1][0]
                samples_plot_outside[:,0] = samples_plot_outside[:,0]*(bounds[0][1]-bounds[0][0]) + bounds[0][0]
                samples_plot_outside[:,1] = samples_plot_outside[:,1]*(bounds[1][1]-bounds[1][0]) + bounds[1][0]
    
    
                # circle = plt.Circle((references_plot[i,0],references_plot[i,1]), theta_distances_plot[0,i], fill=0, color='r')
                circle = Ellipse((references_plot[i,0],references_plot[i,1]), 2*theta_distances_plot[0,i],2*theta_distances_plot[1,i], fill=0, color='r',zorder=1)
                axes[i].add_patch(circle)
    
                if weights is None:
                    axes[i].scatter(samples_plot_inside[:,0],samples_plot_inside[:,1],marker='x',color='c',zorder=zorder)
                    axes[i].scatter(samples_plot_outside[:,0],samples_plot_outside[:,1],marker='x',color='b',zorder=zorder)
                else:
                    markersizes_inside = markersize*weights_inside**0.5
                    markersizes_outside = markersize*weights_outside**0.5
                    axes[i].scatter(samples_plot_inside[:,0],samples_plot_inside[:,1],s=markersizes_inside,marker='.',color='c',zorder=zorder)
                    axes[i].scatter(samples_plot_outside[:,0],samples_plot_outside[:,1],s=markersizes_outside,marker='.',color='b',zorder=zorder)
                
                
                axes[i].scatter(theta_plot[0,i,0], theta_plot[0,i,1],color='r',marker='x',s=80,linewidths=2.5,zorder=2)
                axes[i].scatter(references_plot[i,0], references_plot[i,1],color='g',marker='+',s=80,linewidths=2.5,zorder=2)
                # axes[i].set_xlim([-0.1, 1.1])
                # axes[i].set_ylim([-0.1, 1.1])
                if len(theta_names) >= 2:
                    axes[i].set_xlabel(theta_names.split(',')[0].split('[')[1])
                    axes[i].set_ylabel(theta_names.split(',')[1].split(']')[0])
                # axes[i].set_aspect('equal', adjustable='box')
                # axes[i].plot([bounds[0][0], bounds[1][0]],[bounds[0][0], bounds[1][1]], ':' , c='0.5')
                # axes[i].plot([bounds[0][0], bounds[1][1]],[bounds[0][0], bounds[1][0]], ':' , c='0.5')
                # axes[i].plot([bounds[0][0], bounds[1][1]],[bounds[0][1], bounds[1][1]], ':' , c='0.5')
                # axes[i].plot([bounds[0][1], bounds[1][1]],[bounds[0][0], bounds[1][1]], ':' , c='0.5')
                # axes[i].set_title('Samples in $\Theta_{{DRP}}$: {:.1f}/{:.0f} = {:.1f}%'.format(n_inside,len(samples[:,i,0]), 100*n_inside/len(samples[:,i,0])))
                axes[i].set_title('Samples in $\Theta_{{DRP}}$: {:.1f}%'.format(100*n_inside/len(samples[:,i,0])))

    # Compute coverage
    if weights is None:
        f = np.sum((samples_distances < theta_distances), axis=0) / num_samples
    else:
        f = np.sum((samples_distances < theta_distances)*weights, axis=0) / num_samples

    # Compute expected coverage
    h, alpha = np.histogram(f, density=True, bins= min(20, num_sims // min(10,num_sims)))
    dx = alpha[1] - alpha[0]
    ecp = np.cumsum(h) * dx
    return ecp, alpha[1:], f, figures





def plot_DRP_coverage(references_1d,references_2d,draws1d,draws2d,truths,truth_names,bounds,
                      ref_colors =['b','c','g','y','r','m',] ):

    
    ecp = [[{} for ref_list in references_1d],[{} for ref_list in references_2d]]
    alpha = [[{} for ref_list in references_1d],[{} for ref_list in references_2d]]
    f = [[{} for ref_list in references_1d],[{} for ref_list in references_2d]]
    figs_cov = [[{} for ref_list in references_1d],[{} for ref_list in references_2d]]
    
    num_axes = 9
    columns = 3
    rows = int(np.ceil(num_axes/columns))

    n_sims1d = references_1d[0].shape[0]
    n_sims2d = references_2d[0].shape[0]
    
    for ref_i in range(len(references_1d)):
        for i, name in enumerate(draws1d.keys()):
            figs_cov[0][ref_i][name] = plt.figure(figsize = (4*columns, 4*rows))
            for fig_i in range(min(num_axes,n_sims1d)): 
                figs_cov[0][ref_i][name].add_subplot(rows, columns, fig_i+1)
            figs_cov[0][ref_i][name].suptitle(name)
            
            ecp[0][ref_i][name], alpha[0][ref_i][name], f[0][ref_i][name], _ = get_drp_coverage(draws1d[name], 
                                                                  truths[0][i],
                                                                  theta_names=truth_names[0][i],
                                                                  axes = figs_cov[0][ref_i][name].axes,
                                                                  bounds = np.array(bounds[0])[[i]],
                                                                  references = references_1d[ref_i],
                                                                 )
    
    for ref_i in range(len(references_2d)): 
        for i, names in enumerate(draws2d.keys()):
            figs_cov[1][ref_i][names] = plt.figure(figsize = (4*columns, 4*rows))
            for fig_i in range(min(num_axes,n_sims2d)): 
                figs_cov[1][ref_i][names].add_subplot(rows, columns, fig_i+1)
            figs_cov[1][ref_i][names].suptitle(names)
            
            ecp[1][ref_i][names], alpha[1][ref_i][names], f[1][ref_i][names], _ = get_drp_coverage(draws2d[names], 
                                                                     truths[1][i],
                                                                     theta_names=truth_names[1][i],
                                                                     axes = figs_cov[1][ref_i][names].axes,
                                                                     bounds = bounds[1][i],
                                                                     references = references_2d[ref_i],
                                                                    )
    
    
    rows = len(draws1d.keys())
    fig_DRP = plt.figure(figsize = (4*rows, 4*rows))
    
    
    for i, name in enumerate(draws1d.keys()):
    
        fig_DRP.add_subplot(rows, rows, i+1+i*rows)
        plt.plot([0,1],[0,1], 'k:')
        plt.xlabel("Credibility level (alpha)")
        plt.ylabel("Expected Coverage Probability (ECP)")
        
        for ref_i in range(len(references_1d)):
             
            ecp_ex = np.zeros(len(ecp[0][ref_i][name])+1)
            alpha_ex = np.zeros(len(alpha[0][ref_i][name])+1)
            ecp_ex[1:] = ecp[0][ref_i][name]
            alpha_ex[1:] = alpha[0][ref_i][name]
            plt.plot(alpha_ex, ecp_ex, ref_colors[ref_i])
            
    
    row = 0
    column = 0
    for i, names in enumerate(draws2d.keys()):
    
        row+=1
        if row >= rows:
            column+=1
            row = 1+column
            
        fig_DRP.add_subplot(rows, rows, rows*row+column+1)
        plt.plot([0,1],[0,1], 'k:')
        plt.xlabel("Credibility level (alpha)")
        plt.ylabel("Expected Coverage Probability (ECP)")
        
        for ref_i in range(len(references_2d)):
        
            ecp_ex = np.zeros(len(ecp[1][ref_i][names])+1)
            alpha_ex = np.zeros(len(alpha[1][ref_i][names])+1)
            ecp_ex[1:] = ecp[1][ref_i][names]
            alpha_ex[1:] = alpha[1][ref_i][names]
            plt.plot(alpha_ex, ecp_ex, ref_colors[ref_i])

    
    return fig_DRP, figs_cov




def get_drp_coverage_torch(
    samples: np.ndarray,
    theta: np.ndarray,
    references: Union[str, np.ndarray] = "random",
    weights = None,
    bounds: Union[list[float],np.ndarray] = [],
    theta_names: list[str]=[],
    axes: Union[int,list] = 9,
    metric: str = "euclidean",
    intermediate_figures=False,
    generate_zz = True,
    maxy = 1,
    markersize=70,
    device = 'cpu',
    bins=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates coverage with the distance to random point method.

    Reference: `Lemos, Coogan et al 2023 <https://arxiv.org/abs/2302.03026>`_

    Args:
        samples: the samples to compute the coverage of, with shape ``(n_samples, n_sims, n_dims)``.
        theta: the true parameter values for each samples, with shape ``(n_sims, n_dims)``.
        references: the reference points to use for the DRP regions, with shape
            ``(n_references, n_sims)``, or the string ``"random"``. If the later, then
            the reference points are chosen randomly from the unit hypercube over
            the parameter space.
        metric: the metric to use when computing the distance. Can be ``"euclidean"`` or
            ``"manhattan"``.

    Returns:
        Credibility values (``alpha``) and expected coverage probability (``ecp``).
    """

    
    samples = torch.tensor(samples,device=device)
    theta = torch.tensor(theta,device=device)
    references = torch.tensor(references,device=device)
    weights = torch.tensor(weights,device=device)
       
    # Check that shapes are correct
    if samples.ndim != 3:
        raise ValueError("samples must be a 3D array")

    if theta.ndim != 2:
        raise ValueError("theta must be a 2D array")

    num_samples = samples.shape[0]
    num_sims = samples.shape[1]
    num_dims = samples.shape[2]

    if theta.shape[0] != num_sims:
        raise ValueError("theta must have the same number of rows as samples")

    if theta.shape[1] != num_dims:
        raise ValueError("theta must have the same number of columns as samples")

    # Reshape theta
    theta = theta[None, :, :]

    if len(bounds) == 0:
        low = torch.min(samples, axis=(0,1), keepdims=True, device=device)
        high = torch.max(samples, axis=(0,1), keepdims=True, device=device)
        
    else:
        bounds_temp = torch.tensor(bounds, device=device)[None,:,:]
        low = bounds_temp[None,:,:,0]
        high = bounds_temp[None,:,:,1]
    
    # Normalize
    samples = (samples - low) / (high - low)
    theta = (theta - low) / (high - low)

    # Generate reference points
    if isinstance(references, str) and references == "random":
        references = torch.rand((num_sims, num_dims), device=device)
    else:
        # assert isinstance(references, np.ndarray)  # to quiet pyright
        if references.ndim != 2:
            raise ValueError("references must be a 2D array")

        if references.shape[0] != num_sims:
            raise ValueError("references must have the same number of rows as samples")

        if references.shape[1] != num_dims:
            raise ValueError(
                "references must have the same number of columns as samples"
            )

    # Compute distances
    if metric == "euclidean":
        samples_distances = torch.sqrt(
            torch.sum((references[None] - samples) ** 2, axis=-1)
        )
        theta_distances = torch.sqrt(torch.sum((references - theta) ** 2, axis=-1))
        
    elif metric == "manhattan":
        samples_distances = torch.sum(np.abs(references[None] - samples), axis=-1)
        theta_distances = torch.sum(np.abs(references - theta), axis=-1)
    else:
        raise ValueError("metric must be either 'euclidean' or 'manhattan'")


    ###### Figures #######

    figures = None
    if intermediate_figures:
        
        theta_plot = copy.deepcopy(theta)
        references_plot = copy.deepcopy(references)
        theta_distances_plot = torch.stack([copy.deepcopy(theta_distances),copy.deepcopy(theta_distances)])
        theta_plot[0,:,0] = theta[0,:,0]*(bounds[0][1]-bounds[0][0]) + bounds[0][0]
        references_plot[:,0] = references[:,0]*(bounds[0][1]-bounds[0][0]) + bounds[0][0]
        theta_distances_plot[0,:] = theta_distances[0,:]*(bounds[0][1]-bounds[0][0])
        if num_dims > 1:
            theta_plot[0,:,1] = theta[0,:,1]*(bounds[1][1]-bounds[1][0]) + bounds[1][0]
            references_plot[:,1] = references[:,1]*(bounds[1][1]-bounds[1][0]) + bounds[1][0]
            theta_distances_plot[1,:] = theta_distances[0,:]*(bounds[1][1]-bounds[1][0])
        
        if isinstance(axes,int): 
            n_figs = min(axes,samples.shape[1])                          
            figures = list(np.zeros(n_figs))
            axes = list(np.zeros(n_figs))
            for i in range(len(figures)):
                figures[i], axes[i] = plt.subplots()
                axes[i].set_xlim([0,1])
                axes[i].set_ylim([0,1])
        else:                                         #could use some error handling
            n_figs = min(len(axes),samples.shape[1])

        samples_plot = samples[:,:n_figs,:]
        
        zorder=1
        for i in range(n_figs):
            n_inside=0
            samples_plot_inside = samples_plot[  samples_distances[:,i] < theta_distances[0,i] , i, : ]
            samples_plot_outside = samples_plot[  samples_distances[:,i] >= theta_distances[0,i], i, :  ]
            
            if weights is None:
                n_inside = len(samples_plot_inside[:,0])
            else:
                weights_inside = weights[samples_distances[:,i] < theta_distances[0,i],i]
                weights_outside = weights[samples_distances[:,i] >= theta_distances[0,i],i]
                n_inside = np.sum(np.ones(len(samples_plot_inside[:,0]))*weights_inside)
    
    
            if num_dims == 1:
                
                n_bins_1d = max(10,num_samples//500)
                bin_edges = np.linspace(0,1,n_bins_1d)
    
                maxweight = np.max(weights[:,i])
                minweight = np.max(weights[:,i])
    
                for ed in range(len(bin_edges)-1):
                    samples_plot_inside_temp = samples_plot_inside[bin_edges[ed]<=samples_plot_inside[:,0]]
                    samples_plot_outside_temp = samples_plot_outside[bin_edges[ed]<=samples_plot_outside[:,0]]
                    samples_plot_inside_bin = samples_plot_inside_temp[bin_edges[ed+1]>samples_plot_inside_temp[:,0]]
                    samples_plot_outside_bin = samples_plot_outside_temp[ bin_edges[ed+1]>samples_plot_outside_temp[:,0]]
    
                    if not weights is None:
                        weights_inside_temp = weights_inside[bin_edges[ed]<=samples_plot_inside[:,0]]
                        weights_outside_temp = weights_outside[bin_edges[ed]<=samples_plot_outside[:,0]]
                        weights_in_bin = weights_inside_temp[bin_edges[ed+1]>samples_plot_inside_temp[:,0]]
                        weights_not_in_bin = weights_outside_temp[ bin_edges[ed+1]>samples_plot_outside_temp[:,0]]
                    
                    lenin = len(samples_plot_inside_bin[:,0])
                    lenout = len(samples_plot_outside_bin[:,0])
                    
                    num_samps_bin = lenin+lenout
    
                    samples_plot_inside_bin[:,0] = samples_plot_inside_bin[:,0]*(bounds[0][1]-bounds[0][0]) + bounds[0][0]
                    samples_plot_outside_bin[:,0] = samples_plot_outside_bin[:,0]*(bounds[0][1]-bounds[0][0]) + bounds[0][0]
                    
                    if weights is None:
                        axes[i].scatter(samples_plot_inside_bin[:,0],np.random.uniform(num_samps_bin/num_samples,size=lenin),marker='x',color='c',zorder=zorder)
                        axes[i].scatter(samples_plot_outside_bin[:,0],np.random.uniform(num_samps_bin/num_samples,size=lenout),marker='x',color='b',zorder=zorder)
                    else:
                        markersizes_inside = markersize*weights_in_bin**0.5
                        markersizes_outside = markersize*weights_not_in_bin**0.5
                        axes[i].scatter(samples_plot_inside_bin[:,0],np.random.uniform(0,maxy*weights_in_bin/(maxweight)),s=markersizes_inside,marker='.',color='c',zorder=zorder)
                        axes[i].scatter(samples_plot_outside_bin[:,0],np.random.uniform(0,maxy*weights_not_in_bin/(maxweight)),s=markersizes_outside,marker='.',color='b',zorder=zorder)
                  
                    
                axes[i].axvline(theta_plot[0,i,0],color='r')
                axes[i].axvline(references_plot[i,0],color='g')
                axes[i].axvline(references_plot[i,0]-theta_distances_plot[0,i],color='r',linestyle=":")
                axes[i].axvline(references_plot[i,0]+theta_distances_plot[0,i],color='r',linestyle=":")
                if len(theta_names) >= 1:
                    axes[i].set_xlabel(theta_names[0])
                axes[i].set_title('Samples in $\Theta_{{DRP}}$: {:.1f}/{:.0f} = {:.1f}%'.format(n_inside,num_samples, 100*n_inside/num_samples))
                axes[i].set_title('Samples in $\Theta_{{DRP}}$: {:.1f}%'.format(100*n_inside/num_samples))
                axes[i].set_yticks([])
    
    
    
            if num_dims == 2:
                
                samples_plot_inside[:,0] = samples_plot_inside[:,0]*(bounds[0][1]-bounds[0][0]) + bounds[0][0]
                samples_plot_inside[:,1] = samples_plot_inside[:,1]*(bounds[1][1]-bounds[1][0]) + bounds[1][0]
                samples_plot_outside[:,0] = samples_plot_outside[:,0]*(bounds[0][1]-bounds[0][0]) + bounds[0][0]
                samples_plot_outside[:,1] = samples_plot_outside[:,1]*(bounds[1][1]-bounds[1][0]) + bounds[1][0]
    
                # circle = plt.Circle((references_plot[i,0],references_plot[i,1]), theta_distances_plot[0,i], fill=0, color='r')
                circle = Ellipse((references_plot[i,0],references_plot[i,1]), 2*theta_distances_plot[0,i],2*theta_distances_plot[1,i], fill=0, color='r',zorder=1)
                axes[i].add_patch(circle)
    
                if weights is None:
                    axes[i].scatter(samples_plot_inside[:,0],samples_plot_inside[:,1],marker='x',color='c',zorder=zorder)
                    axes[i].scatter(samples_plot_outside[:,0],samples_plot_outside[:,1],marker='x',color='b',zorder=zorder)
                else:
                    markersizes_inside = markersize*weights_inside**0.5
                    markersizes_outside = markersize*weights_outside**0.5
                    axes[i].scatter(samples_plot_inside[:,0],samples_plot_inside[:,1],s=markersizes_inside,marker='.',color='c',zorder=zorder)
                    axes[i].scatter(samples_plot_outside[:,0],samples_plot_outside[:,1],s=markersizes_outside,marker='.',color='b',zorder=zorder)
                
                axes[i].scatter(theta_plot[0,i,0], theta_plot[0,i,1],color='r',marker='x',s=80,linewidths=2.5,zorder=2)
                axes[i].scatter(references_plot[i,0], references_plot[i,1],color='g',marker='+',s=80,linewidths=2.5,zorder=2)
                
                if len(theta_names) >= 2:
                    axes[i].set_xlabel(theta_names.split(',')[0].split('[')[1])
                    axes[i].set_ylabel(theta_names.split(',')[1].split(']')[0])

                axes[i].set_title('Samples in $\Theta_{{DRP}}$: {:.1f}%'.format(100*n_inside/len(samples[:,i,0])))


    ###### Figures end #######
    

    # Compute coverage
    if weights is None:
        f = torch.sum((samples_distances < theta_distances), axis=0) / num_samples
    else:
        f = torch.sum((samples_distances < theta_distances)*weights, axis=0) / num_samples
    f_pp=f.to('cpu')
    
    if bins is None: bins = min(50, num_sims // min(10,num_sims))
    
    # Compute expected coverage
    h_pp = np.array(torch.histogram(f_pp, density=True, bins=bins, range=(0,1)).hist)
    alpha_pp = np.linspace(0,1,bins+1)
    num_sims_per_bin = num_sims/bins
    uncertainty=3*num_sims_per_bin**(-0.5)
    f_score = np.sum(((h_pp-1)/uncertainty)**2)/bins
    dx_pp = alpha_pp[1] - alpha_pp[0]
    ecp_pp = np.cumsum(h_pp) * dx_pp
    
    # print(num_sims)
    # print(alpha_pp)
    # print(num_sims_per_bin)
    # print(uncertainty)
    
    f_zz = None
    alpha_zz = None
    ecp_zz = None
    if generate_zz:
        f_zz = stdnorm.ppf(0.5+f_pp/2)
        max_f_zz = np.max(f_zz[np.invert(np.isinf(f_zz))])
        f_zz = np.where(np.isinf(f_zz),max_f_zz,f_zz)
        f_zz = np.where(np.isnan(f_zz),0,f_zz)
        # f_zz = f_zz[np.invert(np.isinf(f_zz))]
        # f_zz = f_zz[np.invert(np.isnan(f_zz))]
        h_zz, alpha_zz = np.histogram(f_zz, density=True,range=(0,np.max(f_zz)), bins=bins)
        dx_zz = alpha_zz[1] - alpha_zz[0]
        ecp_zz = np.cumsum(h_zz) * dx_zz
        ecp_zz = stdnorm.ppf(0.5+ecp_zz/2)

    return ecp_pp, alpha_pp[1:], ecp_zz, alpha_zz[1:], f_pp, f_zz, f_score, figures





def plot_DRP_coverage(references_1d,references_2d,draws1d,draws2d,truths,truth_names,bounds,
                      ref_colors =['b','c','g','y','r','m',] ):

    
    ecp = [[{} for ref_list in references_1d],[{} for ref_list in references_2d]]
    alpha = [[{} for ref_list in references_1d],[{} for ref_list in references_2d]]
    f = [[{} for ref_list in references_1d],[{} for ref_list in references_2d]]
    figs_cov = [[{} for ref_list in references_1d],[{} for ref_list in references_2d]]
    
    num_axes = 9
    columns = 3
    rows = int(np.ceil(num_axes/columns))

    n_sims1d = references_1d[0].shape[0]
    n_sims2d = references_2d[0].shape[0]
    
    for ref_i in range(len(references_1d)):
        for i, name in enumerate(draws1d.keys()):
            figs_cov[0][ref_i][name] = plt.figure(figsize = (4*columns, 4*rows))
            for fig_i in range(min(num_axes,n_sims1d)): 
                figs_cov[0][ref_i][name].add_subplot(rows, columns, fig_i+1)
            figs_cov[0][ref_i][name].suptitle(name)
            
            ecp[0][ref_i][name], alpha[0][ref_i][name], f[0][ref_i][name], _ = get_drp_coverage(draws1d[name], 
                                                                  truths[0][i],
                                                                  theta_names=truth_names[0][i],
                                                                  axes = figs_cov[0][ref_i][name].axes,
                                                                  bounds = np.array(bounds[0])[[i]],
                                                                  references = references_1d[ref_i],
                                                                 )
    
    for ref_i in range(len(references_2d)): 
        for i, names in enumerate(draws2d.keys()):
            figs_cov[1][ref_i][names] = plt.figure(figsize = (4*columns, 4*rows))
            for fig_i in range(min(num_axes,n_sims2d)): 
                figs_cov[1][ref_i][names].add_subplot(rows, columns, fig_i+1)
            figs_cov[1][ref_i][names].suptitle(names)
            
            ecp[1][ref_i][names], alpha[1][ref_i][names], f[1][ref_i][names], _ = get_drp_coverage(draws2d[names], 
                                                                     truths[1][i],
                                                                     theta_names=truth_names[1][i],
                                                                     axes = figs_cov[1][ref_i][names].axes,
                                                                     bounds = bounds[1][i],
                                                                     references = references_2d[ref_i],
                                                                    )
    
    
    rows = len(draws1d.keys())
    fig_DRP = plt.figure(figsize = (4*rows, 4*rows))
    
    
    for i, name in enumerate(draws1d.keys()):
    
        fig_DRP.add_subplot(rows, rows, i+1+i*rows)
        plt.plot([0,1],[0,1], 'k:')
        plt.xlabel("Credibility level (alpha)")
        plt.ylabel("Expected Coverage Probability (ECP)")
        
        for ref_i in range(len(references_1d)):
             
            ecp_ex = np.zeros(len(ecp[0][ref_i][name])+1)
            alpha_ex = np.zeros(len(alpha[0][ref_i][name])+1)
            ecp_ex[1:] = ecp[0][ref_i][name]
            alpha_ex[1:] = alpha[0][ref_i][name]
            plt.plot(alpha_ex, ecp_ex, ref_colors[ref_i])
            
    
    row = 0
    column = 0
    for i, names in enumerate(draws2d.keys()):
    
        row+=1
        if row >= rows:
            column+=1
            row = 1+column
            
        fig_DRP.add_subplot(rows, rows, rows*row+column+1)
        plt.plot([0,1],[0,1], 'k:')
        plt.xlabel("Credibility level (alpha)")
        plt.ylabel("Expected Coverage Probability (ECP)")
        
        for ref_i in range(len(references_2d)):
        
            ecp_ex = np.zeros(len(ecp[1][ref_i][names])+1)
            alpha_ex = np.zeros(len(alpha[1][ref_i][names])+1)
            ecp_ex[1:] = ecp[1][ref_i][names]
            alpha_ex[1:] = alpha[1][ref_i][names]
            plt.plot(alpha_ex, ecp_ex, ref_colors[ref_i])

    
    return fig_DRP, figs_cov

























