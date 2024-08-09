#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:04:50 2024

@author: gert
"""

import numpy as np
import matplotlib.pyplot as plt
import swyft
from tqdm.auto import tqdm
import copy
from multiprocessing import Pool

def _get_HDI_thresholds(x, cred_level=[0.68268, 0.95450, 0.99730]):
    x = x.flatten()
    x = np.sort(x)[::-1]  # Sort backwards
    total_mass = x.sum()
    enclosed_mass = np.cumsum(x)
    idx = [np.argmax(enclosed_mass >= total_mass * f) for f in cred_level]
    levels = np.array(x[idx])
    return levels


def make_contour_matrix(tup,):
    
    start=tup[0]
    stop=tup[1]
    n_prior_samples=tup[2]
    predictions=tup[3]
    limit_credibility=tup[4]
    bins=tup[5]
    param_names=tup[6]

    for i in tqdm(range(start,stop)): #range(n_limits): 
    
        
        predictions_i = copy.deepcopy(predictions)
    

        predictions_i[1].logratios = predictions[1].logratios[i*n_prior_samples:(i+1)*n_prior_samples]
        predictions_i[1].params = predictions[1].params[i*n_prior_samples:(i+1)*n_prior_samples]

        counts, _ = swyft.get_pdf(
            predictions_i[1],
            param_names,
            bins = bins,
        )

        if i==start:
            X,Y = np.meshgrid(np.linspace(0,counts.shape[0]-1,counts.shape[0]),np.linspace(0,counts.shape[1]-1,counts.shape[1]))
            matrix_total = np.zeros(X.shape)


        plt.figure('dummy')
        levels_limits=sorted(_get_HDI_thresholds(counts,cred_level=[0,limit_credibility]))
        limit_contour = plt.contourf(counts.T,levels=levels_limits)

        matrix_i = np.ones(X.shape)
        

        for collection in limit_contour.collections:
            for path in collection.get_paths():
                mask = path.contains_points(np.vstack((X.flatten(), Y.flatten())).T,radius=1e-9)
                mask = mask.reshape(X.shape)
                matrix_i[mask] = 0

        matrix_total += matrix_i
        plt.close('dummy')

    return matrix_total


def generate_expected_limits(samples,
                             prior_samples,
                             bounds,
                             net = None,  
                             trainer = None,
                             contour_matrix = None,
                             predictions = None,
                             ax=None,
                             limit_credibility=0.9973,
                             levels = [0.003,0.05,0.34,0.682,0.95,0.9973,1],
                             fill=True,
                             bins=50,
                             batch_size = 1024,
                             param_names = ['m','g'],
                             colors = ['r','#FFA500','y','g','b','k'],
                             alpha = 0.5,
                             alpha_variable = False,
                             n_cores = 1,
                            ):
    
    if isinstance(samples,int):
        n_limits = samples
    else:
        n_limits = len(samples)
        
    if isinstance(prior_samples, int):
        n_prior_samples = prior_samples
    else:
        n_prior_samples = len(prior_samples)

    
    if not np.any(predictions) and not np.any(contour_matrix):
        repeat = n_prior_samples // batch_size + (n_prior_samples % batch_size > 0)
        
        predictions = trainer.infer(
            net,
            samples.get_dataloader(batch_size=1,repeat=repeat),
            prior_samples.get_dataloader(batch_size=batch_size)
        )
        
    if contour_matrix is None:
      

        iterable = [( si*(n_limits//n_cores), 
                     (si+1)*(n_limits//n_cores) + (1-min(1,n_limits//n_cores))*(n_limits%n_cores),
                     n_prior_samples,
                     predictions,
                     limit_credibility,
                     bins,
                     param_names,
                    ) for si in range(n_cores) ]
        
        
        with Pool(n_cores) as pool:
            try:
                res = pool.map(make_contour_matrix,iterable,chunksize = 1,)
                pool.terminate()
            except Exception as err:
                pool.terminate()
                print(err)
        pool.close()

        matrix_total = np.sum(np.array(res), axis=0)
    
    else:
        matrix_total = contour_matrix

    if not ax:
        fig = plt.figure()
        fig.add_subplot(1,1,1)
        ax = fig.axes[0]
    
    for li in range(len(levels)-1):
        ax.contourf(matrix_total,
                    levels=[levels[li]*n_limits,levels[li+1]*n_limits],
                    extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
                    colors = colors[li],
                    alpha = (li+2)*alpha/len(levels) if alpha_variable else alpha,
                    )

    # plt.close('dummy')
    
    if not ax:
        return matrix_total, predictions, fig
    else:
        return matrix_total, predictions, ax








def make_contour_matrix_for_false_exclusions(tup,):
    
    start=tup[0]
    stop=tup[1]
    n_prior_samples=tup[2]
    predictions=tup[3]
    limit_credibility=tup[4]
    bins=tup[5]
    param_names=tup[6]

    for i in tqdm(range(start,stop)): #range(n_limits): 
    
        for pi in range(2):
            
            predictions_i = copy.deepcopy(predictions[pi])
            

            predictions_i[1].logratios = predictions[pi][1].logratios[i*n_prior_samples[pi]:(i+1)*n_prior_samples[pi]]
            predictions_i[1].params = predictions[pi][1].params[i*n_prior_samples[pi]:(i+1)*n_prior_samples[pi]]
    
            counts, _ = swyft.get_pdf(
                predictions_i[1],
                param_names,
                bins = bins,
            )
    
            if i==start and pi==0:
                X,Y = np.meshgrid(np.linspace(0,counts.shape[0]-1,counts.shape[0]),np.linspace(0,counts.shape[1]-1,counts.shape[1]))
                matrix_total = np.zeros(X.shape)
                matrices = [np.ones(X.shape),np.ones(X.shape)]
    
    
            plt.figure('dummy')
            levels_limits=sorted(_get_HDI_thresholds(counts,cred_level=[0,limit_credibility]))
            limit_contour = plt.contourf(counts.T,levels=levels_limits)
    

            for collection in limit_contour.collections:
                for path in collection.get_paths():
                    mask = path.contains_points(np.vstack((X.flatten(), Y.flatten())).T,radius=1e-9)
                    mask = mask.reshape(X.shape)
                    matrices[pi][mask] = 0 # 0 means not excluded
    
            matrix_total += np.logical_and(matrices[0]==1, matrices[1]==0).astype(int)
            
            plt.close('dummy')

    return matrix_total





def find_false_exclusions(samples,
                          prior_samples1,
                          prior_samples2,
                          bounds,
                          contour_matrix = None,
                          predictions1 = None,
                          predictions2 = None,
                          net1 = None,
                          net2 = None,
                          trainer = None,
                          ax=None,
                          limit_credibility=0.9973,
                          levels = [0.003,0.05,0.34,0.682,0.95,0.9973,1],
                          fill=True,
                          bins=50,
                          batch_size = 1024,
                          param_names = ['m','g'],
                          colors = ['r','#FFA500','y','g','b','k'],
                          alpha = 0.5,
                          alpha_variable = False,
                          n_cores=1,
                         ):
    
    n_prior_samples = [None, None]
    prior_samples = [prior_samples1,prior_samples2]
    nets = [net1,net2]
    predictions = [predictions1, predictions2]  

    if isinstance(samples,int):
        n_limits = samples
    else:
        n_limits = len(samples)

    for pi in range(2):
            
        if isinstance(prior_samples[pi], int):
            n_prior_samples[pi] = prior_samples[pi]
        else:
            n_prior_samples[pi] = len(prior_samples[pi])

    
        if not np.any(predictions[pi]) and not np.any(contour_matrix):
            repeat = n_prior_samples[pi] // batch_size + (n_prior_samples[pi] % batch_size > 0)
            
            predictions[pi] = trainer.infer(
                nets[pi],
                samples.get_dataloader(batch_size=1,repeat=repeat),
                prior_samples[pi].get_dataloader(batch_size=batch_size)
            )
        
    if contour_matrix is None:
        
        
        iterable = [( si*(n_limits//n_cores), 
                     (si+1)*(n_limits//n_cores) + (1-min(1,n_limits//n_cores))*(n_limits%n_cores),
                     n_prior_samples,
                     predictions,
                     limit_credibility,
                     bins,
                     param_names,
                    ) for si in range(n_cores) ]
    
        
        with Pool(n_cores) as pool:
            try:
                res = pool.map(make_contour_matrix_for_false_exclusions,iterable,chunksize = 1,)
                pool.terminate()
            except Exception as err:
                pool.terminate()
                print(err)
        pool.close()

        matrix_total = np.sum(np.array(res), axis=0)
    
    else:
        matrix_total = contour_matrix

    if not ax:
        fig = plt.figure()
        fig.add_subplot(1,1,1)
        ax = fig.axes[0]
    
    for li in range(len(levels)-1):
        ax.contourf(matrix_total,
                    levels=[levels[li]*n_limits,levels[li+1]*n_limits],
                    extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
                    colors = colors[li],
                    alpha = (li+2)*alpha/len(levels) if alpha_variable else alpha,
                    )

    # plt.close('dummy')
    
    if not ax:
        return matrix_total, predictions, fig
    else:
        return matrix_total, predictions , ax





























