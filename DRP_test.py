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
from matplotlib import pyplot as plt

__all__ = ("get_drp_coverage",)


def get_drp_coverage(
    samples: np.ndarray,
    theta: np.ndarray,
    references: Union[str, np.ndarray] = "random",
    metric: str = "euclidean",
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

    # Normalize
    low = np.min(theta, axis=1, keepdims=True)
    high = np.max(theta, axis=1, keepdims=True)
    samples = (samples - low) / (high - low + 1e-10)
    theta = (theta - low) / (high - low + 1e-10)

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

    

    print("samples_distances_shape: " + str(samples_distances.shape))
    print("theta_distances_shape: " + str(theta_distances.shape))

    n_figs = min(50,samples.shape[1])
    plot_margs = [0,1]                              #Must be 2-Dimensional!
    figures = list(np.zeros(n_figs))
    axes = list(np.zeros(n_figs))                 
    for i in range(len(figures)):
        figures[i], axes[i] = plt.subplots()
        n_inside=0
        for j in range(len(samples[:,i,plot_margs[0]])):
            if samples_distances[j,i] < theta_distances[0,i]:
                plt.plot(samples[j,i,plot_margs[0]],samples[j,i,plot_margs[1]],'cx')               
                n_inside += 1
            else:
                plt.plot(samples[j,i,plot_margs[0]],samples[j,i,plot_margs[1]],'bx')
        # plt.plot(theta[0,i,plot_margs[0]], theta[0,i,plot_margs[1]],'ro')
        plt.plot(samples[0,i,plot_margs[0]],samples[0,i,plot_margs[1]],'ro') 
        plt.plot(references[i,plot_margs[0]], references[i,plot_margs[1]],'g+')
        circle = plt.Circle((references[i,plot_margs[0]],references[i,plot_margs[1]]), theta_distances[0,i], fill=0, color='r')
        axes[i].add_patch(circle)
        axes[i].set_xlim([-0.1, 1.1])
        axes[i].set_ylim([-0.1, 1.1])
        axes[i].set_aspect('equal', adjustable='box')
        plt.plot([0, 0],[0, 1], ':' , c='0.5')
        plt.plot([0, 1],[0, 0], ':' , c='0.5')
        plt.plot([0, 1],[1, 1], ':' , c='0.5')
        plt.plot([1, 1],[0, 1], ':' , c='0.5')
        # plt.text(-5,60, 'Samples in $\Theta_{{DRP}}$: {:.0f}/{:.0f} = {:.0f}%'.format(n_inside,len(samples[:,i,plot_margs[0]]), 100*n_inside/len(samples[:,i,plot_margs[0]])))
        plt.title('Samples in $\Theta_{{DRP}}$: {:.0f}/{:.0f} = {:.0f}%'.format(n_inside,len(samples[:,i,plot_margs[0]]), 100*n_inside/len(samples[:,i,plot_margs[0]])))

    # Compute coverage
    f = np.sum((samples_distances < theta_distances), axis=0) / num_samples

    # Compute expected coverage
    h, alpha = np.histogram(f, density=True, bins=num_sims // 10)
    dx = alpha[1] - alpha[0]
    ecp = np.cumsum(h) * dx
    return ecp, alpha[1:], f, figures




































