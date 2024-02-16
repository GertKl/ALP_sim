#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:31:30 2024

@author: gert
"""

import numpy as np
import random
from ALP_quick_sim import ALP_sim
import swyft
import scipy.stats as scist
import copy



# class ALP_SWYFT_Simulator(swyft.Simulator):
#     def __init__(self, A, bounds=None):
#         super().__init__()

#         self.transform_samples = swyft.to_numpy32
        
#         self.A = copy.deepcopy(A)
#         self.bounds = bounds

#         self.samplers = []
#         for i,bound in enumerate(bounds):
#             self.samplers.append(scist.uniform(loc=bound[0], scale=bound[1]-bound[0]))

#         #random.seed()
        
#     def sample_prior(self,):
#         #np.random.seed(random.randint(0,2**32-1))
#         param_sample = [sampler.rvs() for sampler in self.samplers]
#         return np.array(param_sample).astype(np.float32)

#     def generate_exp(self,vec):
#         exp = self.A.simulate(vec)['y']
#         return exp.astype(np.float32)

#     def generate_data(self,exp,params):
#         data = self.A.noise({'y':exp},params)['y']
#         return data.astype(np.float32)

#     # def simulate_store_parallel(self, n_sims_per_cpu):
#     #     return store.simulate(self, max_sims=n_sims_per_cpu, batch_size=chunk_size)

#     # def simulate

#     def build(self, graph):
#         params = graph.node('params', self.sample_prior)
#         exp = graph.node('exp', self.generate_exp, params)
#         data = graph.node('data', self.generate_data,exp,params)
        
        
class ALP_SWYFT_Simulator(swyft.Simulator):
    def __init__(self, A, bounds=None):
        super().__init__()

        self.transform_samples = swyft.to_numpy32
        
        self.A = copy.deepcopy(A)
        self.bounds = bounds

        # self.samplers = []
        # for i,bound in enumerate(bounds):
        #     self.samplers.append(scist.uniform(loc=bound[0], scale=bound[1]-bound[0]))

        #random.seed()
        
    def sample_prior(self,):
        #np.random.seed(random.randint(0,2**32-1))
        param_sample = [random.uniform(bound[0],bound[1]) for bound in self.bounds]
        return np.array(param_sample).astype(np.float32)

    def generate_exp(self,vec):
        exp = self.A.simulate(vec)['y']
        return exp.astype(np.float32)

    def generate_data(self,exp,params):
        data = self.A.noise({'y':exp},params)['y']
        return data.astype(np.float32)


    # def generate_power(self,data):
    #     power = abs(np.fft.fft(counts,n=len_fft))[...,:len_fts]
    #     return power.astype(np.float32)

    # def simulate_store_parallel(self, n_sims_per_cpu):
    #     return store.simulate(self, max_sims=n_sims_per_cpu, batch_size=chunk_size)

    # def simulate

    def build(self, graph):
        params = graph.node('params', self.sample_prior)
        exp = graph.node('exp', self.generate_exp, params)
        data = graph.node('data', self.generate_data,exp,params)
        # power = graph.node('power', self.generate_power,data)
        
        
        
        
        
        