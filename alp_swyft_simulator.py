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



        
class ALP_SWYFT_Simulator(swyft.Simulator):
    def __init__(self, A, bounds=None, prior_funcs=None,max_freq=3, len_fft=2**11):
        super().__init__()

        self.transform_samples = swyft.to_numpy32
        
        self.A = copy.deepcopy(A)
        self.bounds = bounds
        
        if not prior_funcs:
            self.prior_funcs = []
            for i in range(len(bounds)):
                self.prior_funcs.append(random.uniform)
        else:            
            self.prior_funcs = []
            for j, info in enumerate(prior_funcs):
                which_distr = info.split('(')[0]
                try:
                    distr_params = eval(info.split(which_distr)[1])
                except SyntaxError as Err:
                    distr_params = None
                    
                if which_distr == 'U':
                    # self.prior_funcs.append(random.uniform)
                    func = lambda a,b : scist.uniform.rvs(loc=a, scale =b-a)
                    self.prior_funcs.append( func ) 
                elif info[0] == 'N':
                    mu = distr_params[0]
                    std = distr_params[1]
                    func = lambda a,b : scist.truncnorm.rvs( (a-mu)/std, (b-mu)/std, loc=mu, scale=std,size=1)[0]
                    self.prior_funcs.append( func ) 
                else:
                    raise ValueError('Invalid prior distribution specified! Can be U(), or N(x,y)')
            
            
            
        
        self.len_fft = len_fft
        self.len_fts = int(max_freq*((np.log10(self.A.emax)-np.log10(self.A.emin))/self.A.nbins)*len_fft)

        # self.samplers = []
        # for i,bound in enumerate(bounds):
        #     self.samplers.append(scist.uniform(loc=bound[0], scale=bound[1]-bound[0]))

        #random.seed()
        
    def sample_prior(self,):
        #np.random.seed(random.randint(0,2**32-1))
        param_sample = [self.prior_funcs[pi](bound[0],bound[1]) for pi, bound in enumerate(self.bounds)]
        return np.array(param_sample).astype(np.float32)

    def generate_exp(self,vec):
        exp = self.A.simulate(vec)['y']
        return exp.astype(np.float32)

    def generate_data(self,exp,params):
        data = self.A.noise({'y':exp},params)['y']
        return data.astype(np.float32)
    
    def generate_pgg(self,exp):
        pgg = self.A.pgg.copy()
        return pgg.astype(np.float32)

    def generate_power(self,data):
        power = abs(np.fft.fft(data,n=self.len_fft))[...,:self.len_fts]
        return power.astype(np.float32)


    # def simulate_store_parallel(self, n_sims_per_cpu):
    #     return store.simulate(self, max_sims=n_sims_per_cpu, batch_size=chunk_size)

    # def simulate

    def build(self, graph):
        params = graph.node('params', self.sample_prior)
        exp = graph.node('exp', self.generate_exp, params)
        data = graph.node('data', self.generate_data,exp,params)
        pgg = graph.node('pgg', self.generate_pgg,exp)
        power = graph.node('power', self.generate_power,data)
        
        
        
        
        
        