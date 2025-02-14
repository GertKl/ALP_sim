#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:52:53 2024

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



# class Network(swyft.SwyftModule):
#     def __init__(self, nbins, marginals, param_names):
#         super().__init__()
#         self.marginals = marginals
#         self.norm = swyft.networks.OnlineStandardizingLayer(torch.Size([nbins]), epsilon=0)
#         if isinstance(marginals,list):
#             num_params = len(marginals) 
#         elif isinstance(marginals, int):
#             num_params = 1
#         else:
#             raise TypeError("POI-indices should be list or integer!")
#         self.logratios = swyft.LogRatioEstimator_1dim(
#             num_features = nbins,
#             num_params=num_params,
#             varnames = param_names)
#         self.learning_rate = 0.0000005
    
#     def forward(self, A, B):
#         data = self.norm(A['data'])
#         return self.logratios(data, B['params'][:,self.marginals])
    
    
    
# class Network(swyft.AdamWReduceLROnPlateau,swyft.SwyftModule):
#     def __init__(self,nbins, marginals, param_names):
#         super().__init__()
        
#         self.marginals=marginals
        
#         self.norm_data = swyft.networks.OnlineStandardizingLayer(torch.Size([nbins]), epsilon=0)
        
#         self.learning_rate = 5e-2
    
#         self.net_data = torch.nn.Sequential(
#             torch.nn.Flatten(),
#             torch.nn.LazyLinear(256),
#             torch.nn.ReLU(),
#             torch.nn.LazyLinear(256),
#             torch.nn.ReLU(),
#             torch.nn.LazyLinear(4)
#         )

#         self.logratios = swyft.LogRatioEstimator_1dim(
#             num_features = 4, 
#             num_params = len(marginals), 
#             varnames = param_names
#             )
         
#     def forward(self, A, B):
#         data = self.norm_data(A['data'])
#         data = self.net_data(data)

#         return self.logratios(data, B['params'][:,self.marginals])
    



class Network(swyft.AdamWReduceLROnPlateau, swyft.SwyftModule):
    
    def __init__(self, nbins, marginals, param_names, **kwargs):
        super().__init__()
        for key in kwargs.keys():
            globals()[key] = kwargs[key]
        

        self.norm = swyft.networks.OnlineStandardizingLayer(torch.Size([nbins]), epsilon=0)
        
        # INSERT OTHER LAYERS HERE
        
        self.logratio_features = nbins
        self.standard_inits(marginals, param_names, **kwargs)
        
    
    def head_net(self, A):
           
        data = self.norm(A['data'])
            
        #INSERT OTHER LAYERS HERE
        
        return data
    

        
    def standard_inits(self, marginals, param_names, **kwargs): #features=64, blocks=2, lr=5e-2, stopping_patience=5, dropout=0):
            
           
            self.marginals=marginals
            self.param_names = param_names
            self.features = features
            self.blocks = blocks
            self.dropout = dropout
            
            self.learning_rate = learning_rate
            self.early_stopping_patience=stopping_patience
            
            self.logratios = swyft.LogRatioEstimator_1dim(
                num_features = self.logratio_features,
                hidden_features=self.features,
                num_blocks=self.blocks,
                dropout=self.dropout,
                num_params = len(self.marginals), 
                varnames = list(np.array(self.param_names)[self.marginals])
            )
            
            if len(marginals)>1:
                marginals_2d = []
                for i, el in enumerate(self.marginals[:-1]):
                    for j in np.arange(i+1,len(self.marginals)):
                        marginals_2d.append( (el,self.marginals[j]) )
                marginals_2d = tuple(marginals_2d) 
                self.logratios2 = swyft.LogRatioEstimator_Ndim(
                    num_features = self.logratio_features,
                    hidden_features=self.features,
                    num_blocks=self.blocks,
                    dropout=self.dropout,
                    marginals = marginals_2d, 
                    varnames = [[np.array(self.param_names)[self.marginals][i] for i in marginal] for marginal in marginals_2d ]
                )

class Network1D(Network):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.logratios2=None
    def forward(self, A, B): 
        data = self.head_net(A)
        return self.logratios(data, B['params'][:,self.marginals])
    
class Network2D(Network):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.logratios=None
    def forward(self, A, B): 
        data = self.head_net(A)
        return self.logratios2(data, B['params'][:,self.marginals])
            
class NetworkCorner(Network):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs) 
    def forward(self, A, B):
        data = self.head_net(A)
        return self.logratios(data, B['params'][:,self.marginals]), self.logratios2(data, B['params'][:,self.marginals])       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
