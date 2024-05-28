#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:40:54 2024

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




class Network(swyft.AdamWReduceLROnPlateau, swyft.SwyftModule):
    
    def __init__(self, nbins, marginals, param_names, **kwargs):
        super().__init__()
        for key in kwargs.keys():
            globals()[key] = kwargs[key]
        
        self.logratio_features = power_features + data_features
    
        self.norm_data = swyft.networks.OnlineStandardizingLayer(torch.Size([nbins]), epsilon=0)
        
        self.norm_power = swyft.networks.OnlineStandardizingLayer(torch.Size([169]), epsilon=0)
        
        self.net_data = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(64),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(64),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(data_features)
        )

        self.net_power = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(64),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(64),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(power_features)
        )
        
        self.standard_inits(marginals, param_names, **kwargs)
        
    
    def head_net(self, A):
        
        data = self.norm_data(A['data'])
        data = self.net_data(data)
        
        power = self.norm_power(A['power'])
        power = self.net_power(power)
        
        final_features = torch.cat((data,power), axis = 1)
         
        return final_features
    
        

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
    def __init__(self, nbins, marginals, param_names, features=64, blocks=2, lr=5e-2, stopping_patience=5, dropout=0):
        super().__init__(nbins, marginals, param_names, features=features, blocks=blocks, lr=lr, stopping_patience=stopping_patience,dropout=dropout)
        self.logratios2=None
    def forward(self, A, B): 
        data = self.head_net(A)
        return self.logratios(data, B['params'][:,self.marginals])
    
class Network2D(Network):
    def __init__(self, nbins, marginals, param_names, features=64, blocks=2, lr=5e-2, stopping_patience=5, dropout=0):
        super().__init__(nbins, marginals, param_names, features=features, blocks=blocks, lr=lr, stopping_patience=stopping_patience, dropout=dropout)
        self.logratios=None
    def forward(self, A, B): 
        data = self.head_net(A)
        return self.logratios2(data, B['params'][:,self.marginals])
            
class NetworkCorner(Network):
    def __init__(self,*args,**kwargs):#self, nbins, marginals, param_names, features=64, blocks=2, lr=5e-2, stopping_patience=5, dropout=0):
        super().__init__(*args,**kwargs) #nbins, marginals, param_names, features=features, blocks=blocks, lr=lr, stopping_patience=stopping_patience, dropout=dropout)
    def forward(self, A, B):
        data = self.head_net(A)
        return self.logratios(data, B['params'][:,self.marginals]), self.logratios2(data, B['params'][:,self.marginals])       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
