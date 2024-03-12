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
    
    
    
    
class Network(swyft.AdamWReduceLROnPlateau,swyft.SwyftModule):
    def __init__(self,nbins, marginals, param_names):
        super().__init__()
        
        self.marginals=marginals
        
        self.norm_data = swyft.networks.OnlineStandardizingLayer(torch.Size([nbins]), epsilon=0)
        
        self.learning_rate = 5e-2
    
        self.net_data = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(256),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(256),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(4)
        )

        self.logratios = swyft.LogRatioEstimator_1dim(
            num_features = 4, 
            num_params = len(marginals), 
            varnames = param_names
            )
         
    def forward(self, A, B):
        data = self.norm_data(A['data'])
        data = self.net_data(data)

        return self.logratios(data, B['params'][:,self.marginals])
    
    
    