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



class Network(swyft.SwyftModule):
    def __init__(self, nbins, marginals, param_names):
        super().__init__()
        self.marginals = marginals
        self.norm = swyft.networks.OnlineStandardizingLayer(torch.Size([nbins]), epsilon=0)
        self.logratios = swyft.LogRatioEstimator_1dim(
            num_features = nbins, 
            num_params = len(marginals), 
            varnames = param_names)
        self.learning_rate = 0.0000005
    
    def forward(self, A, B):
        data = self.norm(A['data'])
        return self.logratios(data, B['params'][:,self.marginals])
    
    
    
    
    
    
    
    
    