#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:46:58 2024

@author: gert
"""

import numpy as np
from scipy.stats import norm, lognorm
import copy

def weight(exp,n_bins):
    x = np.linspace(-1,1,n_bins)
    return 0.5+0.5*np.cos(np.pi*np.sign(x)*np.abs(x)**exp)

def random_fitted_sine_references(
    parsed_samples,
    x_freq = None,
    y_freq = None,
    x_exponent = None,
    y_exponent = None,
    x_phase = None,
    y_phase = None,
    min_freq = 1,
    max_freq = 1.4,
    min_freq_gap = 0.15,
    min_exponent = 0.8,
    max_exponent = 1.5,
    min_exponent_gap = 0.3,
):
  
    try:
        samples = parsed_samples['data']
    except IndexError:
        samples = parsed_samples

    data_length = len(samples[0]) 
    start_of_range1 = np.random.randint(0,data_length)
    end_of_range1 = (start_of_range1 + int(data_length/2)) #np.random.randint( start_of_range1+int(data_length/3) , start_of_range1+int(2*data_length/3) )
    exp1 = np.random.uniform(0.5,2)
    exp2 = np.random.uniform(0.5,2)
    range1 = np.arange(start_of_range1,end_of_range1)%data_length
    range2 = np.arange(end_of_range1,end_of_range1+(start_of_range1-end_of_range1)%data_length)%data_length

    reordered_range1 = (np.concatenate([range1,range2])-int(data_length/2))%data_length
    reordered_range2 = (np.concatenate([range2,range1])-int(data_length/2))%data_length
    weights1 = weight(exp1,data_length)[reordered_range1]
    weights2 = weight(exp2,data_length)[reordered_range2]

    samples1 = samples.copy()
    samples2 = samples.copy()
    
    # samples1 = samples[:,range1]
    # samples2 = samples[:,range2]
 
    bin_mins1 = np.min(samples1, axis=0)
    bin_maxes1 = np.max(samples1, axis=0)
    sums_of_standardized_bins1 = np.sum( weights1*((samples1-bin_mins1)/(bin_maxes1-bin_mins1)) ,axis=1)
    sums_of_standardized_bins1 = np.where(np.isinf(sums_of_standardized_bins1),0,sums_of_standardized_bins1)
    max_sum1 = np.max(sums_of_standardized_bins1)
    min_sum1 = np.min(sums_of_standardized_bins1)
    standardized_sums1 = (sums_of_standardized_bins1-min_sum1)/(max_sum1-min_sum1)

    bin_mins2 = np.min(samples2, axis=0)
    bin_maxes2 = np.max(samples2, axis=0)
    sums_of_standardized_bins2 = np.sum( weights2*((samples2-bin_mins2)/(bin_maxes2-bin_mins2)) ,axis=1)
    sums_of_standardized_bins2 = np.where(np.isinf(sums_of_standardized_bins2),0,sums_of_standardized_bins2)
    max_sum2 = np.max(sums_of_standardized_bins2)
    min_sum2 = np.min(sums_of_standardized_bins2)
    standardized_sums2 = (sums_of_standardized_bins2-min_sum2)/(max_sum2-min_sum2)

    sorted_sums1 = np.sort(standardized_sums1)
    sorted_sums2 = np.sort(standardized_sums2)

    fitted_parameters1 = lognorm.fit(standardized_sums1)
    fitted_parameters2 = lognorm.fit(standardized_sums2)
    
    phase_function1 = lognorm(fitted_parameters1[0], fitted_parameters1[1], fitted_parameters1[2]).cdf
    phase_function2 = lognorm(fitted_parameters2[0], fitted_parameters2[1], fitted_parameters2[2]).cdf

    phase_for_check1 = phase_function1(sorted_sums1)
    phase_for_check2 = phase_function2(sorted_sums2)
    phase1 = phase_function1(standardized_sums1)
    phase2 = phase_function2(standardized_sums2)

    if x_phase is None: x_phase = np.random.uniform(0,1)
    if y_phase is None: y_phase = np.random.uniform(0,1)

    freq_range = max_freq-min_freq
    if x_freq is None: x_freq = np.random.uniform(min_freq,max_freq)
    if y_freq is None: y_freq = np.random.uniform(min(x_freq+min_freq_gap,max_freq)-min_freq, freq_range+max(x_freq-min_freq-min_freq_gap,0))%(freq_range) + min_freq
    
    if x_exponent is None: x_exponent = np.random.uniform(min_exponent,1) if x_freq > y_freq else np.random.uniform(1,max_exponent)
    if y_exponent is None: y_exponent = np.random.uniform(min_exponent,min(1,x_exponent-min_exponent_gap)) if x_freq < y_freq else np.random.uniform(max(1,x_exponent+min_exponent_gap),max_exponent)

    x_values_for_check = 0.5 + 0.5*np.sin(2*np.pi*(( x_freq*phase_for_check1)**x_exponent + x_phase ))
    y_values_for_check = 0.5 + 0.5*np.sin(2*np.pi*(( y_freq*phase_for_check2)**y_exponent + y_phase ))
    
    x_values = 0.5 + 0.5*np.sin(2*np.pi*(( x_freq*phase1)**x_exponent + x_phase ))
    y_values = 0.5 + 0.5*np.sin(2*np.pi*(( y_freq*phase2)**y_exponent + y_phase ))

    random_variables = {
        'range1':range1,
        'range2':range2,
        'x_freq':x_freq,
        'y_freq':y_freq,
        'x_exponent':x_exponent,
        'y_exponent':y_exponent,
        'x_phase':x_phase,
        'y_phase':y_phase,
        'sums1':standardized_sums1,
        'sums2':standardized_sums2,
        'sorted_sums1':sorted_sums1,
        'sorted_sums2':sorted_sums2,
        'fitted_parameters1': fitted_parameters1,
        'fitted_parameters2': fitted_parameters2,
    }
    

    return np.array([x_values,y_values]).transpose(), np.array([x_values_for_check,y_values_for_check]).transpose(), random_variables



class References():

    def __init__(
        self,
        samples_for_fit = None,
        x_freq = None,
        y_freq = None,
        x_exponent = None,
        y_exponent = None,
        x_phase = None,
        y_phase = None,
        min_freq = 1,
        max_freq = 1.4,
        min_freq_gap = 0.15,
        min_exponent = 0.8,
        max_exponent = 1.5,
        min_exponent_gap = 0.3,
        device = 'cpu',
    ):
        
        self.x_freq = x_freq
        self.y_freq = y_freq
        self.x_exponent = x_exponent
        self.y_exponent = y_exponent
        self.x_phase = x_phase
        self.y_phase = y_phase
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_freq_gap = min_freq_gap
        self.min_exponent = min_exponent
        self.max_exponent = max_exponent
        self.min_exponent_gap = min_exponent_gap

        self.fitted_parameters1 = None
        self.fitted_parameters2 = None
        self.phase_function1 = None
        self.phase_function2 = None
        if not samples_for_fit is None: self.fit_samples(samples_for_fit)


    def weight(self,exp,n_bins):
        x = np.linspace(-1,1,n_bins)
        return 0.5+0.5*np.cos(np.pi*np.sign(x)*np.abs(x)**exp)
    
    def standardized_sums(
        self,
        parsed_samples,
    ):
        try:
            samples = parsed_samples['data']
        except IndexError:
            samples = parsed_samples
     
        data_length = len(samples[0]) 
        start_of_range1 = np.random.randint(0,data_length)
        end_of_range1 = (start_of_range1 + int(data_length/2)) #np.random.randint( start_of_range1+int(data_length/3) , start_of_range1+int(2*data_length/3) )
        exp1 = np.random.uniform(0.5,2)
        exp2 = np.random.uniform(0.5,2)
        range1 = np.arange(start_of_range1,end_of_range1)%data_length
        range2 = np.arange(end_of_range1,end_of_range1+(start_of_range1-end_of_range1)%data_length)%data_length
    
        reordered_range1 = (np.concatenate([range1,range2])-int(data_length/2))%data_length
        reordered_range2 = (np.concatenate([range2,range1])-int(data_length/2))%data_length
        
        weights1 = self.weight(exp1,data_length)[reordered_range1]
        weights2 = self.weight(exp2,data_length)[reordered_range2]
    
        #>#
        samples1 = samples.copy()
        samples2 = samples.copy()
        #<#
    
        #>#
        bin_mins1 = np.min(samples1, axis=0)
        bin_maxes1 = np.max(samples1, axis=0)
        #<#
    
        #>#
        sums_of_standardized_bins1 = np.sum( weights1*((samples1-bin_mins1)/(bin_maxes1-bin_mins1)) ,axis=1)
        sums_of_standardized_bins1 = np.where(np.isinf(sums_of_standardized_bins1),0,sums_of_standardized_bins1)
        #<#
    
        max_sum1 = np.max(sums_of_standardized_bins1)
        min_sum1 = np.min(sums_of_standardized_bins1)
        standardized_sums1 = (sums_of_standardized_bins1-min_sum1)/(max_sum1-min_sum1)
    
        #>#
        bin_mins2 = np.min(samples2, axis=0)
        bin_maxes2 = np.max(samples2, axis=0)
        #<#
        
        #>#
        sums_of_standardized_bins2 = np.sum( weights2*((samples2-bin_mins2)/(bin_maxes2-bin_mins2)) ,axis=1)
        #<#
        
        sums_of_standardized_bins2 = np.where(np.isinf(sums_of_standardized_bins2),0,sums_of_standardized_bins2)
    
        max_sum2 = np.max(sums_of_standardized_bins2)
        min_sum2 = np.min(sums_of_standardized_bins2)
        standardized_sums2 = (sums_of_standardized_bins2-min_sum2)/(max_sum2-min_sum2)

        random_variables = {
            'range1':range1,
            'range2':range2,
        }

        return standardized_sums1, standardized_sums2, random_variables

    
    def fit_samples(
        self,
        standardized_sums1,
        standardized_sums2,
    ):
        print('Fitting samples...', flush=True, end='')
        fitted_parameters1 = lognorm.fit(standardized_sums1)
        fitted_parameters2 = lognorm.fit(standardized_sums2)
        print(' done.')
        
        phase_function1 = lognorm(fitted_parameters1[0], fitted_parameters1[1], fitted_parameters1[2]).cdf
        phase_function2 = lognorm(fitted_parameters2[0], fitted_parameters2[1], fitted_parameters2[2]).cdf
        
        self.fitted_parameters1 = fitted_parameters1
        self.fitted_parameters2 = fitted_parameters2

        self.phase_function1 = phase_function1
        self.phase_function2 = phase_function2
        

    def references2D(
        self,
        parsed_samples,
    ):
    
        standardized_sums1,standardized_sums2,rvs = self.standardized_sums(parsed_samples)
    
        #>#
        sorted_sums1 = np.sort(standardized_sums1)
        sorted_sums2 = np.sort(standardized_sums2)
        #<#
    
        #>#
        try:
            phase_for_check1 = self.phase_function1(sorted_sums1)
            phase_for_check2 = self.phase_function2(sorted_sums2)
            phase1 = self.phase_function1(standardized_sums1)
            phase2 = self.phase_function2(standardized_sums2)
        except Exception as Err:
            if self.phase_function1 is None or self.phase_function2 is None:
                self.fit_samples(standardized_sums1,standardized_sums1)
                phase_for_check1 = self.phase_function1(sorted_sums1)
                phase_for_check2 = self.phase_function2(sorted_sums2)
                phase1 = self.phase_function1(standardized_sums1)
                phase2 = self.phase_function2(standardized_sums2)
            else:
                raise Err 
        #<#

        x_freq = copy.copy(self.x_freq)
        y_freq = copy.copy(self.y_freq)
        x_exponent = copy.copy(self.x_exponent)
        y_exponent = copy.copy(self.y_exponent)
        x_phase = copy.copy(self.x_phase)
        y_phase = copy.copy(self.y_phase)
        min_freq = copy.copy(self.min_freq)
        max_freq = copy.copy(self.max_freq)
        min_freq_gap = copy.copy(self.min_freq_gap)
        min_exponent = copy.copy(self.min_exponent)
        max_exponent = copy.copy(self.max_exponent)
        min_exponent_gap = copy.copy(self.min_exponent_gap)

        #>#
        if x_phase is None: x_phase = np.random.uniform(0,1)
        if y_phase is None: y_phase = np.random.uniform(0,1)
    
        freq_range = max_freq-min_freq
        if x_freq is None: x_freq = np.random.uniform(min_freq,max_freq)
        if y_freq is None: y_freq = np.random.uniform(min(x_freq+min_freq_gap,max_freq)-min_freq, freq_range+max(x_freq-min_freq-min_freq_gap,0))%(freq_range) + min_freq
        
        if x_exponent is None: x_exponent = np.random.uniform(min_exponent,1) if x_freq > y_freq else np.random.uniform(1,max_exponent)
        if y_exponent is None: y_exponent = np.random.uniform(min_exponent,min(1,x_exponent-min_exponent_gap)) if x_freq < y_freq else np.random.uniform(max(1,x_exponent+min_exponent_gap),max_exponent)
    
        x_values_for_check = 0.5 + 0.5*np.sin(2*np.pi*(( x_freq*phase_for_check1)**x_exponent + x_phase ))
        y_values_for_check = 0.5 + 0.5*np.sin(2*np.pi*(( y_freq*phase_for_check2)**y_exponent + y_phase ))
        
        x_values = 0.5 + 0.5*np.sin(2*np.pi*(( x_freq*phase1)**x_exponent + x_phase ))
        y_values = 0.5 + 0.5*np.sin(2*np.pi*(( y_freq*phase2)**y_exponent + y_phase ))
        #<#
    
        random_variables = {
            'range1':rvs['range1'],
            'range2':rvs['range2'],
            'x_freq':x_freq,
            'y_freq':y_freq,
            'x_exponent':x_exponent,
            'y_exponent':y_exponent,
            'x_phase':x_phase,
            'y_phase':y_phase,
            'sums1':standardized_sums1,
            'sums2':standardized_sums2,
            'sorted_sums1':sorted_sums1,
            'sorted_sums2':sorted_sums2,
            'fitted_parameters1': self.fitted_parameters1,
            'fitted_parameters2': self.fitted_parameters2,
        }
        
        return np.array([x_values,y_values]).transpose(), np.array([x_values_for_check,y_values_for_check]).transpose(), random_variables
    