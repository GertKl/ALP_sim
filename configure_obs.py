#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:07:13 2024

@author: gert
"""

import sys
sys.path.append('/fp/homes01/u01/ec-gertwk/ALPs_with_SWYFT/analysis_scripts/ALP_sim')

from ALP_quick_sim import ALP_sim

A = ALP_sim(set_obs = 0, set_null = 0)

print('oh yeah')
A.configure_obs(
    nbins = 300,
    nbins_etrue = 900,
    emin = 6e1,
    emax = 3e3,
    livetime = 300,
    irf_file = "/fp/homes01/u01/ec-gertwk/ALPs_with_SWYFT/IRFs/CTA/Prod5-North-20deg-AverageAz-4LSTs09MSTs.180000s-v0.1.fits"
)
print('tada!')
print(A.nbins_etrue)
print('tudeloo')