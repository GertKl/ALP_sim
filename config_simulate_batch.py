#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:10:57 2024

@author: gert
"""



import os
import sys
import numpy as np

# ALP_file_dir = os.path.dirname(os.getcwd())+"/analysis_scripts/ALP_sim"  

# if ALP_file_dir: sys.path.append(ALP_file_dir)   #!!! Change path to location of differential_counts.py and ALP_sim.py
from ALP_quick_sim import ALP_sim
from alp_swyft_simulator import ALP_SWYFT_Simulator

import scipy.stats as scist

import copy

import pickle

import os

import pickle

with open('config_objects.pickle', 'rb') as file:
    config_objects = pickle.load(file)
    
for key in config_objects.keys():
    locals()[key] = config_objects[key]


if __name__ == "__main__":

    # f = open(config.dirc + "/cluster_runs/results/" + config.name_run_ext + "/" + "train.sh", "w")
    
    f = open(os.getcwd()+"/simulate_batch.sh", "w")
    
    f.write("#!/bin/bash")
    f.write("\n\n")
    f.write("\n\n")
    
    if slurm:
        f.write("#SBATCH --job-name=ALP_simulations")
        f.write("\n")
        f.write("#SBATCH --account="+account)
        f.write("\n")
        f.write("#SBATCH --time="+max_time_sim)
        f.write("\n")
        f.write("#SBATCH --partition="+partition_sim)
        f.write("\n")
        f.write("#SBATCH --mem-per-cpu="+str(max_memory_sim)+"G")
        f.write("\n")
        if devel_sim:
            f.write("#SBATCH --qos=devel")
            f.write("\n")
        # if gpus:
        #     f.write("#SBATCH --gpus="+str(gpus))
        #     f.write("\n")
        f.write("#SBATCH --output="+os.getcwd()+"/sim_output.out")
        f.write("\n\n")
        
        f.write("\n\n")
        f.write("set -o errexit  # Exit the script on any error")
        f.write("\n\n")
        f.write("#set -o nounset  # Treat any unset variables as an error")
        f.write("\n\n")
        f.write("module --quiet purge  # Reset the modules to the system default")
        f.write("\n\n")
        f.write("# Set the ${PS1} (needed in the source of the Anaconda environment)")
        f.write("\n\n")
        f.write("export PS1=\$")
        f.write("\n\n")
        f.write("module load Miniconda3/22.11.1-1")
        f.write("\n\n")
        f.write("source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh")
        f.write("\n\n")
        f.write("conda deactivate &>/dev/null")
        f.write("\n\n")
        f.write("conda activate /fp/homes01/u01/ec-gertwk/.conda/envs/"+str(conda_env))
        f.write("\n\n")
        f.write("\n\n")
    
    # else:
        # f.write("\n\n")
        # f.write("conda activate "+str(conda_env))
        # f.write("\n\n")
        
    f.write("python simulate_batch.py")
    
    f.write("\n\n")
    f.write("\n\n")

    f.write("exit")
    
    
    f.close()
