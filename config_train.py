#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:22:12 2024

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

if __name__ == "__main__":



    with open('config_objects.pickle', 'rb') as file:
        config_objects = pickle.load(file)
        
    for key in config_objects.keys():
        locals()[key] = config_objects[key]



    f = open(os.getcwd() + "/train.sh", "w")
    
    f.write("#!/bin/bash")
    f.write("\n\n")
    f.write("\n\n")
    f.write("#SBATCH --job-name=swyft_training")
    f.write("\n")
    f.write("#SBATCH --account="+account)
    f.write("\n")
    f.write("#SBATCH --time="+max_time_train)
    f.write("\n")
    f.write("#SBATCH --partition="+partition_train)
    f.write("\n")
    f.write("#SBATCH --mem-per-cpu="+str(max_memory_train)+"G")
    f.write("\n")
    if devel_train:
        f.write("#SBATCH --qos=devel")
        f.write("\n")
    if gpus:
        f.write("#SBATCH --gpus="+str(gpus))
        f.write("\n")
    f.write("#SBATCH --output="+os.getcwd()+"/train_output.out")
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
    # f.write("\n\n")
    # f.write("\n\n")
    # f.write("# Setup monitoring")
    # f.write("\n\n")
    # f.write("nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv --loop=1 > \"gpu_util-$SLURM_JOB_ID.csv\" &")
    # f.write("\n\n")
    # f.write("NVIDIA_MONITOR_PID=$!  # Capture PID of monitoring process")
    # f.write("\n\n")
    # f.write("echo")
    # f.write("\n")
    # f.write("echo NVIDIA_MONITOR_PID: $NVIDIA_MONITOR_PID")    
    # f.write("\n")
    # f.write("echo")
    # f.write("\n\n")
    # f.write("echo Date and time before starting train.py: \n")
    # f.write("date")
    # f.write("\n\n")
    # f.write("echo")
    f.write("\n\n")
    f.write("python train.py")
    f.write("\n\n")
    # f.write("# After computation stop monitoring")
    # f.write("\n")
    # f.write("kill -SIGINT \"$NVIDIA_MONITOR_PID\"")
    f.write("\n\n")
    f.write("\n\n")
    f.write("exit")
    
    
    f.close()

