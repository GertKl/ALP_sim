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
    
    f = open(os.getcwd()+"/simulate.sh", "w")
    
    f.write("#!/bin/bash")
    f.write("\n\n")
    f.write("\n\n")
    # f.write("#SBATCH --job-name=swyftanalysis")
    # f.write("\n")
    # f.write("#SBATCH --account="+config.account)
    # f.write("\n")
    # f.write("#SBATCH --time="+config.max_time_2)
    # f.write("\n")
    # f.write("#SBATCH --partition="+config.partition_2)
    # f.write("\n")
    # f.write("#SBATCH --mem-per-cpu="+str(config.max_memory_2)+"G")
    # f.write("\n")
    # if config.devel_2:
    #     f.write("#SBATCH --qos=devel")
    #     f.write("\n")
    # if config.gpus:
    #     f.write("#SBATCH --gpus="+str(config.gpus))
    #     f.write("\n")
    # f.write("#SBATCH --output="+config.dirc+"/cluster_runs/results/" + config.name_run_ext + "/train_output/train_output.out")
    # f.write("\n\n")
    
    
    if slurm:
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
    #     f.write("conda activate "+str(conda_env))
    #     f.write("\n\n")
    #     f.write("\n\n")
    
    f.write("python config_simulate_batch.py")
    f.write("\n\n")
    f.write("chmod +x simulate_batch.sh")
    f.write("\n\n")
    f.write("\n\n")
    
    f.write("for ((i=1;i<="+str(n_cpus)+";i++))")
    f.write("\n")
    f.write("do")
    f.write("\n\n")
    if slurm:
        f.write("sbatch ./simulate_batch.sh -$i")
        f.write("\n\n")

    else:
        f.write(". ./simulate_batch.sh")
        f.write("\n\n")
    
    f.write("\n\n")
    f.write("done")
    
    f.write("\n\n")
    f.write("wait")
    f.write("\n\n")
    
    # if slurm:
    #     f.write("rsync -r "+slurm_dir+" "+start_dir)
    
    f.write("\n\n")
    f.write("\n\n")

    f.write("exit")
    
    
    f.close()
    
    
    
    
    
    
    
    
    
    
    
    
    