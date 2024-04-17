#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:22:12 2024

@author: gert
"""

import os
import argparse
import pickle

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-path", type=str)
    args = parser.parse_args()
    
    with open(args.path +'/config_variables.pickle', 'rb') as file:
        config_dict = pickle.load(file)
    for key in config_dict.keys():
        locals()[key] = config_dict[key]
    
    
    if not os.path.exists(args.path+"/train_output/train_outputs"):
        os.makedirs(args.path+"/train_output/train_outputs")



    f = open(args.path+"/train.sh", "w")
    
    f.write("#!/bin/bash")
    f.write("\n\n")
    f.write("\n\n")



    if on_cluster in ["fox"]:
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
        if devel_sim:
            f.write("#SBATCH --qos=devel")
            f.write("\n")
        if gpus:
            f.write("#SBATCH --gpus="+str(gpus))
            f.write("\n")
        f.write("#SBATCH --output="+args.path+"/train_output/train_outputs/slurm-%j.out \\")
        
        
    f.write("\n\n")
    if on_cluster in ["fox", "hepp"]:
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
        if on_cluster in ["fox"]:
            f.write("module load Miniconda3/22.11.1-1")
        elif on_cluster in ["hepp"]:
            f.write("module load Miniconda3/4.9.2")  
        f.write("\n\n")
        f.write("source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh")
        f.write("\n\n")
        f.write("conda deactivate &>/dev/null")
        f.write("\n\n")
        
    if on_cluster in ["fox"]:
        f.write("conda activate /fp/homes01/u01/ec-gertwk/.conda/envs/"+str(conda_env))
    elif on_cluster in ["hepp","local"]:
        f.write("conda activate "+str(conda_env))
    else:
        raise ValueError("Cluster \""+on_cluster+"\" not recognized")
    f.write("\n\n")
        
    if gpus:
        f.write("\n\n")
        f.write("export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:24258'")
        f.write("\n\n")
        f.write("# Setup monitoring")
        f.write("\n\n")
        f.write("nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv --loop=1 > \""+results_dir+"/train_output/train_outputs/gpu_util-$SLURM_JOB_ID.csv\" &")
        f.write("\n\n")
        f.write("NVIDIA_MONITOR_PID=$!  # Capture PID of monitoring process")
        f.write("\n\n")
        f.write("echo")
        f.write("\n")
        f.write("echo NVIDIA_MONITOR_PID: $NVIDIA_MONITOR_PID")    
        f.write("\n")
        f.write("echo")
        f.write("\n\n")
        f.write("echo Date and time before starting train.py: \n")
        f.write("date")
        f.write("\n\n")
        f.write("echo")

    f.write("\n\n")
    
    if on_cluster in ["fox"]:
        f.write("\n python "+results_dir+"/train.py -path "+results_dir)
        if gpus:
            f.write("\n")
            f.write("\n# After computation stop monitoring")
            f.write("\n")
            f.write("kill -SIGINT \"$NVIDIA_MONITOR_PID\"")
            f.write("\n")
    
    elif on_cluster in ["hepp","local"]:
        f.write("\n python "+results_dir+"/train.py -path "+results_dir)
        
    else:
        raise ValueError("Cluster \""+on_cluster+"\" not recognized")


    f.write("\n\n")
    f.write("echo Finished training!")
    f.write("\n\n")
    f.write("\n\n")
    f.write("\n\n")
    f.write("exit")
    
    
    f.close()

