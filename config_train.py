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


    
    if on_cluster:
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
        # f.write("#SBATCH --cpus-per-task=1")
        f.write("\n")
        if devel_train:
            f.write("#SBATCH --qos=devel")
            f.write("\n")
        if gpus:
            f.write("#SBATCH --gpus="+str(gpus))
            f.write("\n")
        f.write("#SBATCH --output="+args.path+"/train_output/train_outputs/slurm-%j.out \\")
        f.write("\n\n")
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
    else:
        f.write("conda activate "+str(conda_env))
        f.write("\n\n")
        f.write("\n\n")
        
        

    f.write("\n\n")
    
    # if on_cluster:
        # f.write("\n running_states=(\\")
        # for running_state in running_states:
        #     f.write("\n \""+running_state+"\" \\")
        # f.write("\n )")
        # f.write("\n")
        # f.write("\n stopping_states=(\\")
        # for stopping_state in stopping_states:
        #     f.write("\n \""+stopping_state+"\" \\")
        # f.write("\n )")    
        # f.write("\n")
        # f.write("\n job_ids=()")
    
    # f.write("\n for ((j=1;j<="+str(1)+";j++)) ; do")
    if on_cluster:
        
        # f.write("\n echo Training in progress. Run squeue -u \" $USER\" to see status.")
        f.write("\n python "+results_dir+"/train.py -path "+results_dir)
        
        # f.write("\n \t")
        # f.write("\n \t job_id=$( sbatch \\")
        # f.write("\n \t --output="+args.path+"/train_output/train_outputs/slurm-%j.out \\")
        # f.write("\n \t python "+results_dir+"/train.py -path "+results_dir+" \\")
        # f.write("\n \t | awk '{print $4}')")
        # f.write("\n \t")
        # f.write("\n \t job_ids+=(\"$job_id\")")
        # f.write("\n")
        # f.write("\n done")
        
        # f.write("\n\n")
        # f.write("\n echo Training in progress. Run squeue -u \" $USER\" to see status. ")
        # f.write("\n continue=1")
        # f.write("\n while [[ $continue == 1 ]] ; do")
        # f.write("\n \t sleep 5")
        # f.write("\n \t continue=0")
        # f.write("\n \t for job_id in \"${job_ids[@]}\" ; do")
        # f.write("\n \t \t state_str=$( sacct --noheader --format=State --jobs=$job_id )")
        # f.write("\n \t \t IFS=' ' read -ra state <<< $state_str")
        # f.write("\n \t \t for running_state in \"${running_states[@]}\" ; do")
        # f.write("\n \t \t \t if [[ $state == $running_state ]] ; then")
        # f.write("\n \t \t \t \t continue=1")
        # f.write("\n \t \t \t \t break")
        # f.write("\n \t \t \t fi")
        # f.write("\n \t \t done")
        # f.write("\n\n")
        # f.write("\n \t \t if [[ $continue == 0 ]] ; then")
        # f.write("\n \t \t \t recognized=0")
        # f.write("\n \t \t \t for stopping_state in \"${stopping_states[@]}\" ; do")
        # f.write("\n \t \t \t \t if [[ $state == $stopping_state ]] ; then")
        # f.write("\n \t \t \t \t \t recognized=1")
        # f.write("\n \t \t \t \t fi")
        # f.write("\n \t \t \t done")
        # f.write("\n")
        # f.write("\n \t \t \t if [[ $recognized == 0 ]] ; then")
        # f.write("\n \t \t \t \t continue=1")
        # f.write("\n \t \t \t \t echo WARNING: job $job_id is in unrecognized state: $state. \
        #         Training will not abort by itself while this persists. run \" scancel $job_id \" if \
        #         you want to abort the process.")
        # f.write("\n \t \t \t fi")
        # f.write("\n \t \t fi")
        # f.write("\n \t done")
        # f.write("\n done")
        # f.write("\n echo Finished training!")
        # f.write("\n\n")
        if gpus:
            f.write("\n")
            f.write("\n# After computation stop monitoring")
            f.write("\n")
            f.write("kill -SIGINT \"$NVIDIA_MONITOR_PID\"")
    
    else:
        f.write(". python "+results_dir+"/train.py -path "+results_dir)
        f.write("\n")
        f.write("done")
        f.write("\n\n")
        f.write("echo Finished training!")
        
    f.write("\n\n")
    f.write("\n\n")
    f.write("\n\n")
    f.write("exit")
    
    
    f.close()

