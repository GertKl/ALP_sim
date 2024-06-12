#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:10:57 2024

@author: gert
"""


import os
import argparse
import pickle


    
filename_variables = "config_variables.pickle"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-path", type=str)
    args = parser.parse_args()
        
    # loading config parameters
    with open(args.path+'/' +filename_variables, 'rb') as file:
        config_dict = pickle.load(file)
    for key in config_dict.keys():
        locals()[key] = config_dict[key]
    
    
    if not os.path.exists(args.path+"/sim_output/sim_outputs"):
        os.makedirs(args.path+"/sim_output/sim_outputs")
    
    f = open(args.path+"/simulate.sh", "w")
    
    f.write("#!/bin/bash")
    f.write("\n\n")
    f.write("\n\n")
   
    if on_cluster in ["fox", "hepp"]:
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
    f.write("\n\n")
    
    
    f.write("\n python "+results_dir+"/config_simulate_batch.py -path "+results_dir)
    f.write("\n")
    f.write("\n chmod +x "+results_dir+"/simulate_batch.sh")
    f.write("\n")
    
    
    if on_cluster in ["fox"]:
        f.write("\n IFS=',' read -ra running_states <<< \"$1\"")
        f.write("\n IFS=',' read -ra stopping_states <<< \"$2\"")  
        f.write("\n")
        f.write("\n job_ids=()")
        f.write("\n for ((j=1;j<="+str(n_jobs_sim)+";j++)) ; do")
        f.write("\n \t")
        f.write("\n \t job_id=$( sbatch \\")
        f.write("\n \t --output="+args.path+"/sim_output/sim_outputs/slurm-%j.out \\")
        f.write("\n \t "+results_dir+"/simulate_batch.sh \\")
        f.write("\n \t | awk '{print $4}')")
        f.write("\n \t")
        f.write("\n \t job_ids+=(\"$job_id\")")
        f.write("\n")
        f.write("\n done")
        
        f.write("\n\n")
        f.write("\n echo Simulation in progress. Run squeue -u \"$USER\" to see status. ")
        f.write("\n for job_id in \"${job_ids[@]}\" ; do")
        f.write("\n \t sleep 0.05")
        f.write("\n \t continue=1")
        f.write("\n \t while [[ $continue == 1 ]] ; do")
        # f.write("\n \t sleep "+str(10+int(n_jobs_sim/10.))+"")
        f.write("\n \t \t continue=0")
        # f.write("\n \t \t echo checking status of job $job_id")
        f.write("\n \t \t state_str=$( sacct --noheader --format=State --jobs=$job_id )")
        f.write("\n \t \t IFS=' ' read -ra state <<< $state_str")
        f.write("\n \t \t for running_state in \"${running_states[@]}\" ; do")
        # f.write("\n \t \t \t echo running state: $running_state")
        f.write("\n \t \t \t if [[ $state == $running_state ]] ; then")
        f.write("\n \t \t \t \t continue=1")
        f.write("\n \t \t \t \t break")
        f.write("\n \t \t \t fi")
        f.write("\n \t \t done")
        f.write("\n\n")
        f.write("\n \t \t if [[ $continue == 0 ]] ; then")
        f.write("\n \t \t \t recognized=0")
        f.write("\n \t \t \t for stopping_state in \"${stopping_states[@]}\" ; do")
        # f.write("\n \t \t \t \t echo stopping state: $stopping_state")
        f.write("\n \t \t \t \t if [[ $state == $stopping_state ]] ; then")
        f.write("\n \t \t \t \t \t recognized=1")
        f.write("\n \t \t \t \t fi")
        f.write("\n \t \t \t done")
        f.write("\n")
        f.write("\n \t \t \t if [[ $recognized == 0 ]] ; then")
        f.write("\n \t \t \t \t continue=1")
        f.write("\n \t \t \t \t echo WARNING: job $job_id is in unrecognized state: $state. \
                Simulation will not abort by itself while this persists. run \" scancel $job_id \" if \
                you want to abort the process.")
        f.write("\n \t \t \t fi")
        f.write("\n \t \t fi")
        f.write("\n \t \t if [[ $continue == 1 ]] ; then")
        f.write("\n \t \t \t sleep 60")
        f.write("\n \t \t fi")
        f.write("\n \t done")
        f.write("\n done")

    elif on_cluster in ["hepp","local"]:
        f.write("\n . "+results_dir+"/simulate_batch.sh")
        
        
        
        # f.write("\n")
        # f.write("\n job_ids=()")
        # f.write("\n for ((j=1;j<="+str(int(n_jobs_sim))+";j++)) ; do")
        # f.write("\n \t")
        # # f.write("\n \t echo yaaaaay")
        # f.write("\n \t")
        # f.write("\n \t . "+results_dir+"/simulate_batch.sh &")
        # f.write("\n \t")
        # f.write("\n")
        # f.write("\n done")
        # f.write("\n")
        # f.write("\n wait")
        
        
        
    
    else:
        raise ValueError("Cluster \""+on_cluster+"\" not recognized")
    
    f.write("\n\n")
    f.write("\n echo Finished simulating!")
    f.write("\n\n")
    f.write("exit")
    
    f.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
