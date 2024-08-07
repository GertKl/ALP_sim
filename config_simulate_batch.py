#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:10:57 2024

@author: gert
"""



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
    
    
    f = open(args.path+"/simulate_batch.sh", "w")
    
    f.write("#!/bin/bash")
    f.write("\n\n")
    f.write("\n\n")
    
    if on_cluster in ["fox"]:
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
    
    

    
        
    f.write("python "+args.path+"/simulate_batch.py -path "+args.path)
    

    
    f.write("\n\n")
    f.write("\n\n")

    f.write("exit")
    
    
    f.close()
