#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:53:03 2024

@author: gert
"""


import argparse


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-path", type=str)
    args = parser.parse_args()
    
    
    f = open(args.path+"/set_parameters.sh", "w")
    
    f.write("#!/bin/bash")

    f.write("\n\n")
    f.write("\nIFS=';' read -ra config_arguments <<< $1")
    f.write("\nfor config_argument in \"${config_arguments[@]}\"")
    f.write("\ndo")
    f.write("\n\tIFS='=' read -ra config_argument_name_and_value <<< $config_argument")
    f.write("\n\tdeclare \"${config_argument_name_and_value[0]}\"=\"${config_argument_name_and_value[1]// /}\"")
    f.write("\ndone")
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
    f.write("conda activate /fp/homes01/u01/ec-gertwk/.conda/envs/$conda_env")
    f.write("\n\n")

    f.write("\n\n")
    f.write("\npython $results_dir/set_parameters.py -args \"$1\"")
    f.write("\n\n")
    f.write("exit")
    
    
    f.close()





