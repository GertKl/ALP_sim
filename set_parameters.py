#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:18:47 2024

@author: gert
"""


import argparse
import os
import shutil

      

excluded_analysis_files = ["notebooks", "__pycache__", ".gitignore", ".git"]
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    
    
    parser.add_argument("-env1", type=str)
    parser.add_argument("-env2", type=str)
    parser.add_argument("-env3", type=str)
    parser.add_argument("-env4", type=str)
    
    parser.add_argument("-tech1", type=str)
    parser.add_argument("-tech2", type=str)
    parser.add_argument("-tech3", type=str)
    parser.add_argument("-tech4", type=str)
    
    parser.add_argument("-mod1", type=str)
    parser.add_argument("-mod2", type=str)
    parser.add_argument("-mod3", type=str)
    
    parser.add_argument("-params1", type=str)
    parser.add_argument("-params2", type=str)
    
    parser.add_argument("-sim1", type=str)
    parser.add_argument("-sim2", type=str)
    parser.add_argument("-sim3", type=str)
    parser.add_argument("-sim4", type=str)
    parser.add_argument("-sim5", type=str)
    parser.add_argument("-sim6", type=str)
    parser.add_argument("-sim7", type=str)
    parser.add_argument("-sim8", type=str)
    
    parser.add_argument("-train1", type=str)
    parser.add_argument("-train2", type=str)
    parser.add_argument("-train3", type=str)
    parser.add_argument("-train4", type=str)
    parser.add_argument("-train5", type=str)
    parser.add_argument("-train6", type=str)
    parser.add_argument("-train7", type=str)
    parser.add_argument("-train8", type=str)
    parser.add_argument("-train9", type=str)
    parser.add_argument("-train10", type=str)
    parser.add_argument("-train11", type=str)
    parser.add_argument("-train12", type=str)
    parser.add_argument("-train13", type=str)
    
    parser.add_argument("-inf1", type=str)
    parser.add_argument("-inf2", type=str)
    parser.add_argument("-inf3", type=str)
    parser.add_argument("-inf4", type=str)
    parser.add_argument("-inf5", type=str)
    parser.add_argument("-inf6", type=str)
    parser.add_argument("-inf7", type=str)
    parser.add_argument("-inf8", type=str)
    parser.add_argument("-inf9", type=str)
    
    args = parser.parse_args()
    
    
    
    # Defining results directory path, and creating it if it doesn't already exist:
        
    results_dir = args.startdir + "/results/" + args.tech1
    if not os.path.exists(results_dir): 
        os.makedirs(results_dir)
    else:
        raise NotImplementedError("Not implemented double-runs yet")
    
    
    # Copying analysis_scripts to results folder
    
    analysis_scripts_loc = args.env3
    for item in os.listdir(analysis_scripts_loc ):
        if item not in excluded_analysis_files:
            shutil.copy(os.path.join(analysis_scripts_loc,item), 
                        os.path.join(results_dir,item))
            
    
    
    
    # Format marginals-specifying vector

    marg_temp = eval(args.mod2.replace(":",","))
    try:
        marg = list(marg_temp)
    except TypeError as Err:
        if isinstance(marg_temp, int):
            marg = [marg_temp]
        else:
            raise Err
    


    # Deconstructing model parameters-information
    
    param_info_vec = args.params2.replace(" ","").split("::")
    
    
    
    
    
    
    


    

    
    




