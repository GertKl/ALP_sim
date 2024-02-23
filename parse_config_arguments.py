#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:38:57 2024

@author: gert
"""


import argparse
import os
import shutil

      

excluded_analysis_files = ["notebooks", "__pycache__", ".gitignore", ".git"]
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    
    
    parser.add_argument("-args", type=str)
    args = parser.parse_args()
    
    
    args_list = args.args.split(";") 
    
    print(args_list)