#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:35:44 2024

@author: gert
"""



import importlib
import pickle
from types import ModuleType


def import_file(
        path: str,
        ) -> ModuleType:
    
    '''
    Imports the python file at the specified path and returns it as a module. 
    
    Input:
        -  path:            Path of the python file. 

    Output:
        -  module:          Module
    '''
    
    module_name = path.split("/")[-1].split(".py")[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    globals()[module_name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(globals()[module_name])
    
    module = globals()[module_name]
    
    return module


def save_variables(
        variables: dict,        
        path: str,
        ) -> None:
    
    '''
    Saves variables (stored in a dict) to a pickle file. 
    
    Input:
        -  path:            Path of the pickle file. 
    '''

    with open(path,'wb') as file:
        pickle.dump(variables, file)










