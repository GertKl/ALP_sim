#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:03:05 2022
@author: gert

Writes a file which contains the definition of the parameter-extention 
function, as well as a list of the names of the parameters of interest (PoIs). 
The former takes as input a (short) list of the values of the parameters that
are not predefined, and outputs a (longer) list of values of all parameters,
including the ones with pre-defined values.   


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




if __name__ == "__main__":

    

    with open('poi_objects.pickle', 'rb') as file:
        poi_objects = pickle.load(file)
            
    for key in poi_objects.keys():
        locals()[key] = poi_objects[key]



    print()
    print("Writing POI configuration function")


    
    #Defining strings that represent how each parameter will appear in 
    #the function definition. 
 
    params = list(np.zeros(len(model_parameter_vals)))
    n_nones = 0
    
    
    
    
    for i, val_i in enumerate(model_parameter_vals):
        if val_i: 
            params[i] = str(val_i)
        else: 
            params[i] = "params["+str(n_nones)+"]"
            n_nones += 1
       

       
            
    #Making a list of parameter names consisting only of the parameters of 
    #interest.    
        
    available_names = ["m","g","PWL_Amplitude", "PWL_spec_index", "E0", "Ecut", 
                       "rms_B", "e_density_norm", "e_density_norm_2", 
                       "cluster_extension", "e_density_1", "e_density_2", 
                       "e_density_3", "e_density_4", "B_scaling", 
                       "Max_turbulence", "min_turbulence", 
                       "turbulence_spec_index"]
    
    available_units = ["","","", "", "", "", 
                       "", "", "", 
                       "", "", "", 
                       "", "", "", 
                       "", "", 
                       ""]
    
    param_names = []
    param_units = []
    
    for j, val_j in enumerate(model_parameter_vals):
        if not val_j: 
            param_names.append(available_names[j])
            param_units.append(available_units[j])




    #Writing the function definition, and list of parameter names to file. 

    f = open("param_function.py", "w")
    
    f.write("#!/usr/bin/env python3")
    f.write("\n")
    f.write("# -*- coding: utf-8 -*-")
    f.write("\n\n")
    f.write("def param_function(params):")
    f.write("\n\n")
    f.write("\t full_par = [")
    f.write("\n")
    f.write("\t \t\t"+params[0]+", \t # mass m in neV")
    f.write("\n")
    f.write("\t \t\t"+params[1]+", \t # coupling constant g in 10^(-11) /GeV")
    f.write("\n\n")
    f.write("\t \t\t"+params[2]+", \t # Amplitude of power law, in TeV-1 cm-2 s-1")
    f.write("\n")
    f.write("\t \t\t"+params[3]+", \t # Spectral index of the PWL")
    f.write("\n\n")
    f.write("\t \t\t"+params[4]+", \t # Reference energy (?) E0, In GeV")
    f.write("\n")
    f.write("\t \t\t"+params[5]+", \t # Cut-off energy Ecut, in GeV")
    f.write("\n")
    f.write("\t \t\t"+params[6]+", \t # rms of B field, default = 10.")
    f.write("\n")
    f.write("\t \t\t"+params[7]+", \t # normalization of electron density, default = 39.")
    f.write("\n")
    f.write("\t \t\t"+params[8]+", \t # second normalization of electron density, see Churazov et al. 2003, Eq. 4, default = 4.05")
    f.write("\n")
    f.write("\t \t\t"+params[9]+", \t # extension of the cluster, default = 500.")
    f.write("\n")
    f.write("\t \t\t"+params[10]+", \t # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 80.")
    f.write("\n")
    f.write("\t \t\t"+params[11]+", \t # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 200.")
    f.write("\n")
    f.write("\t \t\t"+params[12]+", \t # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 1.2")
    f.write("\n")    
    f.write("\t \t\t"+params[13]+", \t # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 0.58")
    f.write("\n")
    f.write("\t \t\t"+params[14]+", \t # scaling of B-field with electron denstiy, default = 0.5")
    f.write("\n")
    f.write("\t \t\t"+params[15]+", \t # maximum turbulence scale in kpc^-1, taken from A2199 cool-core cluster, see Vacca et al. 2012, default = 0.18")
    f.write("\n")
    f.write("\t \t\t"+params[16]+", \t # minimum turbulence scale, taken from A2199 cool-core cluster, see Vacca et al. 2012, default = 9.")
    f.write("\n")
    f.write("\t \t\t"+params[17]+"  \t # turbulence spectral index, taken from A2199 cool-core cluster, see Vacca et al. 2012, default = -2.80")
    f.write("\n")
    f.write("\t \t"+"]")
    f.write("\n\n")
    f.write("\t return full_par")
    f.write("\n\n\n\n")
    f.write("param_names="+str(param_names))
    f.write("\n\n\n\n")
    f.write("param_units="+str(param_units))
    #f.truncate()
    
    f.close()


    
    print("Finished writing parameter-extension function")
    

