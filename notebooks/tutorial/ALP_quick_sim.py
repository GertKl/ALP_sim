#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:37:57 2023

@author: gert
"""


import numpy as np
import random

import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)

import logging
from typing import Union, Optional

import astropy.units as u
from pathlib import Path
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from astropy.table import Table

from gammapy.modeling import Fit
from gammapy.irf import load_cta_irfs
from gammapy.data import Observation
from gammapy.utils.random import get_random_state

# models modules
from gammapy.modeling.models import (
    Model,
    Models,
    SkyModel,
    PowerLawSpectralModel,
    PowerLawNormSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    PointSpatialModel,
    GaussianSpatialModel,
    TemplateSpatialModel,
    FoVBackgroundModel,
    SpectralModel,
    Parameter, 
    TemplateSpectralModel
)
# dataset modules
from gammapy.datasets import (
    MapDataset, 
    MapDatasetEventSampler,
    SpectrumDatasetOnOff,
    SpectrumDataset, 
    Datasets
)
from gammapy.maps import MapAxis, WcsGeom, Map, MapCoord
from gammapy.makers import MapDatasetMaker, SpectrumDatasetMaker

from gammaALPs.core import Source, ALP, ModuleList
from gammaALPs.base import environs, transfer

# from differential_counts import DifferentialCounts


 
    
    
class ALP_sim():
    
    ''' 
    A class to simulate ALP spectra in gamma-rays from NGC1275. Contains several versions of the
    physical model (see methods starting with "model"), including some toy models for testing, and 
    a noise function. Also streamlines computation of example-cases, and subsequent plotting of 
    spectra, see method compute_case(). 
    '''
    
    def __init__(self,
                 set_obs=1
                 ) -> None:
       
        ''' 
        Input:
            -  set_geom:            Can be set to 0, IF geom parameters are to be changed before 
                                    simulation, to save time.  
        '''
        
        # Model parameters. See method set_model_params().  
        self.params = [0, 0]
        
        # Geom configuration parameters. See method configure_geom().
        self.emin = 10      #GeV
        self.emax = 1e5     #GeV 
        self.nbins = 50 
        self.nbins_etrue = 150
        self.pointing = [150.58,-13.26]
        self.livetime = 250 * u.hr 
        self.irf_file = "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
        
        # Model configuration parameters. See method configure_model().
        self.with_bkg = True
        self.with_signal = True
        self.with_bkg_model = True
        self.with_signal_model = True
        self.with_edisp = True
        self.nB = 1
        self.ALP_seed = None

        # Plot configuration parameters. See method configure_plot().
        self.ymin = None
        self.ymax = None
        self.xmin = None
        self.xmax = None 
        self.figsize = (9,5)
        self.fontsize = 15
        self.dnde = 0
        self.ax=None
        self.fig=None
        self.ax_survival=None
        self.fig_survival=None
         
        # Observed counts, in general computed in class instance. But can also be imported, see
        # method import_obs(). 
        self.counts_obs = None

        # Parameter expansion function. See method full_params_default.  
        self.full_param_vec = ALP_sim.full_params_default
        
        # Intermediate results, not to be changed by user. 
        self.counts_exp = None
        self.pgg = None
        self.pgg_EBL = None
        self.pgg_combined = None
        self.bin_centers = None
        self.bin_widths = None
 
        
        if set_obs: self.set_obs()
    

    
    
    def configure_obs(self,
                        signal: Union[bool,None]="_empty",
                        edisp: Union[bool,None]="_empty",
                        bkg: Union[bool,None]="_empty",
                        emin: Union[float,None]="_empty",
                        emax: Union[float,None]="_empty",
                        nbins: Union[int,None]="_empty",
                        nbins_etrue: Union[int,None]="_empty",
                        pointing: Union[list[float],None]="_empty",
                        livetime: Union[float,None]="_empty",
                        irf_file: Union[str,None]="_empty"
                        ) -> None:
        
        ''' 
        Sets the input parameters to the model (parameters of interest and nuisance parameters).
        Can have any dimension up to the number of model parameters, but must be adapted to 
        parameter expansion function, see method full_params_default(). Parameters that are not
        specified in funciton call stay unchanged.
        
        Input:
            -  signal:              Whether or not to include gamma-ray excess in simulation.
            -  edisp:               Whether or not to include energy dispersion in simulation.
            -  bkg:                 Whether or not to include cosmic-ray background in simulation.
            -  emin:                Minimum spectrum energy [GeV].
            -  emax:                Maximum spectrum energy [GeV].
            -  nbins:               Number of bins in spectrum.
            -  nbins_etrue:         Number of bins for which to compute ALP apsorption.
            -  pointing:            2D list, icrs coordinates of target, in degrees.
            -  livetime:            Effective observation time [hours].
            -  irf_file:            Path of IRF file to use.
        '''
        
        if bkg != "_empty": self.with_bkg = bkg
        if signal != "_empty": self.with_signal = signal
        if edisp != "_empty": self.with_edisp = edisp
        if emin != "_empty": self.emin = emin
        if emax != "_empty": self.emax = emax
        if nbins != "_empty": self.nbins = nbins
        if nbins_etrue != "_empty": self.nbins_etrue = nbins_etrue
        if pointing != "_empty": self.pointing = pointing
        if livetime != "_empty": self.livetime = livetime * u.hr
        if irf_file != "_empty": self.irf_file = irf_file
        
        self.set_obs()
        
             
    
    def configure_model(self,
                        params: Union[list[float], None]="_empty",
                        bkg: Union[bool,None]="_empty",
                        signal: Union[bool,None]="_empty",
                        nB: Union[int,None]="_empty",
                        ALP_seed="_empty"
                        ) -> None:
        ''' 
        Sets model parameters physical and non-physical. Parameters that are not
        specified in funciton call stay unchanged.
   
        Input:
            -  params:              List of model parameters (parameters of interest and nuisance 
                                    parameters), for use in example simulations (see method 
                                    compute_case). Can have any dimension up to the number of model 
                                    parameters, but must be adapted to parameter expansion function, 
                                    see method full_params_default(). 
            -  bkg:                 Whether or not to include cosmic-ray background in simulation.
                                    Changing this here (as opposed to in method configure_obs) is
                                    generally faster.
            -  signal:              Whether or not to include excess gamma-rays in simulation.
                                    Changing this here (as opposed to in method configure_obs) is
                                    generally faster.
            -  edisp:               Whether or not to apply energy dispersion.
            -  nB:                  [Redundant] Number of B-field realizations to compute
            -  ALP_seed:            Seed for random B-field realizations. Set to other than None for 
                                    reproduction of same realization (?).
        '''
        
        if params != "_empty": self.params = params
        if bkg != "_empty": self.with_bkg_model = bkg
        if signal != "_empty": self.with_signal_model = signal
        if nB != "_empty": self.nB = nB
        if ALP_seed != "_empty": self.ALP_seed = ALP_seed  
       
        
    def configure_plot(self,
                        xmin: Union[float,None]="_empty",
                        xmax: Union[float,None]="_empty",
                        ymin: Union[float,None]="_empty",
                        ymax: Union[float,None]="_empty",
                        figsize: Union[tuple[int,int],None]="_empty",
                        fontsize: Union[float,None]="_empty",
                        dnde: Union[bool,None]="_empty"
                        ):
        
        ''' 
        Sets plot parameters that are unlikely to be changed between consecutive calls of method
        compute_case().Parameters that are not specified in funciton call stay unchanged.
        
        Input:
            -  xmin:                Minimum x value of plot, in given unit.
            -  xmax:                Maximum x value of plot, in given unit.
            -  ymin:                Minimum y value of plot, in given unit.
            -  ymax:                Maximum y value of plot, in given unit.
            -  Figsize:             2D tuple (width, height) of figure.  
            -  Fontsize:            Fontsize of title and axis labels.
            -  dnde:                Plots in terms of counts if 0, and in terms of differential 
                                    counts wrt energy if 1.
        '''
        
        if xmin != "_empty": self.xmin = xmin
        if xmax != "_empty": self.xmax = xmax
        if ymin != "_empty": self.ymin = ymin
        if ymax != "_empty": self.ymax = ymax
        if figsize != "_empty": self.figsize = figsize
        if fontsize != "_empty": self.fontsize = fontsize
        if dnde != "_empty": self.dnde = dnde
        
    
    def import_obs(self,
                   obs: Union[list,np.ndarray,dict[str,np.ndarray]]=None,
                   exp: Union[list,np.ndarray,dict[str,np.ndarray]]=None
                   ) -> None:
        
        ''' 
        Input:
            -  obs:              Vector of observed counts (within dict), e.g. from earlier simulations.
            -  exp:              Vector of expected counts (within dict), e.g. from earlier simulations.
        '''
        
            
        if isinstance(obs, np.ndarray):
            counts_obs = obs
            if len(counts_obs) == self.nbins:
                self.counts_obs = counts_obs
            else:
                raise ValueError("Imported observed counts should have same length as self.nbins")
        elif isinstance(obs, list):
            counts_obs = np.array(obs)
            if len(counts_obs) == self.nbins: 
                self.counts_obs = counts_obs
            else:
                raise ValueError("Imported observed counts should have same length as self.nbins")
        elif isinstance(obs, dict):
            try:
                counts_obs = obs['y']
                if len(counts_obs) == self.nbins: 
                    self.counts_obs = counts_obs
                else:
                    raise ValueError("Imported observed counts should have same length as self.nbins")
            except KeyError:
                raise KeyError("The observed counts should be a list, numpy.ndarray, or dictionary of \
                               the form {'y': numpy.ndarray}")
        else:
            if obs == None:
                print("No observed counts specified for import, keeping any old ones")
            else:
                raise TypeError("The observed counts should be a list, numpy.ndarray, or dictionary of \
                               the form {'y': numpy.ndarray}")
            
                               
        if isinstance(exp, np.ndarray):
            counts_exp = exp
            if len(counts_exp) == self.nbins:
                self.counts_exp = counts_exp
            else:
                raise ValueError("Imported observed counts should have same length as self.nbins")
        elif isinstance(exp, list):
            counts_exp = np.array(exp)
            if len(counts_exp) == self.nbins:
                self.counts_exp = counts_exp
            else:
                raise ValueError("Imported observed counts should have same length as self.nbins")
        elif isinstance(exp, dict):
            try:
                counts_exp = exp['y']
                if len(counts_exp) == self.nbins:
                    self.counts_exp = counts_exp
                else:
                    raise ValueError("Imported observed counts should have same length as self.nbins")
            except KeyError:
                raise KeyError("The expected counts should be a list, numpy.ndarray, or dictionary of \
                               the form {'y': numpy.ndarray}")
        else:
            if exp == None:
                print("No expected counts specified for import, keeping any old ones")
            else:
                raise TypeError("The expected counts should be a list, numpy.ndarray, or dictionary of \
                               the form {'y': numpy.ndarray}")
        





        
    
    def set_obs(self):
        
        '''
        Sets geometry of observations, to be used for generation of fake gamma- and cosmic-ray data.
        '''
        
        logging.disable(logging.WARNING)
        emin_TeV = str(self.emin/1000)
        emax_TeV = str(self.emax/1000)

        energy_axis      = MapAxis.from_energy_bounds( emin_TeV+" TeV", emax_TeV+" TeV", nbin=self.nbins, per_decade=False, name="energy" )
        
        if self.with_edisp:
            energy_axis_true = MapAxis.from_energy_bounds( emin_TeV+" TeV", emax_TeV+" TeV", nbin=self.nbins_etrue, per_decade=False, name="energy_true")
        else:
            energy_axis_true = MapAxis.from_energy_bounds( emin_TeV+" TeV", emax_TeV+" TeV", nbin=self.nbins, per_decade=False, name="energy_true")
            
        migra_axis = MapAxis.from_bounds(0.5, 2, nbin=self.nbins_etrue, node_type="edges", name="migra")

        try:
            irfs     = load_cta_irfs(self.irf_file)
            self.point = SkyCoord(self.pointing[0], self.pointing[1], frame="icrs", unit="deg")
            self.observation = Observation.create(pointing=self.point, livetime=self.livetime, irfs=irfs)
        except:
            try:
                self.observation = Observation.read(self.irf_file)
                self.point = self.observation.pointing_radec
                if not isinstance(self.observation,Observation): 
                    raise TypeError("Loaded object is of type " + str(type(self.observation))+ ", but expected type Observation.")
            except Exception as e:
                print(e)
                raise IOError("Could not load IRF, neither as a CTA IRF, nor as a gammapy Observation.")
                
        

        geom       = WcsGeom.create(frame="icrs", skydir=self.point, width=(2, 2), binsz=0.02, axes=[energy_axis])
        d_empty = MapDataset.create(geom, energy_axis_true=energy_axis_true, migra_axis=migra_axis, name="my-dataset")


        available_irfs = []
        if 'aeff' in self.observation.available_irfs: available_irfs.append('exposure')
        if 'edisp' in self.observation.available_irfs and self.with_edisp: available_irfs.append('edisp')
        if 'bkg' in self.observation.available_irfs and self.with_bkg: available_irfs.append('background')

        maker   = MapDatasetMaker(selection=available_irfs)
        self.dataset = maker.run(d_empty, self.observation)
        
        bin_axis = 'energy' #if self.with_edisp else 'energy_true'
        
        self.bin_centers = np.array(self.dataset.geoms['geom'].axes[bin_axis].center)
        self.bin_widths = np.array(self.dataset.geoms['geom'].axes[bin_axis].bin_width)
        self.bin_centers = self.bin_centers*1000
        self.bin_widths = self.bin_widths*1000
        
        logging.disable(logging.NOTSET)

   
    @staticmethod
    def full_params_default(
                            params: list[float]
                            ) -> list[float]:
        
        ''' 
        The default parameter expansion function. The expansion function allows to flexibly choose 
        which model parameters are considered as input to the model method (e.g. self.model,
        self.model_log, etc. Toy models are not affected). For example, when this present default 
        function is used, the only inputs to the model methods are the values of mass and coupling, 
        i.e. a 2D list. If you wanted to make, for example, the rms of the B-field the third input 
        parameter, first copy self.full_params_default to a new function new_func(params), change 
        the value corresponding to the B-field RMS value to "params[2]", and then set 
        self.full_param_vec (see init method) to new_func. See also method full_params_spectral for
        a different example. The model methods will then expect a 3D list instead. When running the 
        model method, self.full_param_vec(params) is called, effectively expanding the model 
        parameters to the full list of 18.  
        
        
        Input:
            -  params:              Input parameters to model methods. 

        Output:
            -  full_par             Full list of all 18 model parameter values. 


        '''
        
        full_par = [
                    params[0],          # mass m in neV
                    params[1],          # coupling constant g in 10^(-11) /GeV
                    
                    5.75 * 1e-9,        # Amplitude of power law, in "TeV-1 cm-2 s-1" # 10e-6 
                    2.36859,            # Spectral index of the PWL
                    153.86,             # Reference energy (?) E0, In GeV
                    819.72,             #Cut-off energy Ecut, in GeV
                    
                    
                    10.,                # rms of B field, default = 10.
                    39.,                # normalization of electron density, default = 39.
                    4.05,               # second normalization of electron density, see Churazov et al. 2003, Eq. 4, default = 4.05
                    500.,               # extension of the cluster, default = 500.
                    80.,                # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 80.
                    200.,               # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 200.
                    1.2,                # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 1.2
                    0.58,               # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 0.58
                    0.5,                # scaling of B-field with electron denstiy, default = 0.5
                    0.18,               # maximum turbulence scale in kpc^-1, taken from A2199 cool-core cluster, see Vacca et al. 2012, default = 0.18
                    9.,                 # minimum turbulence scale, taken from A2199 cool-core cluster, see Vacca et al. 2012, default = 9.
                    -2.80               # turbulence spectral index, taken from A2199 cool-core cluster, see Vacca et al. 2012, default = -2.80
                    ]
    
        return full_par      
    
    
    @staticmethod
    def full_params_spectral(
                            params: list[float]
                            ) -> list[float]:
        
        ''' 
        Parameter expansion function for spectral fit (see method model_spectral). See method 
        full_params_default for more documentation.         
        
        Input:
            -  params:              Input parameters to model methods. 

        Output:
            -  full_par             Full list of all 18 model parameter values. 


        '''
        
        full_par = [
                    0,                  # mass m in neV
                    0,                  # coupling constant g in 10^(-11) /GeV
                    
                    params[0],          # Amplitude of power law, in "TeV-1 cm-2 s-1" # 10e-6 
                    params[1],          # Spectral index of the PWL
                    153.86,             # Reference energy (?) E0, In GeV
                    params[2],          # Cut-off energy Ecut, in GeV
                    
                    
                    10.,                # rms of B field, default = 10.
                    39.,                # normalization of electron density, default = 39.
                    4.05,               # second normalization of electron density, see Churazov et al. 2003, Eq. 4, default = 4.05
                    500.,               # extension of the cluster, default = 500.
                    80.,                # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 80.
                    200.,               # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 200.
                    1.2,                # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 1.2
                    0.58,               # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 0.58
                    0.5,                # scaling of B-field with electron denstiy, default = 0.5
                    0.18,               # maximum turbulence scale in kpc^-1, taken from A2199 cool-core cluster, see Vacca et al. 2012, default = 0.18
                    9.,                 # minimum turbulence scale, taken from A2199 cool-core cluster, see Vacca et al. 2012, default = 9.
                    -2.80               # turbulence spectral index, taken from A2199 cool-core cluster, see Vacca et al. 2012, default = -2.80
                    ]
    
        return full_par   
    
    
    @staticmethod
    def compute_ALP_absorption(modulelist, axion_mass, coupling, emin, emax, bins):
        '''
        Copied from Giacomo.
        
        Input:
            -  modulelist:     ModuleList object assuming a given source
            -  axion_mass:     axion mass / 1 neV
            -  coupling  :     axion-gamma coupling / 1e-11 GeV^-1
            -  emin      :     min energy / GeV
            -  emin      :     max energy / GeV
            -  bins      :     number of points in energy log-sperated
        Output:
            -  energy points
            -  gamma absorption for the above energy points

        '''
        ebins            = np.logspace(np.log10(emin),np.log10(emax),bins)

        modulelist.alp.m = axion_mass
        modulelist.alp.g = coupling
        modulelist.EGeV  = ebins

        px,  py,  pa     = modulelist.run(multiprocess=2)
        pgg              = px + py

        return modulelist.EGeV, pgg[0]
    

    def model(self, 
              params: list[float]
              ) -> dict[str,np.ndarray]:
        
        '''
        Function for simulated observed gamma-ray spectra, including background and ALP-mixing. See
        also methods configure_model and configure_geom.
        
        Input:
            -  params:          Model input paramaters (of interest, and nuisance). Dimensionality 
                                is determined by configuration of self.full_param_vec. 

        Output:
            -  out              Histogram of observed events, as function of energy, within a dict.
                                i.e. {'y': histogram}  
        '''
        
        logging.disable(logging.WARNING)
        
        v = self.full_param_vec(params)
        
        v[17] = -v[17]
        
        m = v[0] * 1e-9 * u.eV
        g = v[1] * 1e-11 * 1/u.GeV
        
        
        nbins = self.nbins #if self.with_edisp else self.nbins_etrue
        
        
        source     = Source(z = 0.017559, ra = '03h19m48.1s', dec = '+41d30m42s') # this is for ngc1275
        pin        = np.diag((1.,1.,0.)) * 0.5
        alp        = ALP(0,0) 
        modulelist_loc = ModuleList(alp, source, pin = pin)
        modulelist_loc.add_propagation("ICMGaussTurb", 
                      0, # position of module counted from the source. 
                      nsim = self.nB, # number of random B-field realizations
                      B0 = v[6],  # rms of B field, default = 10.
                      n0 = v[7],  # normalization of electron density, default = 39.
                      n2 = v[8], # second normalization of electron density, see Churazov et al. 2003, Eq. 4, default = 4.05
                      r_abell = v[9], # extension of the cluster, default = 500.
                      r_core = v[10],   # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 80.
                      r_core2 = v[11], # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 200.
                      beta = v[12],  # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 1.2
                      beta2= v[13], # electron density parameter, see Churazov et al. 2003, Eq. 4, default = 0.58
                      eta = v[14], # scaling of B-field with electron denstiy, default = 0.5
                      kL = v[15], # maximum turbulence scale in kpc^-1, taken from A2199 cool-core cluster, see Vacca et al. 2012, default = 0.18
                      kH = v[16],  # minimum turbulence scale, taken from A2199 cool-core cluster, see Vacca et al. 2012, default = 9.
                      q = v[17], # turbulence spectral index, taken from A2199 cool-core cluster, see Vacca et al. 2012, default = -2.80
                      seed=self.ALP_seed # random seed for reproducability, set to None for random seed., default = 0
                     )
        modulelist_loc.add_propagation("EBL",1, model = 'dominguez') # EBL attenuation comes second, after beam has left cluster
        modulelist_loc.add_propagation("GMF",2, model = 'jansson12', model_sum = 'ASS') # finally, the beam enters the Milky Way Field
            
        enpoints, pgg   = ALP_sim.compute_ALP_absorption(
                        modulelist = modulelist_loc, # modulelist from gammaALP
                        axion_mass = m, # neV
                        coupling   = g , # 10^(-11) /GeV
                        emin       = self.emin,  # Gev
                        emax       = self.emax,  # GeV
                        bins       = nbins) # log-bins in enrgy for which computing the ALP-absorption
        
        enpoints, pggEBL   = ALP_sim.compute_ALP_absorption(
                        modulelist = modulelist_loc, # modulelist from gammaALP
                        axion_mass = 0, # neV
                        coupling   = 0 , # 10^(-11) /GeV
                        emin       = self.emin,  # Gev
                        emax       = self.emax,  # GeV
                        bins       = nbins) # log-bins in enrgy for which computing the ALP-absorption
        
        
        self.enpoints_pgg = enpoints
        self.pgg = pgg.copy()
        self.pgg_EBL = pggEBL.copy()
              
        self.pgg_combined = pgg.copy()
        
        
        absorption = TemplateSpectralModel(enpoints*u.Unit("GeV"), pgg, interp_kwargs={"method":"linear"})
        
        exp_pwl = ExpCutoffPowerLawSpectralModel(
            amplitude=v[2] * u.Unit("TeV-1 cm-2 s-1"),  # 5.75e-9 * u.Unit("TeV-1 cm-2 s-1"),
            reference=v[4] *1e-3 * u.TeV   ,            # 0.15386 * u.TeV,
            index=v[3]     ,                            # 2.36859,
            lambda_= 1/(v[5]*1e-3) * u.Unit("TeV-1")    # 1.22 * u.Unit("TeV-1"),
        )
        
        exp_pwl.parameters["amplitude"].min = 0
        exp_pwl.parameters["index"].min = 0
        exp_pwl.parameters["lambda_"].min = 0
        
        
        spectral_model      =  absorption * exp_pwl
        
        point_str = self.point.to_string().split(' ')
        point_units = self.point.info.unit.split(',')
        point_str_with_units = [point_str[0] + " " + point_units[0], point_str[1] + " " + point_units[1]]
        
        spatial_model_point = PointSpatialModel(lon_0=point_str_with_units[0], lat_0=point_str_with_units[1], frame="icrs")
        sky_model_pntpwl    = SkyModel(spectral_model=spectral_model,spatial_model=spatial_model_point, name="point-pwl")
        bkg_model           = FoVBackgroundModel(dataset_name="my-dataset")
        
        # finally we combine source and bkg models
        models = Models( [sky_model_pntpwl,bkg_model] )
        self.dataset.models = models

        if self.with_bkg_model:
            if not 'bkg' in self.observation.available_irfs:
                raise ValueError("Predicted background counts cannot be requested when background IRF is not available. Check if IRF file contains background, or if the background IRF has been deactivated using the method configure_obs.")


        if (self.with_signal_model and self.with_signal) and self.with_bkg_model:
            counts = np.array(self.dataset.npred().get_spectrum().data)[:,0,0]
        elif not (self.with_signal_model and self.with_signal) and self.with_bkg_model:
            counts = np.array(self.dataset.npred_background().get_spectrum().data)[:,0,0]
        elif (self.with_signal_model and self.with_signal) and not self.with_bkg_model:
            counts = np.array(self.dataset.npred_signal().get_spectrum().data)[:,0,0]
        else:
            counts = np.array(self.dataset.npred_signal().get_spectrum().data)[:,0,0]
            counts = counts - counts


        logging.disable(logging.NOTSET)  
        
        
        #counts = np.where(counts<1e-2,1e-2,counts)
        
 
        out = dict(y=np.array(counts))
        
     
        return out
    
    
    def model_log(self, 
                  params: list[float]
                  ) -> dict[str,np.ndarray]:
            
        '''
        Function for simulated observed gamma-ray spectra (without noise), taking log of input 
        values. Otherwise the same as method model. 
        
        Input:
            -  params:          Model input paramaters (of interest, and nuisance). Dimensionality 
                                is determined by configuration of self.full_param_vec. 

        Output:
            -  out              Histogram of observed events, as function of energy, within a dict.
                                i.e. {'y': histogram}       
        '''
        
        new_v = params.copy()
   
        for i, param in enumerate(new_v):  
            new_v[i] = 10**param
        
        out = self.model(new_v)
        
        return out


    def model_spectral_fit(self, 
                            params: list[float]
                            ) -> list[float]:
                
        '''
        Function for simulated observed gamma-ray spectra, not including ALP_mixing (and without 
        noise). Otherwise the same as method model. For this to work, the most natural thing to do 
        is to set self.full_param_vec to method full_params_spectral (see init function, and the 
        named method). 
        
        Input:
            -  params:          Model input paramaters (of interest, and nuisance). Dimensionality 
                                is determined by configuration of self.full_param_vec. 

        Output:
            -  out              Histogram of observed events, as function of energy, within a dict.
                                i.e. {'y': histogram}       
        '''
        
        logging.disable(logging.WARNING)
        
        v = self.full_param_vec(params)
        
        exp_pwl = ExpCutoffPowerLawSpectralModel(
            amplitude=v[2] * u.Unit("TeV-1 cm-2 s-1"),  # 5.75e-9 * u.Unit("TeV-1 cm-2 s-1"),
            reference=v[4] *1e-3 * u.TeV   ,             # 0.15386 * u.TeV,
            index=v[3]     ,                      # 2.36859,
            lambda_= 1/(v[5]*1e-3) * u.Unit("TeV-1")    # 1.22 * u.Unit("TeV-1"),
        )
        
        exp_pwl.parameters["amplitude"].min = 0
        exp_pwl.parameters["index"].min = 0
        exp_pwl.parameters["lambda_"].min = 0
        
        
        spectral_model      =  exp_pwl
        # spectral_model      =  absorption * exp_pwl
        
        
        spatial_model_point = PointSpatialModel(lon_0="150.58 deg", lat_0="-13.26 deg", frame="icrs")
        sky_model_pntpwl    = SkyModel(spectral_model=spectral_model,spatial_model=spatial_model_point, name="point-pwl")
        bkg_model           = FoVBackgroundModel(dataset_name="my-dataset")
        
        # finally we combine source and bkg models
        models = Models( [sky_model_pntpwl,bkg_model] )
        self.dataset.models = models

        if self.with_bkg_model:
            if not 'bkg' in self.observation.available_irfs:
                raise ValueError("Predicted background counts cannot be requested when background IRF is not available. Check if IRF file contains background, or if the background IRF has been deactivated using the method configure_obs.")


        if (self.with_signal_model and self.with_signal) and self.with_bkg_model:
            counts               = np.array(self.dataset.npred().get_spectrum().data)[:,0,0]
        elif not (self.with_signal_model and self.with_signal) and self.with_bkg_model:
            counts               = np.array(self.dataset.npred_background().get_spectrum().data)[:,0,0]
        elif (self.with_signal_model and self.with_signal) and not self.with_bkg_model:
            counts               = np.array(self.dataset.npred_signal().get_spectrum().data)[:,0,0]
        else:
            counts               = np.array(self.dataset.npred_signal().get_spectrum().data)[:,0,0]
            counts = counts - counts


        logging.disable(logging.NOTSET)  
        
        
        counts = np.where(counts<1e-2,1e-2,counts)
        
 
        out = dict(y=np.array(counts))
        
     
        return out
    

    def model_spectral_fit_log(self,
                               params: list[float]
                               ) -> list[float]:       
        '''
        Function for simulated observed gamma-ray spectra, without ALP_mixing (and without noise), 
        and taking log of input values. Otherwise the same as method model_spectral_fit. 
        
        Input:
            -  params:          Model input paramaters (of interest, and nuisance). Dimensionality 
                                is determined by configuration of self.full_param_vec (see init 
                                method). 

        Output:
            -  out              Histogram of observed events, as function of energy, within a dict.
                                i.e. {'y': histogram}       
        '''
        
        
        new_v = params.copy()
        
        for i, param in enumerate(new_v): 
            new_v[i] = 10**param

        out = self.model_spectral_fit(new_v)
        
        return out
    
    
    def model_toy_line(self,
                       params: list[float]
                       ) -> list[float]: 
        
        '''
        Function for simulated toy spectrum of the form data=ax+b (without noise), where x is the 
        bin-number. Takes as input two parameters [a, b], independent of how self.full_param_vec is 
        configured. 
        
        Input:
            -  params:          List of two model input paramaters [a, b]. If input list has higher 
                                dimension than 2, all elements beyond the second are ignored.

        Output:
            -  out              Linear histogra, as function of bin-number, within a dict.
                                i.e. {'y': histogram}       
        '''
        
        
        v = self.full_param_vec(params)
        
        m = v[0] 
        g = v[1] 
        
        out = dict(y=np.arange(0,1, 1/self.nbins)*m + g)
    
        return out
    
    
    def model_toy_sine(self,
                       params: list[float]
                       ) -> list[float]: 
        
        '''
        Function for simulated toy spectrum of the form data= (bx+d)sin(ax+c) + ex + f + b (with 
        some calibration constants),without noise, where x is the bin-number. Takes as input six 
        parameters [a, b, c, d, e, f], independent of how self.full_param_vec is configured. 
        
        Input:
            -  params:          List of six model input paramaters [a, b, c, d, e, f]. If input list 
                                has higher dimension than 6, all elements beyond the second are 
                                ignored.

        Output:
            -  out              Linear histogra, as function of bin-number, within a dict.
                                i.e. {'y': histogram}       
        '''
  
        v = params
        
        m = v[0] 
        g = v[1]*0.01
        
        x = np.arange(0,8, 8/self.nbins)
        
        return dict(y= 100*(-v[4]*x + 8*v[4] +  v[5] + g*8.1  + (-g*x + g*8.1 + v[3])*np.sin(m*x + v[2])))
        

    def compute_case(self,
                     new_fig: bool=True,
                     new_counts: bool=True, 
                     plot_obs: bool=True, 
                     plot_exp: bool=True,
                     plot_survival: bool=False,
                     model: str="", 
                     color: str='k', 
                     color_obs: str='k', 
                     linestyle: str="-", 
                     legend: bool=True) -> None:
        
        '''
        Function for convenient simulation of example-spectra and consecutive visualization.  
        
        Input:
            -  new_fig:         Creates a new figure if True. Adds to existing figure if False.
                                Saved in self.fig. 
            -  new_counts:      Computes new observations and expected values if True. Otherwise,
                                plots using existing values, stored in self.counts_obs and 
                                self.counts_exp. 
            -  plot_obs:        Whether or not to plot observations.
            -  plot_exp:        Whether or not to plot expected values.
            -  plot_survival:   Whether or not to visualize the photon survival probability 
                                (separate figure, placed in self.fig_survival). 
            -  model:           Which model to simulate from. Options: "", "log", "spectral_fit",
                                "spectral_fit_log", "toy_line", "toy_sine".
            -  color:           Plot color for expected values. 
            -  color_obs:       Plot color for observed values. 
            -  linestyle:       Linestyle for expected values. 
            -  Legend:          Whether or not to include a legend. 

        Output:
            -  out              Linear histogra, as function of bin-number, within a dict.
                                i.e. {'y': histogram}       
        '''        
        
        
        '''
        TODO: Adapt parameter values and units in figure labels to the specific model
        '''
        
        mod = model
        if model != "": mod = "_"+model
        
        mod_func = eval("self.model"+mod)
        
         
        if new_counts: 
            self.counts_exp = mod_func(self.params)['y']
            self.counts_obs = self.noise({'y':self.counts_exp}, self.params)['y']

        
        if self.dnde: 
            counts_exp_plot = self.counts_exp/self.bin_widths
            counts_obs_plot = self.counts_obs/self.bin_widths
        else: 
            counts_exp_plot = self.counts_exp.copy()
            counts_obs_plot = self.counts_obs.copy()
        
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        if not xmin: xmin=min(self.bin_centers[counts_obs_plot != 0])*0.5
        if not xmax: xmax=max(self.bin_centers[counts_obs_plot != 0])*1.5
        if not ymin: ymin=min(counts_obs_plot[counts_obs_plot != 0])*0.5
        if not ymax: ymax=max(counts_obs_plot[counts_obs_plot != 0])*1.5
          

        if (plot_obs or plot_exp):  
            if new_fig or not self.fig:  
                self.fig, self.ax = plt.subplots(figsize=self.figsize)
                self.ax.grid(True,which='both',linewidth=0.8)
                if self.dnde:
                    self.ax.set_ylabel('dN/dE',size=self.fontsize)
                else:
                    self.ax.set_ylabel('Counts',size=self.fontsize)
                self.ax.set_xlabel('E [GeV]',size=self.fontsize)
                self.ax.set_xscale("log")
                self.ax.set_yscale("log")
                self.ax.set_title(label="$\gamma$-ray events from NGC1275", fontsize=self.fontsize)
                
            if xmin: self.ax.set_xlim(xmin=xmin)
            if xmax: self.ax.set_xlim(xmax=xmax)
            if ymin: self.ax.set_ylim(ymin=ymin)
            if ymax: self.ax.set_ylim(ymax=ymax)
            
            if self.with_bkg and self.with_bkg_model and self.with_signal and self.with_signal_model and self.with_edisp:
                appendix = " "
            elif self.with_signal and self.with_signal_model and self.with_edisp:
                appendix = " (w/o bkg) "
            elif self.with_signal and self.with_signal_model and not self.with_edisp and not (self.with_bkg and self.with_bkg_model):
                appendix = " (w/o bkg or edisp) "
            elif self.with_signal and self.with_signal_model and not self.with_edisp and self.with_bkg and self.with_bkg_model:
                appendix = " (w/o edisp) "
            elif self.with_bkg and self.with_bkg_model and not (self.with_signal and self.with_signal_model):
                appendix = " (only bkg) "
            else:
                appendix = "(w/o background or signal...?)"
            
            
            if plot_exp:
                self.ax.plot(self.bin_centers,counts_exp_plot,linewidth=2,alpha=0.52,color=color, linestyle=linestyle, label="Expected" + appendix + "[$m = {:.1f} \, \mathrm{{neV}} \mathrm{{,}} \; g = {:.1f} \\times  10^{{-11}} \, \mathrm{{ GeV}}^{{-1}} $]".format(self.params[0],self.params[1]) )
                print("Total flux (counts): " + str(np.sum(counts_exp_plot)) )
                
                    
            if plot_obs:
                self.ax.errorbar(self.bin_centers, counts_obs_plot, np.sqrt(counts_obs_plot),fmt='.', c=color_obs, elinewidth=2, markersize=5, capsize=4, label="Simulated"+appendix+"for m="+str(self.params[0])+" neV, g="+str(self.params[1])+" $ \\times$ $ 10^{-11}$ GeV$^{-1}}$" )
    
                print("Total flux (counts): " + str(np.sum(counts_obs_plot)) )
     

            if legend: 
                self.ax.legend(loc="upper right", fontsize=min(9*self.figsize[1]/5, 9*self.figsize[0]/9))
            else:
                self.ax.legend().set_visible(False)   
            
            
        if plot_survival:
            
            if new_fig or not self.fig_survival:
                self.fig_survival, self.ax_survival = plt.subplots(figsize=(self.figsize[0], self.figsize[1]*0.5))
                self.ax_survival.grid(True,which='both',linewidth=0.3)
                self.ax_survival.set_ylabel('Photon survival probability', size=15)
                self.ax_survival.set_xlabel('E [GeV]',size=15)
                self.ax_survival.set_ylim([0.,1.1])
                self.ax_survival.set_xscale("log")
                self.ax_survival.plot(self.enpoints_pgg, self.pgg_EBL, "-",color="k",
                         label="intrinsic + EBL")
            
            if xmin: self.ax_survival.set_xlim(xmin=xmin)
            if xmax: self.ax_survival.set_xlim(xmax=xmax)
            
            self.ax_survival.plot(self.enpoints_pgg, self.pgg,color=color,linestyle=linestyle, 
                     label=r"intrinsic + EBL + ALP [m = {:.1f} neV,  g = {:.1f} $ \times  10^{{-11 }} / \mathrm{{GeV}} $]".format(self.params[0],self.params[1]))
            
            #plt.plot([5e1,5e1],[0,1.5], c='0.5', linestyle='--', label="Range in paper")
            #plt.plot([2.8e4,2.8e4],[0,1.5], c='0.5', linestyle='--' )
            
            if legend:
                self.ax_survival.legend(loc="upper right", fontsize=min(9*self.figsize[1]/5, 9*self.figsize[0]/9))
            else:
                self.ax_survival.legend().set_visible(False)
            
            
            

            

        
    def noise(self,
              sim: dict[str,np.ndarray], 
              params: list[float]
              ) -> dict[str,np.ndarray]:
        
        '''
        Adds poissonian noise to an observation.  
        
        Input:
            -  sim:             Observation of the form {'y': np.array}  
            -  params:          Input parameters that produced the observation (required by SWYFT,
                                and helpful if error).  


        Output:
            -  out              Observations with noise. 
        ''' 

        try:    
            d = np.random.poisson(sim['y'].astype(np.float64))*1.0
            #d = sim['y']
            #d = np.log10(np.where(d==0,0.1,d))
                               
        except ValueError as e:
            print('ValueError in noise function, for the following simulation:')
            print()
            print("Sim values: " + str(sim))
            print()
            print("Parameter values: ")
            for i, vel in enumerate(params):
                print("v["+str(i)+"]: "+str(vel))
            return {}
            
            raise e
        
        return dict(y=d)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
