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
import matplotlib.colors as mcolors

import logging
from typing import Union, Optional
import warnings

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
                 set_obs=1,
                 set_null=1
                 ) -> None:
       
        ''' 
        Input:
            -  set_geom:            Whether to generate a geometry object for the observations. Can 
                                    be set to 0, IF geom parameters are to be changed before 
                                    simulation, to save time.
            -  set_null:            Whether to set the counts of the null hypothesis. Can be set to 
                                    0, IF model parameters are to be edited,  or uf a custom 
                                    null-hypothersis is to be defined (see method generate_null()), 
                                    to save time. 
        '''
        
       
        # Geom configuration parameters. See method configure_geom().
        self.emin = 10      #GeV
        self.emax = 1e5     #GeV 
        self.nbins = 50 
        self.nbins_etrue = 150
        self.pointing = [150.58,-13.26]
        self.livetime = 250 * u.hr 
        self.irf_file = "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
        
        # Model configuration parameters. See method configure_model().
        self.params = [0, 0]
        self.with_bkg = True
        self.with_signal = True
        self.with_bkg_model = True
        self.with_signal_model = True
        self.with_edisp = True
        self.with_logcounts = False
        self.with_residuals = False
        self.nB = 1
        self.ALP_seed = None
        self._floor = None
        self._floor_obs = None
        
        

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
         
        # Saved counts, in general computed within class instances, using methods compute_case() or 
        #  generate_null(). But can also be imported, see method import_counts(). 
        self.counts_obs = None
        self.counts_exp = None
        self.counts_null = None

        # Parameter expansion function. See method full_params_default.  
        self.full_param_vec = ALP_sim.full_params_default
        
        # Intermediate results, not to be changed by user. 
        self.pgg = None
        self.pgg_EBL = None
        self.pgg_combined = None
        self.bin_centers = None
        self.bin_widths = None
        

        # State variables
        self._need_new_null=True        # Whether a new null-hypothesis should be computed when running a simulation for residuals.
                                        #  Updated in self.generate_null(), self.configure_model() and self.configure_obs().          
        self._residuals=None            # Whether saved simulations are counts or residuals. 
                                        #  Updated in self.subtract_null(), self.import_counts() and self.compute_case().           
                                        
        
        if set_obs: self.set_obs()
        if set_null: 
            if not set_obs:
                raise ValueError("Null hypothesis can't be set if geom hasn't been set (using \
                                 self.set_obs()). Set set_obs to True or set_null to False in object \
                                instantiation.")
            else:
                self.generate_null()
    
        
    
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
                        irf_file: Union[str,None]="_empty",
                        to_residuals=True
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
            -  to_residuals:        Unless set to false, using configure_obs() will update 
                                    the null-hypothesis (self.counts_null) next time a simulation is
                                    generated in residual mode (activated using 
                                    self.configure_model()). 
        '''
        
        model_changed = False
        
        if bkg != "_empty" and bkg != self.with_bkg: 
            self.with_bkg = bkg
            model_changed = True
        if signal != "_empty" and signal != self.with_signal: 
            self.with_signal = signal
            model_changed = True
        if edisp != "_empty" and edisp != self.with_edisp: 
            self.with_edisp = edisp
            model_changed = True
        if emin != "_empty" and emin != self.emin: 
            self.emin = emin
            model_changed = True
        if emax != "_empty" and emax != self.emax: 
            self.emax = emax
            model_changed = True
        if nbins != "_empty" and nbins != self.nbins: 
            self.nbins = nbins
            model_changed = True
        if nbins_etrue != "_empty" and nbins_etrue != self.nbins_etrue: 
            self.nbins_etrue = nbins_etrue
            model_changed = True
        if pointing != "_empty" and pointing != self.pointing: 
            self.pointing = pointing
            model_changed = True
        if livetime != "_empty" and livetime != self.livetime: 
            self.livetime = livetime * u.hr
            model_changed = True
        if irf_file != "_empty" and irf_file != self.irf_file: 
            self.irf_file = irf_file
            model_changed = True
        
        self.set_obs()
        
        if model_changed and to_residuals:
            self._need_new_null = True
             
    
    def configure_model(self,
                        params: Union[list[float], None]="_empty",
                        bkg: Union[bool,None]="_empty",
                        signal: Union[bool,None]="_empty",
                        logcounts: Union[bool,None]="_empty",
                        residuals: Union[bool,None]="_empty",
                        floor: Union[None,int,float]="_empty",
                        floor_obs: Union[None,int,float]="_empty",
                        nB: Union[int,None]="_empty",
                        ALP_seed="_empty", 
                        new_null=True
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
            -  logcounts:           Whether or not the model should ouput the logarithm of counts.
            -  residuals:           Whether or not the model should ouput the residual of (log)counts.
            -  floor:               The minimum value for the elements of the histogram that is
                                    generated by the model. Lower values are set to this minimum
                                    value when the model function is run. 
            -  floor_obs:           The minimum value for the elements of the histogram that is
                                    generated by the noise function. Lower values are set to this 
                                    minimum value when the noise function is run. 
            -  edisp:               Whether or not to apply energy dispersion.
            -  nB:                  [Redundant] Number of B-field realizations to compute
            -  ALP_seed:            Seed for random B-field realizations. Set to other than None for 
                                    reproduction of same realization (?).
            -  new_null:            Unless set to false, using configure_model() will update 
                                    the null-hypothesis (self.counts_null) next time a simulation is
                                    generated in residual mode (activated using 
                                    self.configure_model()). 
        '''
        
        model_changed = False
                
        if params != "_empty" and not np.array_equiv(np.array(params),np.array(self.params)): 
            self.params = params
            model_changed = True
        if bkg != "_empty" and bkg != self.with_bkg_model: 
            self.with_bkg_model = bkg
            model_changed = True
        if signal != "_empty" and signal != self.with_signal_model: 
            self.with_signal_model = signal
            model_changed = True
        if logcounts != "_empty" and logcounts != self.with_logcounts: 
            self.with_logcounts = logcounts
            model_changed = True
        if residuals != "_empty" and residuals != self.with_residuals: 
            self.with_residuals = residuals
            model_changed = True
        if floor != "_empty" and floor != self._floor: 
            self._floor = floor
            model_changed = True
        if floor_obs != "_empty" and floor_obs != self._floor_obs: 
            self._floor_obs = floor_obs
        if nB != "_empty" and nB != self.nB: 
            self.nB = nB
            model_changed = True
        if ALP_seed != "_empty" and ALP_seed != self.ALP_seed: #TODO: standardize ALP_seed when using None for null-hypothesis?
            self.ALP_seed = ALP_seed
            model_changed = True
        if model_changed and new_null:
            self._need_new_null = True
       

        #TODO: error band around expected counts? 
        
        
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
        
    
    def import_counts(self,
                   obs: Union[list,np.ndarray,dict[str,np.ndarray]]=None,
                   exp: Union[list,np.ndarray,dict[str,np.ndarray]]=None,
                   null: Union[list,np.ndarray,dict[str,np.ndarray]]=None,
                   are_residuals=False
                   ) -> None:
        
        ''' 
        
        Takes in a histogram of counts, and saves them to self.counts_obs and self.counts_exp. 
        The length of the histogram must be equal to self.nbins (set using method configure_obs()). 
        
        Input:
            -  obs:              Vector of observed counts, e.g. from earlier simulations.
            -  exp:              Vector of expected counts, e.g. from earlier simulations.
            -  null:             Vector of null-hypothesis counts, e.g. from earlier simulations.
            -  are_residuals     Whether the imported counts are residuals of the null-hypothesis.
                                 Setting to true means both expected and observed counts will be
                                 interpreted as residuals, no matter the other input to this 
                                 method. 
        '''
        
        #TODO: Test import of null hypothesis
        
        obs_counts, obs_isnone = ALP_sim.format_counts(obs)
        exp_counts, exp_isnone = ALP_sim.format_counts(exp)
        null_counts, null_isnone = ALP_sim.format_counts(null)
        
        
        if not obs_isnone:
            if len(obs_counts['y']) != self.nbins:
                raise ValueError("Imported observed counts should have same length as self.nbins")
            else:
                if not np.array_equiv(obs_counts['y'], obs_counts['y'].astype(int)): 
                    warnings.warn("Imported observed counts should all be integers")
                self.counts_obs = obs_counts                
        else:
            print("No observed counts specified for import, keeping any old ones")
            
        
        if not exp_isnone:
            if len(exp_counts['y']) == self.nbins:
                self.counts_exp = exp_counts
            else:
                raise ValueError("Imported expected counts should have same length as self.nbins")
        else:
            print("No expected counts specified for import, keeping any old ones")
            
        
        if not null_isnone:
            if len(null_counts['y']) == self.nbins:
                self.counts_null = null_counts
            else:
                raise ValueError("Imported null-hypothesis counts should have same length as self.nbins")
        else:
            print("No null-hypothesis counts specified for import, keeping any old ones")
        
        self._residuals = are_residuals
    
    
    def generate_null(self, 
                      params: list[float]=[],
                      model: str="",
                      ) -> dict[str,np.ndarray]:
        
        ''' 

        
        Input:
            - params:           Input parameters corresponding to chosen null hypothesis. If left
                                empty, will correspond to [m,g] = [0,0], and nuisance parameters set
                                to default values. 
            - model:            Which simulation model to run. Choices are "", "log", "spectral_fit",
                                "spectral_fit_log", "toy_line" and "toy_sine". Default is "".
        '''
        
        if isinstance(params,list) or isinstance(params,np.ndarray):
            if len(params) > 0:
                params_use = params
            else:
                params_use = self.params.copy()
                params_use[0] = 0
                params_use[1] = 0
        else:
            raise TypeError("Params are of unexpected type. Expected list or numpy array.")
            
            
        mod = model
        if model != "": mod = "_"+model
        
        try:
            mod_func = eval("self.model"+mod)
                       
        except AttributeError as AttrErr:
            if self.counts_null == None:
                self.generate_null()
                # print("counts_null: " + str(self.counts_null))
                if self.counts_null == None:
                    raise ValueError("self.generate_null() did not result in self.counts_null that\
                                     are not None.")
            else:
                raise AttrErr
    
        
        self._need_new_null=False
        
        if self.with_residuals:
            self.with_residuals=False
            try:
                self.counts_null = mod_func(params_use)
                self.with_residuals=True
            except Exception as Err:
                self.with_residuals=True
                raise Err
        else:
            self.counts_null = mod_func(params_use)
        
        

    def subtract_null(self,
                       counts: Union[None,list,np.ndarray,dict[str,np.ndarray]]=None,
                       add: Union[None,bool]=None, 
                       ) -> Union[None,dict[str,np.ndarray]]:
        
        
        ''' 
        Can take in a histogram of counts, and output that histogram, but with the expected counts
        of the null-hypothesis subtracted. Handles situations where arrays involve infs (as a result
        of applying log of counts). If no counts are given as input, it converts self.counts_obs or 
        self.counts exp (depending on whether they already are counts or residuals, according to 
        self._residuals). See method set_null() to set the null-hypothesis. If the null-hypothesis 
        has a different length from the inputed counts, a new hypthesis is computed and stored in 
        self.null-hypothesis.
        
        Input:
            -  counts:          Vector of observed counts, e.g. from earlier simulations.
            -  add              If True, null hypothesis is added to input counts, rather than
                                subtracted. If False, null hypothesis is subtracted. If not None, 
                                the action is forced over other concerns. 
        '''
        

        counts_calc, counts_isnone = ALP_sim.format_counts(counts)
        counts_calc = counts_calc['y']
               
        null = self.counts_null['y'] if add else -self.counts_null['y']

        if counts_isnone:
            
            if add!=None:
                add_new=add
                self._residuals = not self._residuals if add==self._residuals else None
            else:
                add_new = self._residuals
                if self._residuals==None:
                    warnings.warn("Unclear whether counts currently represent residuals or not. To \
                                  clarify, compute new counts using self.compute_case(), or import \
                                  counts using self.import_counts(are_residuals=True/False).")
                else:
                    self._residuals = not self._residuals
            
            if self.counts_obs==None or self.counts_exp==None:
                raise ValueError("Instance method ALP_sim.subtract_null cannot be run without \
                                 arguments when self.counts_obs or self.counts_exp is None.")
            else:
                self.counts_obs = self.subtract_null(self.counts_obs, add = add_new)
                self.counts_exp = self.subtract_null(self.counts_exp, add = add_new)
            
        else:
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    # counts_calc = counts_calc + null
                    counts_calc = np.where(np.logical_and(abs(counts_calc)==np.inf,null==-counts_calc),0,counts_calc+null)

                return {'y':counts_calc}
            
            except ValueError as VErr:
                if len(counts_calc) != len(null): 
                    ValueError("Length of input counts doesn't match length of null hypothesis. Try \
                               generating a suitable null hypothesis using the method set_null().")
                else:
                    raise VErr 
            
              
    
    def set_obs(self,
                ) -> None:
        
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
                        emax       = self.emax,  # GeVnp.where(errorbars==-np.inf, 0, 
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
        
        
        out = self.convert_counts(counts)
        
        
        # if self.with_logcounts:
        #     with np.errstate(divide='ignore', invalid='ignore'):
        #         counts = np.where(counts==0,-np.inf,np.log10(counts))

        # if self.with_residuals: 
        #     if self._need_new_null:
        #         self.generate_null()
        #     counts = self.subtract_null(counts)['y']
 
        # out = dict(y=np.array(counts))
        
     
        return out
    
    
    def convert_counts(self,
                       counts: Union[None,list,np.ndarray,dict[str,np.ndarray]]=None,
                       ) -> dict[str,np.ndarray]:
        
        ''' 
        Transforms a histogram of counts, tests that they are in a suitable form, and converts them 
        into a dictionary of (residuals of) (the logarithm of) counts, depending on the model 
        configuration (defined using self.configure_model()). For use at the end of model methods
        (i.e. in self.model() and self.model_*()). May update self.counts_null if in residual mode.   
        
        Input:
            -  counts:          Histogram of observed counts, primarily from the earlier part of a
                                simulation function.
        
        returns:
            
            - out:              (residuals of) (the logarithm of) counts, depending on the model 
                                configuration (defined using self.configure_model()).
        '''
        
        
        # out = ALP_sim.format_counts(counts)[0]['y']
        
        out = counts
        
        # print("out: " + str(out))ALP_spectra_B
        
        if self.with_logcounts:
            with np.errstate(divide='ignore', invalid='ignore'):
                out = np.where(out==0,-np.inf,np.log10(out))

        if self.with_residuals: 
            if self._need_new_null: 
                self.generate_null()
            out = self.subtract_null(out)['y']
            
        
        if self._floor !=None: out = np.where(out<self._floor,self._floor,out)
 
        out = dict(y=np.array(out))
        
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
            lambda_= 1/(v[5]*1e-3) * u.Unit("TeV-1")    # 1.22 * u.Unit("TeV-1"),ALP_spectra_B
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
        
 
        out = self.convert_counts(counts)
        
     
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
                     new_counts: Union[bool,None]=True,
                     null: bool=False, 
                     plot_obs: bool=True, 
                     plot_exp: bool=True,
                     plot_survival: bool=False,
                     errorbands: bool=True,
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
                                self.counts_exp. In combination with null=True, new_counts==True
                                establishes a new null-hypothesis. If new_counts=None, a new 
                                null-hypothesis is established depending on whether the model-
                                or obs configuration has been changed previously (using methods
                                self.configure_model() or self.configure_obs()). 
            -  null             If set to True, plots the null_hypothesis. In combination with 
                                new_counts=True, establishes new null-hypothesis by calling
                                self.generate_null(). Also establishes new null-hypothesis if model-
                                or obs configuration has been changed previously (using methods
                                self.configure_model() or self.configure_obs()). 
            -  plot_obs:        Whether or not to plot observations.
            -  plot_exp:        Whether or not to plot expected values.
            -  plot_survival:   Whether or not to visualize the photon survival probability 
                                (separate figure, placed in self.fig_survival). 
            -  errorbands:      Whether or not to plot errorbands around expected counts.
            -  model:           Which model to simulate from. Options: "", "log", "spectral_fit",
                                "spectral_fit_log", "toy_line", "toy_sine".
            -  color:           Plot color for expected values. 
            -  color_obs:       Plot color for observed values. 
            -  linestyle:       Linestyle for expected values. 
            -  Legend:          Whether or not to include a legend. 


        '''        
        
        
        # Strings for labels, to be modified depending on input arguments. 
        string1=""
        string2=""
        string3="Counts"
        string4 = "$m = {:.1f} \, \mathrm{{neV}} \mathrm{{,}} \; g = {:.1f} \\times  10^{{-11}} \, \mathrm{{ GeV}}^{{-1}} $".format(self.params[0],self.params[1])
        
        
        mod = model
        if model != "": mod = "_"+model
        
        mod_func = eval("self.model"+mod)
        

        if not null:
            if new_counts or self._need_new_null:
                self.counts_exp = mod_func(self.params)
                self.counts_obs = self.noise(self.counts_exp, self.params)
                if self.with_residuals: self._residuals = True
            counts_exp_plot = self.counts_exp['y']
            counts_obs_plot = self.counts_obs['y']
        else:
            if new_counts or (new_counts==None and self._need_new_null): self.generate_null()
            counts_exp_plot = self.counts_null
            if self.with_residuals: counts_exp_plot = self.subtract_null(counts_exp_plot)
            counts_obs_plot = self.noise(counts_exp_plot, self.params)['y']
            counts_exp_plot = counts_exp_plot['y']
            string4 = "null hypothesis"

        # counts_exp_plot = self.counts_exp['y']
        # counts_obs_plot = self.counts_obs['y']
        # if self.with_residuals: counts_null_plot = self.counts_null['y']


        errorbars = counts_obs_plot.copy()
        uncertainty = counts_exp_plot.copy()
        if self.with_residuals:
            errorbars = self.subtract_null(errorbars,add=True)['y']
            uncertainty = self.subtract_null(uncertainty,add=True)['y']
            # errorbars = errorbars + counts_null_plot
            # uncertainty = uncertainty + counts_null_plot
            string1 = "Residuals of "
        if self.with_logcounts:
            with np.errstate(divide='ignore', invalid='ignore'):
                # print("errorbars: " + str(errorbars))
                upper_error = np.where(errorbars==-np.inf, 0, np.log10(10**errorbars + np.sqrt(10**errorbars)) - errorbars)
                lower_error = np.where(errorbars==-np.inf, 0, errorbars - np.log10(10**errorbars - np.sqrt(10**errorbars)))
                upper_uncertainty = np.where(uncertainty==-np.inf, 0, np.log10(10**uncertainty + np.sqrt(10**uncertainty)) - uncertainty)
                lower_uncertainty = np.where(uncertainty==-np.inf, 0, uncertainty - np.log10(np.max(np.array([10**uncertainty - np.sqrt(10**uncertainty),np.zeros(len(uncertainty))]),axis=0)))
            string2 = "log of "
        else:
            lower_error, upper_error = np.sqrt(errorbars), np.sqrt(errorbars)
            lower_uncertainty, upper_uncertainty = np.sqrt(uncertainty), np.sqrt(uncertainty)

        # print("Upper_uncertainty: " +  str(upper_uncertainty))     
        
        # Determining the axis limits of the plot
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        
        if not np.array([xmin,xmax,ymin,ymax]).all():
            minmax_array = np.concatenate((counts_exp_plot,counts_obs_plot))
            
            minmax_array_nonzero = minmax_array[minmax_array!=0]
            ymin_nonzero = min(minmax_array_nonzero) if minmax_array_nonzero.any() else np.inf
            bin_centers_nonzero = self.bin_centers.copy()#[counts_exp_plot!=0]
            # print("bin_centers: " + str(self.bin_centers))
            # print("counts_exp_plot: " + str(counts_exp_plot))
            # print("bin_centers[counts]: " + str(self.bin_centers[counts_exp_plot!=0]) )
            xmin_nonzero = min(bin_centers_nonzero) if bin_centers_nonzero.any() else np.inf
        else:
            ymin_nonzero = ymin
            xmin_nonzero = xmin

        
        if not xmin: 
            xmin=min(self.bin_centers)#[counts_exp_plot != 0])

        if not xmax: 
            xmax=max(self.bin_centers)#[counts_exp_plot != 0])
        
        if xmin == xmax:
            xmin, xmax = xmin-1, xmax+1
        else:
            xmin, xmax = xmin-0.5*abs(xmin), xmax+0.5*abs(xmax)
            
        if not ymin:
            lower_counts = np.concatenate((counts_obs_plot - lower_error, counts_exp_plot))
            ymin=min(lower_counts[lower_counts != -np.inf])
            # print("ymin2: " + str(ymin))
            # print("lower_counts2: " + str(lower_counts))
            # print("lower_error: " + str(lower_error))
            # print("counts_obs_plot: " + str(counts_obs_plot))
        if not ymax:
            ymax=max(np.concatenate((counts_obs_plot + upper_error, counts_exp_plot)))       
        if ymin == ymax:
            ymin, ymax = ymin-1, ymax+1
        else:
            ymin, ymax = ymin-0.1*abs(ymin), ymax+0.1*abs(ymax)



        if self.dnde: 
            #TODO: Implement for residuals and logcounts
            counts_exp_plot = counts_exp_plot/self.bin_widths
            counts_obs_plot = counts_obs_plot/self.bin_widths
            if self.with_residuals: counts_null_plot = self.counts_null/self.bin_widths
            ymin = ymin/self.bin_widths
            ymax = ymax/self.bin_widths
            string3 = "dN/dE"
        else:
            counts_null_plot=self.counts_null




        # Plotting
        if (plot_obs or plot_exp): 
            
            if new_fig or not self.fig:
                self.fig, self.ax = plt.subplots(figsize=self.figsize)
                self.ax.grid(True,which='both',linewidth=0.8)
                self.ax.set_ylabel(string1+string2+string3,size=self.fontsize)
                self.ax.set_xlabel('E [GeV]',size=self.fontsize)
                self.ax.set_title(label="$\gamma$-ray events from NGC1275", fontsize=self.fontsize)
                xmin_prev, xmax_prev, ymin_prev, ymax_prev = np.inf, -np.inf, np.inf, -np.inf
            else:
                xmin_prev, xmax_prev = self.ax.get_xlim()
                ymin_prev, ymax_prev = self.ax.get_ylim()
        
            # print("xmax: " + str(xmax))
            # print("xmin_nonzero: " + str(xmin_nonzero))
        
            if abs(xmax/xmin_nonzero) > 100 and not xmin < 0: 
                self.ax.set_xscale("log")
                xmin = 0.5*xmin_nonzero
            else:
                self.ax.set_xscale("linear")
            
            if abs(ymax/ymin_nonzero) > 100 and not ymin < 0: 
                self.ax.set_yscale("log")
                legend_position="upper right"
                ymin = 0.5*ymin_nonzero
                ymax = 5*ymax
            else:
                self.ax.set_yscale("linear")
                legend_position="upper right"
            
            
            # print("ymin1: " + str(ymin))
            # print("ymin_prev1: " + str(ymin_prev))
            
            xmin=min(xmin,xmin_prev)
            xmax=max(xmax,xmax_prev)
            ymin=min(ymin,ymin_prev)
            ymax=max(ymax,ymax_prev)
                         
            self.ax.set_xlim(xmin=xmin)
            self.ax.set_xlim(xmax=xmax)
            self.ax.set_ylim(ymin=ymin)
            self.ax.set_ylim(ymax=ymax)
            
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
                
                # print("counts_exp_plot: " + str(counts_exp_plot))
                counts_exp_noinf = counts_exp_plot[np.logical_and(counts_exp_plot != -np.inf,counts_exp_plot!=np.inf)]               
                               
                self.ax.plot(self.bin_centers[counts_exp_plot != -np.inf],counts_exp_noinf,linewidth=2,alpha=0.52,color=color, linestyle=linestyle, label="Expected" + appendix + " [" + string4 + "]" )
                
                    
                if errorbands:
                    
                    color_band = self.ax.lines[-1].get_color()
                    color_band = mcolors.to_rgb(color_band)
                    color_band_lightness = 0.7
                    color_band_light = (color_band[0] + (1-color_band[0])*color_band_lightness,color_band[1]+ (1-color_band[2])*color_band_lightness, color_band[2]+(1-color_band[2])*color_band_lightness)
                    
                    # print(1)
                    # print(lower_uncertainty)
                    # print(upper_uncertainty)
                    
                    lower_uncertainty_noinf = lower_uncertainty[np.logical_and(counts_exp_plot != -np.inf,counts_exp_plot!=np.inf)]
                    lower_uncertainty_noinf = np.where(lower_uncertainty_noinf==np.inf,abs(counts_exp_noinf - ymin)+0.1*abs(ymin),lower_uncertainty_noinf)
                    # lower_uncertainty_noinf = np.where(lower_uncertainty_noinf==0,counts_exp_noinf - ymin,lower_uncertainty_noinf)        
                    # lower_uncertainty_noinf = np.where(lower_uncertainty_noinf==-np.inf,0,lower_uncertainty_noinf)  
                    upper_uncertainty_noinf = upper_uncertainty[np.logical_and(counts_exp_plot != -np.inf,counts_exp_plot!=np.inf)]
                    upper_uncertainty_noinf = np.where(upper_uncertainty_noinf==np.inf,0.1*abs(ymax)+abs(ymax-counts_exp_noinf),upper_uncertainty_noinf)
                    
                    # print(2)
                    # print(lower_uncertainty_noinf)
                    # print(upper_uncertainty_noinf)
                    
                    self.ax.fill_between(self.bin_centers[counts_exp_plot != -np.inf], counts_exp_noinf-lower_uncertainty_noinf, counts_exp_noinf + upper_uncertainty_noinf, color=color_band_light, alpha=0.5)
                          
                
            if plot_obs:
                
                counts_obs_noinf = counts_obs_plot[np.logical_and(counts_obs_plot!=-np.inf,counts_obs_plot!=np.inf)]
                lower_error_noinf = lower_error[np.logical_and(counts_obs_plot!=-np.inf,counts_obs_plot!=np.inf)]
                lower_error_noinf = np.where(lower_error_noinf==np.inf,abs(counts_obs_noinf - ymin),lower_error_noinf)
                upper_error_noinf = lower_error[np.logical_and(counts_obs_plot!=-np.inf,counts_obs_plot!=np.inf)]
                upper_error_noinf = np.where(upper_error_noinf==np.inf,abs(counts_obs_noinf - ymax),upper_error_noinf)
                          
                self.ax.errorbar(self.bin_centers[counts_obs_plot != -np.inf], counts_obs_noinf, [lower_error_noinf, upper_error_noinf],fmt='.', c=color_obs, elinewidth=2, markersize=5, capsize=4, label="Simulated"+appendix+"for " + string4 )
    
     

            if legend: 
                self.ax.legend(loc=legend_position, fontsize=min(9*self.figsize[1]/5, 9*self.figsize[0]/9))
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
                self.ax_survival.legend(loc=legend_position, fontsize=min(9*self.figsize[1]/5, 9*self.figsize[0]/9))
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
            d = sim['y'].astype(np.float64)
            
            if self.with_residuals:
                
                # d = d + self.counts_null['y']
                d = self.subtract_null(d,add=True)['y']
                
                if self.with_logcounts: 
                    d = 10**d
                    d = np.random.poisson(d)    
                    with np.errstate(divide='ignore', invalid='ignore'): 
                        d = np.where(d==0,-np.inf,np.log10(d))
                else: 
                    d = np.random.poisson(d)
            
                # d = d - self.counts_null['y']
                d = self.subtract_null(d,add=False)['y']
           
            else:
                if self.with_logcounts: 
                    d = 10**d
                    d = np.random.poisson(d)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        d = np.where(d==0,-np.inf,np.log10(d))  
                else: 
                    d = np.random.poisson(d)
            
            if self._floor_obs !=None: d = np.where(d<self._floor_obs,self._floor_obs,d)
            
            
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
    
    
    
    #TODO: implement
    def full_params_new(self,
                        params: list[float]
                        ) -> list[float]:
                    
        ''' 
        A new way to extend the parameters, using a Dict to define which nuisance parameters are
        to be included. To be implemented later. 
        
        
        Input:
            -  params:              Input parameters to model methods. 

        Output:
            -  full_par             Full list of all 18 model parameter values. 

        '''             
        
        
        return None
        
        
        
    
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
    
    
    
    
    @staticmethod
    def format_counts(
                      counts: Union[list,np.ndarray,dict[str,np.ndarray]]
                      ) -> dict[str,np.ndarray]:
            
        ''' 
        Transforms a histogram of counts, tests that they are in a suitable form, and converts them 
        into a dictionary of form {'y': np.ndarray}. For flexibility of input in other functions. 
        
        Input:
            -  counts:           Histogram of observed counts, e.g. from earlier simulations.
        
        returns:
            
            -  counts_calc       Histogram of observed counts, in standard format.      
        '''
        
        
        isnone = False
        
        if isinstance(counts,np.ndarray):
            counts_calc = {'y':counts}
        elif isinstance(counts,list):
            counts_calc = {'y':np.array(counts)}
        elif isinstance(counts, dict):
            try:
                counts_calc = {'y':np.array(counts['y'])}
            except:
                KeyError("The input counts should be a list, numpy.ndarray, or dictionary of \
                               the form {'y': numpy.ndarray}")
        else:
            try:
                if counts==None:
                    counts_calc = None
                    isnone = True
                else:
                    raise TypeError("The input counts should be a list, numpy.ndarray, or dictionary of \
                               the form {'y': numpy.ndarray}")
            except:
                    TypeError("The input counts should be a list, numpy.ndarray, or dictionary of \
                               the form {'y': numpy.ndarray}")
        
         
        return counts_calc, isnone
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
