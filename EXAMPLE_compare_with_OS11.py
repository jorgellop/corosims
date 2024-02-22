#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:21:34 2023

@author: llopsayson
"""
from cgisim_sims import cgisim_sims
import astropy.io.fits as pyfits
import matplotlib.pylab as plt
import numpy as np
import proper
import roman_phasec_proper

if __name__ == '__main__':
    cgisim_obj = cgisim_sims()
    name_scene = 'example_compating2OS11'
    #%% DMs
    # dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_mild_contrast_dm1.fits' )
    # dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_mild_contrast_dm2.fits' )
    dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_best_contrast_dm1.fits' )
    dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_best_contrast_dm2.fits' )
    cgisim_obj.options['dm1'] = dm1
    cgisim_obj.options['dm2'] = dm2

    #%% Read in the jitter and Z4-11 timeseries from OS11 file
    datadir_Z411 = '/Users/llopsayson/Documents/Python/cgisim_sims/data/hlc_os11_v2/'
    flnm_Z411 = 'hlc_os11_inputs.fits'
    
    inFile = pyfits.open(datadir_Z411+flnm_Z411)
    hlc_os11_inputs = inFile[0].data
    
    # Retrieve jitter values
    jitt_sig_x_arr = hlc_os11_inputs[:,78] * 1# masRMS
    jitt_sig_y_arr = hlc_os11_inputs[:,79] * 1# masRMS
    plt.figure(111) 
    plt.plot(jitt_sig_x_arr)
    plt.plot(jitt_sig_y_arr)
    
    # Retrieve Z4-11 values
    z411_mat = hlc_os11_inputs[:,46:54]
    plt.figure(112)
    for II in range(11-4):
        plt.plot(z411_mat[:,II])

    #%% Extra stuff from OS11
    dm1_ztilt_deg = 0#0.11
    dm2_ztilt_deg = 0#-0.11
    dm_sampling_m = 0.9906e-3   # actuator spacing in meters
    dm_offset = 0.14e-3/dm_sampling_m
    dm1_xc_act = 23.5# + dm_offset        
    dm1_yc_act = 23.5         
    dm2_xc_act = 23.5# - dm_offset
    dm2_yc_act = 23.5         
    fpm_x_offset_m =0# 0.82e-6          # FPM x,y offset in meters
    fpm_y_offset_m = 0
    fpm_z_shift_m = 0#70e-6           # occulter offset in meters along optical axis (+ = away from prior optics)

    passvalue_proper = {'dm1_ztilt_deg':dm1_ztilt_deg,
                        'dm2_ztilt_deg':dm2_ztilt_deg,
                        'dm1_xc_act':dm1_xc_act,
                        'dm1_yc_act':dm1_yc_act,
                        'dm2_xc_act':dm2_xc_act,
                        'dm2_yc_act':dm2_yc_act,
                        'fpm_x_offset_m':fpm_x_offset_m,
                        'fpm_y_offset_m':fpm_y_offset_m,
                        'fpm_z_shift_m':fpm_z_shift_m}
    #%% Create scene with LO errors
    cgisim_obj.generate_scene(name=name_scene,jitter_x=jitt_sig_x_arr,jitter_y=jitt_sig_y_arr,zindex=np.arange(4,11+1),zval_m=z411_mat,
                              passvalue_proper=passvalue_proper)
    
    # Change number of timesteps:
    cgisim_obj.scene['num_timesteps'] = 3
    #%% Generate speckle series for scene
    cgisim_obj.generate_speckleSeries_from_scene(num_images_printed=0)