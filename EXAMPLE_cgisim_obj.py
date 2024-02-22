#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:25:33 2023

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
    name_scene = 'example_OS11_dmsBestContrast'
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
    #%% Create scene with LO errors
    cgisim_obj.generate_scene(name=name_scene,jitter_x=jitt_sig_x_arr,jitter_y=jitt_sig_y_arr,zindex=np.arange(4,11+1),zval_m=z411_mat)
    
    # Change number of timesteps:
    cgisim_obj.scene['num_timesteps'] = 50
    #%% Generate speckle series for scene
    cgisim_obj.generate_speckleSeries_from_scene(num_images_printed=50)