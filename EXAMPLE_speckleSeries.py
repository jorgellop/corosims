#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:23:25 2024

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

    name_speckleSeries = 'example_speckleSeries_oneSource'
    
    flag_use_emccd = False
    
    # Science target star
    star_vmag = 5.04
    cgisim_obj.sources[0]['star_vmag']=star_vmag
    cgisim_obj.sources[0]['star_type']='g0v'
    cgisim_obj.sources[0]['name']='47UMa'
    
    #%% Read in the jitter and Z4-11 timeseries from OS11 file
    datadir_Z411 = 'data/hlc_os11_v2/'
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
    cgisim_obj.generate_scene(name=name_speckleSeries,jitter_x=jitt_sig_x_arr,jitter_y=jitt_sig_y_arr,
                              zindex=np.arange(4,11+1),zval_m=z411_mat)
    
    # Initialize schedule_index_array
    cgisim_obj.scene['schedule']['schedule_index_array'] = []
    
    # Reference observation
    batch_ID = 0
    num_frames_ref = 12
    sourceid = 0 # what star?
    cgisim_obj.scene['schedule']['batches'][0] = {'num_timesteps':num_frames_ref,
                                                     'batch_ID':batch_ID,
                                                     'sourceid':sourceid}
    

    
    #%% Generate speckle series for scene
    cgisim_obj.generate_speckleSeries_from_scene(num_images_printed=0,flag_return_contrast=False,
                                                 use_emccd=False,use_photoncount=False)
    
    
