#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:17:42 2023

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
    use_emccd=True
    
    name_scene = 'example_los_jitter_noctrl_onlyJitter_emccd'
    
    star_vmag = 5.00
    cgisim_obj.sources[0]['star_vmag']=star_vmag
    cgisim_obj.sources[0]['star_type']='g0v'

    #%% DMs
    # dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_mild_contrast_dm1.fits' )
    # dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_mild_contrast_dm2.fits' )
    dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_best_contrast_dm1.fits' )
    dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_best_contrast_dm2.fits' )
    cgisim_obj.options['dm1'] = dm1
    cgisim_obj.options['dm2'] = dm2

    #%% Read in the jitter and Z4-11 timeseries from OS11 file
    datadir_jit = '/Users/llopsayson/Documents/Python/cgisim_sims/data/los_jitter_noctrl/'
    flnm_jit = 'los_jitter_noctrl.fits'
    flnm_time = 't_sec.fits'
    
    inFile = pyfits.open(datadir_jit+flnm_jit)
    jit_mat = inFile[0].data
    
    # Retrieve jitter values
    jitt_sig_x_arr0 = jit_mat[0] # masRMS
    jitt_sig_y_arr0 = jit_mat[1] # masRMS
    
    # Read time stamps
    inFile = pyfits.open(datadir_jit+flnm_time)
    time_arr0 = inFile[0].data
    dtime = time_arr0[1]-time_arr0[0]
    
    # bin timesteps
    exptime = 30 #sec
    num_timesteps_per_exp = int(exptime/dtime[0])
    num_timesteps = int(len(time_arr0)/(num_timesteps_per_exp))
    jitt_sig_x_arr = np.zeros(num_timesteps)
    jitt_sig_y_arr = np.zeros(num_timesteps)
    time_arr = np.zeros(num_timesteps)
    for II in range(num_timesteps):
        jitt_sig_x_arr[II] = np.mean(jitt_sig_x_arr0[II*num_timesteps_per_exp:(II+1)*num_timesteps_per_exp])
        jitt_sig_y_arr[II] = np.mean(jitt_sig_y_arr0[II*num_timesteps_per_exp:(II+1)*num_timesteps_per_exp])
        time_arr[II] = II*exptime
    exptime_arr = np.ones(num_timesteps)*exptime
    
    plt.figure(113) 
    plt.plot(jitt_sig_x_arr)
    plt.plot(jitt_sig_y_arr)
    plt.ylabel('Jitter nmRMS')
    plt.xlabel('steps')
    
    plt.figure(114)
    plt.plot(time_arr0,jitt_sig_x_arr0)
    plt.plot(time_arr0,jitt_sig_y_arr0)
    plt.ylabel('LOS Error at Time Stamp')
    plt.xlabel('time [sec]')

    # dasdsada
    #%% Create scene with LO errors
    cgisim_obj.generate_scene(name=name_scene,jitter_x=jitt_sig_x_arr/10,jitter_y=jitt_sig_y_arr/10,exptime=exptime_arr)
    
    # Change number of timesteps:
    cgisim_obj.scene['num_timesteps'] = 194#*4
    
    #%% Generate speckle series for scene
    if use_emccd:
        cgisim_obj.define_emccd(em_gain=5000)
        cgisim_obj.scene['bin_schedule'] = np.array([cgisim_obj.scene['num_timesteps']])
        cgisim_obj.generate_speckleSeries_from_scene(num_images_printed=1,
                                                     title_fig='vmag {}, exptime {}sec, nframes {}'.format(star_vmag,exptime,cgisim_obj.scene['num_timesteps']),
                                                     use_emccd=True,use_photoncount=True)
    else:
        cgisim_obj.generate_speckleSeries_from_scene(num_images_printed=50,vmin_fig=0,vmax_fig=5e-8)
