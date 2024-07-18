#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:23:25 2024

@author: llopsayson
"""

import cgisim_sims
import matplotlib.pylab as plt
import numpy as np

if __name__ == '__main__':
    
    name_speckleSeries = 'example_speckleSeries_ScienceAndRef'
    obs_obj = cgisim_sims.Observation(name=name_speckleSeries)

    
    flag_use_emccd = False
    
    #%% Define source and scene
    # Science target star
    star_vmag = 2.25
    obs_obj.sources[0]['vmag']=star_vmag
    obs_obj.sources[0]['star_type']='a0v'
    obs_obj.sources[0]['name']='referenceStar'
    
    # Ref target star
    star_vmag = 5.04
    obs_obj.sources[1]['vmag']=star_vmag
    obs_obj.sources[1]['star_type']='g0v'
    obs_obj.sources[1]['name']='47UMa'
    
    
    obs_obj.create_scene(name='SCI')
    obs_obj.add_point_source_to_scene(scene_name='SCI',source_name='47UMa')
    obs_obj.create_scene(name='REF')
    obs_obj.add_point_source_to_scene(scene_name='REF',source_name='referenceStar')

    #%% Define jitter timeseries
    num_timesteps_batch0=10
    jitter_sig_x = 0.5 # mas
    jitter_sig_y = 0.5 # mas
    jitt_sig_x_arr_0 = np.random.normal(0,jitter_sig_x,num_timesteps_batch0)
    jitt_sig_y_arr_0 = np.random.normal(0,jitter_sig_y,num_timesteps_batch0)
    
    num_timesteps_batch1=30
    jitter_sig_x = 3 # mas
    jitter_sig_y = 0.5 # mas
    jitt_sig_x_arr_1 = np.random.normal(0,jitter_sig_x,num_timesteps_batch1)
    jitt_sig_y_arr_1 = np.random.normal(0,jitter_sig_y,num_timesteps_batch1)

    plt.figure(111)
    plt.plot(np.append(jitt_sig_x_arr_0,jitt_sig_x_arr_1))
    plt.plot(np.append(jitt_sig_y_arr_0,jitt_sig_y_arr_1))
    
    #%% Create batches
    # Batch 0
    obs_obj.create_batch(scene_name='REF',jitter_x=jitt_sig_x_arr_0,jitter_y=jitt_sig_y_arr_0)
    
    # Batch 1
    obs_obj.create_batch(scene_name='SCI',jitter_x=jitt_sig_x_arr_1,jitter_y=jitt_sig_y_arr_1)
    #%% Generate speckle series for scene
    obs_obj.generate_speckleSeries(num_images_printed=0,flag_return_contrast=False,
                                                 use_emccd=False,use_photoncount=False,flag_compute_normalization=True)
    
    #%% Add detector noise
    # Define detector
    obs_obj.corgisim.define_emccd(em_gain=5000.0)
    
    num_frames_interp_batch0 = 1000
    obs_obj.batches[0]['num_frames_interp'] =  num_frames_interp_batch0
    obs_obj.batches[0]['exptime'] = 2.0 # sec
    num_frames_interp_batch1 = 200
    obs_obj.batches[1]['num_frames_interp'] =  num_frames_interp_batch1
    obs_obj.batches[1]['exptime'] = 30.0 # sec

    obs_obj.add_detector_noise_to_batches()
    
    #%% plot results
    # Load data
    obs_obj.load_batches_cubes()
    
    batches = obs_obj.batches
    fntsz = 10
    pixel_scale = obs_obj.corgisim.options['pixel_scale']
    max_fov = obs_obj.corgisim.sz_im* pixel_scale / 2
    zoom_pix = 30
    for II,batch in enumerate(batches):
        # generate figures
        fig, axes = plt.subplots(2, 2,figsize=(2* 4.5, 2 * 3.75))#, dpi=300)
        # First Image:
        im = axes[0,0].imshow(batch['im_cube'][0], extent=[max_fov,-max_fov,max_fov,-max_fov], cmap='hot')
        axes[0,0].set_xlim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        axes[0,0].set_ylim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        cb = fig.colorbar(im, ax=axes[0,0])#,label='Jy')
        # cb.ax.set_ylabel('Contrast')
        cb.ax.yaxis.label.set_size(fntsz)
        axes[0,0].invert_yaxis()
        # axes[0,0].set_xlabel('RA [arcsec]', fontsize = fntsz)
        axes[0,0].set_ylabel('Dec [arcsec]', fontsize = fntsz)
        axes[0,0].set_title('First Image of Timeseries - No CCD Noise', fontsize = fntsz)

        # First Image:
        im = axes[0,1].imshow(batch['im_cube_emccd'][0], extent=[max_fov,-max_fov,max_fov,-max_fov], cmap='hot')
        axes[0,1].set_xlim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        axes[0,1].set_ylim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        cb = fig.colorbar(im, ax=axes[0,1])#,label='Jy')
        cb.ax.set_ylabel('Contrast')
        cb.ax.yaxis.label.set_size(fntsz)
        axes[0,1].invert_yaxis()
        # axes[0,0].set_xlabel('RA [arcsec]', fontsize = fntsz)
        # axes[0,1].set_ylabel('Dec [arcsec]', fontsize = fntsz)
        axes[0,1].set_title('First Image of Timeseries - With CCD Noise', fontsize = fntsz)

        # First Image:
        im = axes[1,0].imshow(batch['im_coadded'], extent=[max_fov,-max_fov,max_fov,-max_fov], cmap='hot')
        axes[1,0].set_xlim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        axes[1,0].set_ylim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        cb = fig.colorbar(im, ax=axes[1,0])#,label='Jy')
        # cb.ax.set_ylabel('Contrast')
        cb.ax.yaxis.label.set_size(fntsz)
        axes[1,0].invert_yaxis()
        axes[1,0].set_xlabel('RA [arcsec]', fontsize = fntsz)
        axes[1,0].set_ylabel('Dec [arcsec]', fontsize = fntsz)
        axes[1,0].set_title('Coadded Image - No CCD Noise', fontsize = fntsz)

        # First Image:
        im = axes[1,1].imshow(batch['im_coadded_emccd'], extent=[max_fov,-max_fov,max_fov,-max_fov], cmap='hot')
        axes[1,1].set_xlim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        axes[1,1].set_ylim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        cb = fig.colorbar(im, ax=axes[1,1])#,label='Jy')
        cb.ax.set_ylabel('Contrast')
        cb.ax.yaxis.label.set_size(fntsz)
        axes[1,1].invert_yaxis()
        axes[1,1].set_xlabel('RA [arcsec]', fontsize = fntsz)
        # axes[1,1].set_ylabel('Dec [arcsec]', fontsize = fntsz)
        axes[1,1].set_title('Coadded Image - With CCD Noise', fontsize = fntsz)
    
