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
    if True: # Best DMs
        dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_best_contrast_dm1.fits' )
        dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_best_contrast_dm2.fits' )
        cgisim_obj.options['dm1'] = dm1
        cgisim_obj.options['dm2'] = dm2

    name_scene = 'example_pointSourceCompanions_rollsAndRefv6'
    
    flag_use_emccd = False
    
    # Science target star
    star_vmag = 5.04
    cgisim_obj.sources[0]['star_vmag']=star_vmag
    cgisim_obj.sources[0]['star_type']='g0v'
    cgisim_obj.sources[0]['name']='47UMa'
    
    # Ref star
    # starref_vmag = 2.25
    starref_vmag = 5.04
    cgisim_obj.sources[1]['star_vmag']=starref_vmag
    cgisim_obj.sources[1]['star_type']='g0v'
    cgisim_obj.sources[1]['name']='rPup'
    
    # planet
    starref_vmag = star_vmag + 8*2.5
    cgisim_obj.sources[2]['star_vmag']=starref_vmag
    cgisim_obj.sources[2]['star_type']='g0v'
    cgisim_obj.sources[2]['name']='47UMab'

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
        
    # Batch IDs
    batch_id_os11 = hlc_os11_inputs[:,2]
    
    #%% Create scene with LO errors
    cgisim_obj.generate_scene(name=name_scene,jitter_x=jitt_sig_x_arr,jitter_y=jitt_sig_y_arr,
                              zindex=np.arange(4,11+1),zval_m=z411_mat)
    
    # Initialize schedule_index_array
    cgisim_obj.scene['schedule']['schedule_index_array'] = []
    
    # Reference observation
    index_batch_ref = 0
    batch_ID = 0
    # num_frames_ref = 3
    num_frames_ref = len(np.where(batch_id_os11==batch_ID)[0])
    sourceid_ref = 1 # what star?
    V3PA = 0 #roll angle
    exptime = 30
    cgisim_obj.scene['schedule']['batches'][0] = {'num_timesteps':num_frames_ref,
                                                     'batch_ID':batch_ID,
                                                     'sourceid':sourceid_ref,
                                                     'exptime':exptime,
                                                     'V3PA':V3PA}
    cgisim_obj.scene['schedule']['schedule_index_array'].append(np.ones(num_frames_ref)*index_batch_ref)
    
    # Science observation ROLL1
    index_batch_roll1 = 1
    batch_ID = 100
    # num_frames_roll1 = 3
    num_frames_roll1 = len(np.where(batch_id_os11==batch_ID)[0])
    sourceid_sci = 0 # what star?
    V3PA_roll1 = 13 #roll angle
    exptime = 30
    cgisim_obj.scene['schedule']['batches'].append({'num_timesteps':num_frames_roll1,
                                                     'batch_ID':batch_ID,
                                                     'sourceid':sourceid_sci,
                                                     'exptime':exptime,
                                                     'V3PA':V3PA_roll1})
    cgisim_obj.scene['schedule']['schedule_index_array'].append(np.ones(num_frames_ref)*index_batch_roll1)

    # Science observation ROLL2
    index_batch_roll2 = 2
    batch_ID = 101
    # num_frames_roll2 = 3
    num_frames_roll2 = len(np.where(batch_id_os11==batch_ID)[0])
    sourceid_sci = 0 # what star?
    V3PA_roll2 = -13 #roll angle
    exptime = 30
    cgisim_obj.scene['schedule']['batches'].append({'num_timesteps':num_frames_roll2,
                                                     'batch_ID':batch_ID,
                                                     'sourceid':sourceid_sci,
                                                     'exptime':exptime,
                                                     'V3PA':V3PA_roll2})
    cgisim_obj.scene['schedule']['schedule_index_array'].append(np.ones(num_frames_ref)*index_batch_roll2)

    #%% Add ppoint source companion
    cgisim_obj.add_point_source_to_scene(sourceid=2,central_sourceid=0,xoffset=200,yoffset=200)
    
    #%% Generate speckle series for scene
    # cgisim_obj.define_emccd(em_gain=5000)

    cgisim_obj.generate_speckleSeries_from_scene(num_images_printed=0,flag_return_contrast=False,
                                                 use_emccd=False,use_photoncount=False)
    
    
    #%% roll subtract
    from scipy.ndimage import rotate # TODO: move to top imports? - SH
    
    datadir = cgisim_obj.scene['outdir']
    outdir_images = datadir
    
    flnm =  'Ii_coadded_batch0.fits'
    data = pyfits.open(datadir+flnm)
    im_ref = data[0].data
    
    flnm =  'Ii_coadded_batch100.fits'
    data = pyfits.open(datadir+flnm)
    im_roll1 = data[0].data
    
    flnm =  'Ii_coadded_batch101.fits'
    data = pyfits.open(datadir+flnm)
    im_roll2= data[0].data

    # Subtract ref
    im_sub1 = im_roll1 - im_ref
    im_sub2 = im_roll2 - im_ref
    
    # Derotate
    im1 = rotate(im_sub1,V3PA_roll1)
    im2 = rotate(im_sub2,V3PA_roll2)
    
    # Coadd
    im_fin = (im1+im2)/2
    
    plt.figure(112)
    plt.imshow(im_fin)

    # Save
    hdulist = pyfits.PrimaryHDU(im_fin)
    hdulist.writeto(outdir_images+'im_roll_subtracted_example.fits',overwrite=True)

    