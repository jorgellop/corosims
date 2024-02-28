#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:17:13 2024

@author: llopsayson
"""

import numpy as np
from cgisim_sims import cgisim_sims
import astropy.io.fits as pyfits
import matplotlib.pylab as plt




if __name__ == '__main__':
    cgisim_obj = cgisim_sims()
    
    name_scene = 'example_pointSourceCompanions_rollsAndRefv6'
    label_out = 'vEETCpc'
    cgisim_obj.generate_scene(name=name_scene)
    
    
    # Reference observation
    batch_ID = 0
    num_frames_interp = 49
    exptime = 4.8
    cgisim_obj.scene['schedule']['batches'][0] = {'num_frames_interp':num_frames_interp,
                                                     'batch_ID':batch_ID,
                                                     'exptime':exptime}
    
    # Science observation ROLL1
    batch_ID = 100
    num_frames_interp = 49
    exptime = 4.8
    V3PA_roll1 = 13 #roll angle
    cgisim_obj.scene['schedule']['batches'].append({'num_frames_interp':num_frames_interp,
                                                     'batch_ID':batch_ID,
                                                     'exptime':exptime})

    # Science observation ROLL2
    batch_ID = 101
    num_frames_interp = 49
    exptime = 4.8
    V3PA_roll2 = -13 #roll angle
    cgisim_obj.scene['schedule']['batches'].append({'num_frames_interp':num_frames_interp,
                                                     'batch_ID':batch_ID,
                                                     'exptime':exptime})

    #%% Read in cube
    cgisim_obj.load_batches_cubes()
    
    cgisim_obj.define_emccd(em_gain=5000)
    cgisim_obj.add_detector_noise_to_batches()
    #%% roll subtract
    from scipy.ndimage import rotate
    
    datadir = cgisim_obj.scene['outdir']
    outdir_images = datadir
    
    flnm =  'Ii_coadded_emccd_batch0.fits'
    data = pyfits.open(datadir+flnm)
    im_ref = data[0].data
    
    flnm =  'Ii_coadded_emccd_batch100.fits'
    data = pyfits.open(datadir+flnm)
    im_roll1 = data[0].data
    
    flnm =  'Ii_coadded_emccd_batch101.fits'
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
    hdulist.writeto(outdir_images+'im_roll_subtracted_emccd_'+label_out+'.fits',overwrite=True)

    #%% Photon Count    datadir = cgisim_obj.scene['outdir']
    from PhotonCount.corr_photon_count import get_count_rate

    outdir_images = datadir
    
    flnm =  'Ii_cube_emccd_batch0.fits'
    data = pyfits.open(datadir+flnm)
    im_ref = data[0].data
    
    flnm =  'Ii_cube_emccd_batch100.fits'
    data = pyfits.open(datadir+flnm)
    im_roll1 = data[0].data
    
    flnm =  'Ii_cube_emccd_batch101.fits'
    data = pyfits.open(datadir+flnm)
    im_roll2= data[0].data

    # PC:
    photoncount_ref = get_count_rate(im_ref * cgisim_obj.emccd.eperdn - cgisim_obj.emccd.bias, 500, cgisim_obj.emccd.em_gain)
    photoncount_r1 = get_count_rate(im_roll1 * cgisim_obj.emccd.eperdn - cgisim_obj.emccd.bias, 500, cgisim_obj.emccd.em_gain)
    photoncount_r2 = get_count_rate(im_roll2 * cgisim_obj.emccd.eperdn - cgisim_obj.emccd.bias, 500, cgisim_obj.emccd.em_gain)

    # Subtract ref
    im_sub1 = photoncount_r1 - photoncount_ref
    im_sub2 = photoncount_r2 - photoncount_ref
    
    # Derotate
    im1 = rotate(im_sub1,V3PA_roll1)
    im2 = rotate(im_sub2,V3PA_roll2)
    
    # Coadd
    im_fin_pc = (im1+im2)/2
    
    plt.figure(112)
    plt.imshow(im_fin_pc)

    # Save
    hdulist = pyfits.PrimaryHDU(im_fin_pc)
    hdulist.writeto(outdir_images+'im_roll_subtracted_pc_'+label_out+'.fits',overwrite=True)
