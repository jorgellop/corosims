#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:25:33 2023

@author: llopsayson
"""
from cgisim_sims import cgisim_sims
import matplotlib.pylab as plt
import os
from PhotonCount.corr_photon_count import get_count_rate
import numpy as np
import astropy.io.fits as pyfits

if __name__ == '__main__':
    outdir = 'output/emccd_tests/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    cgisim_obj = cgisim_sims()
    
    # Change star default
    name_label = '47UMa'
    star_vmag = 11.00
    cgisim_obj.sources[0]['star_vmag']=star_vmag
    cgisim_obj.sources[0]['star_type']='g0v'
    cgisim_obj.sources[0]['name']=name_label
    
    #%% No detector Image
    jitter_sig_x=0
    jitter_sig_y=0
    im10 = cgisim_obj.generate_image(flag_return_contrast=False,jitter_sig_x=jitter_sig_x,jitter_sig_y=jitter_sig_y)
    
    # fig = plt.figure(figsize=(6,6))
    # plt.imshow(im1, cmap='hot')
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.gca().invert_yaxis()
    
    #%% Define emccd
    cgisim_obj.define_emccd(em_gain=5000.0)
    
    flag_compute_all_images = False
    
    fact = 1
    num_frames=int(75872*1/fact)*1
    exptime = 79  *fact# sec
    
    im_fin = im10*0
    Ii_cube = []
    for II in range(num_frames):    
        #% Image with emccd
        if flag_compute_all_images:
            im1 = cgisim_obj.generate_image(flag_return_contrast=False,use_emccd=True,exptime=exptime)
        else:
            im1 = cgisim_obj.add_detector_noise(im10, exptime)
        im_fin = im_fin+im1/num_frames
        Ii_cube.append(im1)
        
    flag_dark = True
    if flag_dark:
        num_frames_dark = 50
        for II in range(num_frames_dark): 
            imdark =+ cgisim_obj.add_detector_noise(im10*0, exptime)
    else:
        imdark = 0
    Ii_cube = np.stack(Ii_cube)#-imdark
    #%%
    photoncount_im = get_count_rate(Ii_cube * cgisim_obj.emccd.eperdn - cgisim_obj.emccd.bias, 4000, cgisim_obj.emccd.em_gain)
    
    # fig = plt.figure(figsize=(6,6))
    # plt.imshow(im1, cmap='hot')
    # plt.colorbar(fraction=0.046, pad=0.04)
    # plt.gca().invert_yaxis()
    # plt.title("One Frame")
    # plt.savefig(outdir+'im_one_frame_faint.png')
    fig = plt.figure(figsize=(6,6))
    plt.imshow(im_fin, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
    plt.title("emccd - Vmag {}, tint {}s, nframes {} (tot. t. {}h)".format(int(star_vmag),exptime,num_frames,int(exptime*num_frames/3600)))
    plt.savefig(outdir+'im_'+name_label+'_vmag{}_tint{}s_nframes{}.png'.format(int(star_vmag),exptime,num_frames))

    fig = plt.figure(figsize=(6,6))
    plt.imshow(photoncount_im, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
    plt.title("Phot. Count - Vmag {}, tint {}s, nframes {} (tot. t. {}h)".format(int(star_vmag),exptime,num_frames,int(exptime*num_frames/3600)))
    plt.savefig(outdir+'im_photCount_'+name_label+'_vmag{}_tint{}s_nframes{}.png'.format(int(star_vmag),exptime,num_frames))
    
    # hdulist = pyfits.PrimaryHDU(im10)
    # hdulist.writeto(outdir+'test_im0_v3.fits',overwrite=True)
