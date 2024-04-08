#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:26:55 2024

@author: llopsayson
"""
from cgisim_sims import cgisim_sims
import matplotlib.pylab as plt
import roman_phasec_proper
import proper
from utils import crop_data,make_circ_mask

if __name__ == '__main__':
    
    # Select coronagraph and bandpass
    cor_type = 'hlc_band4'
    # cor_type = 'spc-wide_band1'
    bandpass='4'
    
    flag_print = True
    crop_sz = None
    outdir_images = 'output/resport_generatingDMs_wFALCO/'
    
    # Generate mask
    sampling_hlc = {'hlc_band1':{'1':0.435},
                    'hlc_band4':{'4':0.303},
                    'spc-wide':{'4':0.303,'1':0.435},
                    'spc-wide_band1':{'4':0.303,'1':0.435}} #TODO
    iwa = 3
    owa = 9.7
    iwa_mask = make_circ_mask(201,0,0,iwa/sampling_hlc[cor_type][bandpass])
    owa_mask = make_circ_mask(201,0,0,owa/sampling_hlc[cor_type][bandpass])
    mask_field = owa_mask-iwa_mask

    #%% Nominal DM 
    # Initialize object
    cgisim_obj = cgisim_sims(cor_type = cor_type, bandpass=bandpass)
    
    dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/spc_wide_band4_mild_contrast_dm1.fits' )
    dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/spc_wide_band4_mild_contrast_dm2.fits' )
    cgisim_obj.options['dm1'] = dm1
    cgisim_obj.options['dm2'] = dm2

    # stellar_diameter = 4 #mas
    cgisim_obj.sources[0]['star_vmag']=5.04
    cgisim_obj.sources[0]['star_type']='g0v'
    cgisim_obj.sources[0]['name']='47UMa'
    # cgisim_obj.sources[0]['stellar_diameter']= stellar_diameter #mas
    
    #% Generate image
    im1 = cgisim_obj.generate_image(source_id=0,flag_return_contrast=True,
                                     use_fpm=1,zindex=None,zval_m=None,
                                     jitter_sig_x=0.,jitter_sig_y=0.,
                                     passvalue_proper=None,use_emccd=False)
    
    if crop_sz is not None:
        im1 = crop_data(im1,crop_sz)
    fig = plt.figure(figsize=(6,6))
    plt.imshow(im1, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
    if flag_print:
        plt.savefig(outdir_images+'imdh_'+cor_type+'_band'+bandpass+'_nominalDMsolution.png')
    fig = plt.figure(figsize=(6,6))
    plt.imshow(im1*mask_field, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
    if flag_print:
        plt.savefig(outdir_images+'imdhmasked_'+cor_type+'_band'+bandpass+'_nominalDMsolution.png')
        

    fig = plt.figure(figsize=(6,6))
    plt.imshow(cgisim_obj.options['dm1'], cmap='grey')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
    plt.title('DM1',fontsize=22)
    if flag_print:
        plt.savefig(outdir_images+'dm1_'+cor_type+'_band'+bandpass+'_nominalDMsolution.png')
    fig = plt.figure(figsize=(6,6))
    plt.imshow(cgisim_obj.options['dm2'], cmap='grey')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
    plt.title('DM2',fontsize=22)
    if flag_print:
        plt.savefig(outdir_images+'dm2_'+cor_type+'_band'+bandpass+'_nominalDMsolution.png')
    
    #%% FALCO DM
    datadir_dms = 'data/dm_maps/'
    # dm1 = proper.prop_fits_read( datadir_dms+'/HLC_falco_S1T5_NI1.5e-09_dm1.fits' )
    # dm2 = proper.prop_fits_read( datadir_dms+'/HLC_falco_S1T5_NI1.5e-09_dm2.fits' )
    # dm1 = proper.prop_fits_read( datadir_dms+'/SPLC_falco_S6T2_NI3.9e-09_dm1.fits' )
    # dm2 = proper.prop_fits_read( datadir_dms+'/SPLC_falco_S6T2_NI3.9e-09_dm2.fits' )
    # dm1 = proper.prop_fits_read( datadir_dms+'/SPLC_falco_S7T1_NI5.0e-09_dm1.fits' )
    # dm2 = proper.prop_fits_read( datadir_dms+'/SPLC_falco_S7T1_NI5.0e-09_dm2.fits' )
    dm1 = proper.prop_fits_read( datadir_dms+'/HLC_falco_S8T3_NI6.5e-09_dm1.fits' )
    dm2 = proper.prop_fits_read( datadir_dms+'/HLC_falco_S8T3_NI6.5e-09_dm2.fits' )
    cgisim_obj.options['dm1'] = dm1
    cgisim_obj.options['dm2'] = dm2

    #% Generate image
    im2 = cgisim_obj.generate_image(source_id=0,flag_return_contrast=True,
                                     use_fpm=1,zindex=None,zval_m=None,
                                     jitter_sig_x=0.,jitter_sig_y=0.,
                                     passvalue_proper=None,use_emccd=False)
    if crop_sz is not None:
        im2 = crop_data(im2,crop_sz)
    fig = plt.figure(figsize=(6,6))
    plt.imshow(im2, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
    if flag_print:
        plt.savefig(outdir_images+'imdh_'+cor_type+'_band'+bandpass+'_falcoDMsolution.png')
    fig = plt.figure(figsize=(6,6))
    plt.imshow(im2*mask_field, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
    if flag_print:
        plt.savefig(outdir_images+'imdhmasked_'+cor_type+'_band'+bandpass+'_falcoDMsolution.png')
    fig = plt.figure(figsize=(6,6))
    plt.imshow(cgisim_obj.options['dm1'], cmap='grey')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
    plt.title('DM1',fontsize=22)
    if flag_print:
        plt.savefig(outdir_images+'dm1_'+cor_type+'_band'+bandpass+'_falcoDMsolution.png')
    fig = plt.figure(figsize=(6,6))
    plt.imshow(cgisim_obj.options['dm2'], cmap='grey')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
    plt.title('DM2',fontsize=22)
    if flag_print:
        plt.savefig(outdir_images+'dm2_'+cor_type+'_band'+bandpass+'_falcoDMsolution.png')
