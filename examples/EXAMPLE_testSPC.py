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
import numpy as np

if __name__ == '__main__':
    
    # Select coronagraph and bandpass
    # cor_type = 'hlc_band1'
    cor_type = 'spc-wide'
    bandpass='4'

    # Initialize object
    cgisim_obj = cgisim_sims(cor_type = cor_type, bandpass=bandpass)

    # dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_best_contrast_dm1.fits' )
    # dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_best_contrast_dm2.fits' )
    dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/spc_wide_band4_best_contrast_dm1.fits' )
    dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/spc_wide_band4_best_contrast_dm2.fits' )
    cgisim_obj.options['dm1'] = dm1
    cgisim_obj.options['dm2'] = dm2

    # stellar_diameter = 4 #mas
    cgisim_obj.sources[0]['star_vmag']=5.04
    cgisim_obj.sources[0]['star_type']='g0v'
    cgisim_obj.sources[0]['name']='47UMa'
    # cgisim_obj.sources[0]['stellar_diameter']= stellar_diameter #mas
    
    #%% Generate image
    passvalue_proper = {'source_x_offset_mas':0,'source_y_offset_mas':0} 

    im10 = cgisim_obj.generate_image(source_id=0,flag_return_contrast=True,
                                     use_fpm=1,zindex=None,zval_m=None,
                                     jitter_sig_x=2,jitter_sig_y=2,
                                     passvalue_proper=passvalue_proper,use_emccd=False)
    
    fig = plt.figure(figsize=(6,6))
    plt.imshow(im10, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
    
    fig = plt.figure(figsize=(6,6))
    plt.imshow(np.log10(im10), cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()

    ni_avg = cgisim_obj.compute_avarage_contrast(im10,iwa=6,owa=20.1)
    print(ni_avg)
