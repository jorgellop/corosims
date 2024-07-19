#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:26:55 2024

@author: llopsayson
"""
import cgisim_sims
import matplotlib.pylab as plt

if __name__ == '__main__':
    
    # Select coronagraph and bandpass
    # cor_type = 'hlc_band1'
    # cor_type = 'hlc_band4'
    # cor_type = 'spc-wide'
    cor_type = 'spc-wide_band1'
    
    bandpass='1'
    
    # Initialize object
    corgi = cgisim_sims.corgisims_core(cor_type = cor_type, bandpass=bandpass)

    corgi.define_source('a0v', 2)
    #%% Generate image
    im10 = corgi.generate_image(flag_return_contrast=True,
                                use_fpm=1,zindex=None,zval_m=None,
                                jitter_sig_x=0,jitter_sig_y=0.,
                                passvalue_proper=None,use_emccd=False)
    
    fig = plt.figure(figsize=(6,6))
    plt.imshow(im10, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
