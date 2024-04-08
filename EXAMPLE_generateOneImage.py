#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:26:55 2024

@author: llopsayson
"""
from cgisim_sims import cgisim_sims
import matplotlib.pylab as plt

if __name__ == '__main__':
    
    # Select coronagraph and bandpass
    # cor_type = 'hlc_band1'
    cor_type = 'spc-wide'
    bandpass='4'
    
    # Initialize object
    cgisim_obj = cgisim_sims(cor_type = cor_type, bandpass=bandpass)

    # stellar_diameter = 4 #mas
    cgisim_obj.sources[0]['star_vmag']=5.04
    cgisim_obj.sources[0]['star_type']='g0v'
    cgisim_obj.sources[0]['name']='47UMa'
    # cgisim_obj.sources[0]['stellar_diameter']= stellar_diameter #mas
    
    #%% Generate image
    im10 = cgisim_obj.generate_image(source_id=0,flag_return_contrast=True,
                                     use_fpm=1,zindex=None,zval_m=None,
                                     jitter_sig_x=0.,jitter_sig_y=0.,
                                     passvalue_proper=None,use_emccd=False)
    
    fig = plt.figure(figsize=(6,6))
    plt.imshow(im10, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
