#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:26:55 2024

@author: llopsayson
"""
from cgisim_sims import cgisim_sims
import astropy.io.fits as pyfits
import matplotlib.pylab as plt
import numpy as np
import proper
import roman_phasec_proper
import matplotlib.pylab as plt

if __name__ == '__main__':
    

    cgisim_obj = cgisim_sims()
    if True: # Best DMs
        dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_best_contrast_dm1.fits' )
        dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_best_contrast_dm2.fits' )
        cgisim_obj.options['dm1'] = dm1
        cgisim_obj.options['dm2'] = dm2

    
    #%%
    stellar_diameter = 4 #mas
    cgisim_obj.sources[0]['star_vmag']=5.04
    cgisim_obj.sources[0]['star_type']='g0v'
    cgisim_obj.sources[0]['name']='47UMa'
    cgisim_obj.sources[0]['stellar_diameter']= stellar_diameter #mas
    
    
    im10 = cgisim_obj.generate_image(source_id=0,flag_return_contrast=True,
                                     jitter_sig_x=.3,jitter_sig_y=.1,
                                     use_fpm=1,zindex=None,zval_m=None,
                                    passvalue_proper=None,
                                    flag_return_contrast=True,use_emccd=False,exptime=1.0)
