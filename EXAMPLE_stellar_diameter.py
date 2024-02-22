#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:00:30 2024

@author: llopsayson
"""

from cgisim_sims import cgisim_sims
import astropy.io.fits as pyfits
import matplotlib.pylab as plt
import numpy as np
import proper
import roman_phasec_proper
import matplotlib.pylab as plt
import os
from datetime import date

if __name__ == '__main__':
    
    flag_loop_diameters = False
    # Output directory
    today = date.today()
    date_str = today.strftime("%Y%m%d")
    label_run = "stellar_diameter_tests_"+date_str
    outdir = "output/stellar_diameter_tests/"+label_run+"/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cgisim_obj = cgisim_sims()
    if True: # Best DMs
        dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_best_contrast_dm1.fits' )
        dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_best_contrast_dm2.fits' )
        cgisim_obj.options['dm1'] = dm1
        cgisim_obj.options['dm2'] = dm2

    
    #%%
    # dasdasd
    # Science target star
    stellar_diameter_arr = np.arange(0.1,2.2,0.2)
    for stellar_diameter in stellar_diameter_arr:
        cgisim_obj.sources[0]['star_vmag']=5.04
        cgisim_obj.sources[0]['star_type']='g0v'
        cgisim_obj.sources[0]['name']='47UMa'
        cgisim_obj.sources[0]['stellar_diameter']= stellar_diameter #mas
        
        ni_im = cgisim_obj.generate_image(source_id=0,flag_return_contrast=True,
                                         jitter_sig_x=.3,jitter_sig_y=.1)
    
        r_arr,ni_curve = cgisim_obj.compute_contrast_curve(ni_im)
    
        # Figures
        sz_im = np.shape(ni_im);zoom_in_pix=9/0.435*1.2
        zoom_in_lim = np.array([np.round(sz_im[0]/2-zoom_in_pix),np.round(sz_im[0]/2+zoom_in_pix)])
        fig = plt.figure(figsize=(6,6))
        plt.imshow(ni_im, cmap='hot')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.gca().invert_yaxis()
        plt.title("Stellar diam. {:.3f} mas".format(stellar_diameter))
        plt.xlim(zoom_in_lim)
        plt.ylim(zoom_in_lim)
        plt.clim([1e-10, 3e-8])
        plt.savefig(outdir+'im_dh_{:.3f}mas.png'.format(stellar_diameter), dpi=1000)
    
        
        fntsz = 18
        fig = plt.figure(figsize=(9,6))
        plt.semilogy(r_arr,ni_curve,color='k',linewidth=2)
        plt.ylim([1e-9, 1e-8])
        plt.xlim([3-0.5, 9+0.5])
        plt.grid(visible=True)
        plt.title("Stellar diam. {:.3f} mas".format(stellar_diameter),fontsize=fntsz)
        plt.xlabel('Angular Separation [lambda/D]',fontsize=fntsz)
        plt.ylabel('Normalized Intensity',fontsize=fntsz)
        plt.yticks(fontsize=15);plt.xticks(fontsize=15)
        plt.legend(fontsize=fntsz)
        plt.savefig(outdir+'contrast_curve_diam{:.1f}mas.png'.format(stellar_diameter), dpi=1000)

    #%% Create gifs
    from PIL import Image
    import glob
    import contextlib
    duration = 700
    
    # filepaths
    fp_in = outdir+"im_dh_*.png"
    fp_out = outdir+"gif_im_dh_stellar_diameter.gif"
    with contextlib.ExitStack() as stack:
    
        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in sorted(glob.glob(fp_in)))
    
        # extract  first image from iterator
        img = next(imgs)
    
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=duration, loop=0)

    fp_in = outdir+"contrast_curve_*.png"
    fp_out = outdir+"gif_contrast_curve_stellar_diameter.gif"
    with contextlib.ExitStack() as stack:
    
        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in sorted(glob.glob(fp_in)))
    
        # extract  first image from iterator
        img = next(imgs)
    
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=duration, loop=0)
