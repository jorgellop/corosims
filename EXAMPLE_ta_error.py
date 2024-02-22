#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:06:23 2024

@author: llopsayson
"""

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
    
    flag_loop = True
    flag_print = True
    # Output directory
    today = date.today()
    date_str = today.strftime("%Y%m%d")
    label_run = "ta_error_tests_"+date_str
    outdir = "output/ta_error_tests/"+label_run+"/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cgisim_obj = cgisim_sims()
    if True: # Best DMs
        dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_best_contrast_dm1.fits' )
        dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_best_contrast_dm2.fits' )
        cgisim_obj.options['dm1'] = dm1
        cgisim_obj.options['dm2'] = dm2

    if not flag_loop:
        ta_offset_r = 0.25*21.8
        ta_offset_pa = 0
        ta_offset_x = ta_offset_r*np.cos(ta_offset_pa*np.pi/180)
        ta_offset_y = ta_offset_r*np.sin(ta_offset_pa*np.pi/180)
        passvalue_proper = {'source_x_offset_mas':ta_offset_x, 
                                         'source_y_offset_mas':ta_offset_y} 

        ni_im = cgisim_obj.generate_image(source_id=0,flag_return_contrast=True,
                                         jitter_sig_x=.3,jitter_sig_y=.1,
                                         x_ta_offset=0,
                                         y_ta_offset=0,
                                         passvalue_proper=passvalue_proper)
        ni_avg = cgisim_obj.compute_avarage_contrast(ni_im)
        
        # Figures
        sz_im = np.shape(ni_im);zoom_in_pix=9/0.435*1.2
        zoom_in_lim = np.array([np.round(sz_im[0]/2-zoom_in_pix),np.round(sz_im[0]/2+zoom_in_pix)])
        fig = plt.figure(figsize=(6,6))
        plt.imshow(ni_im, cmap='hot')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.gca().invert_yaxis()
        plt.title("TA err. {:.1f}mas ({:.0f}deg), Avg. NI. {:.2E} ".format(ta_offset_r,ta_offset_pa,ni_avg),
                  fontsize = 15)
        plt.xlim(zoom_in_lim)
        plt.ylim(zoom_in_lim)
        # plt.clim([1e-10, 3e-6])
        if flag_print:
            plt.savefig(outdir+'im_dh_TAerror_{:.1f}mas_{}deg.png'.format(ta_offset_r,ta_offset_pa), dpi=1000)

    else:
        #%%
        fntsz = 18
        # Science target star
        flag_no_jitter_conv = True
        if flag_no_jitter_conv:
            label_extra='no_jitter_conv'
        else:
            label_extra='w_jitter_conv'

        ta_offset_r_arr = np.arange(0.1,10,0.6)
        for ta_offset_r in ta_offset_r_arr:
            ta_offset_pa = 0
            ta_offset_x = ta_offset_r*np.cos(ta_offset_pa*np.pi/180)
            ta_offset_y = ta_offset_r*np.sin(ta_offset_pa*np.pi/180)
            if flag_no_jitter_conv:
                passvalue_proper = {'source_x_offset_mas':ta_offset_x, 
                                                 'source_y_offset_mas':ta_offset_y} 
                ta_offset_x_jit = 0
                ta_offset_y_jit = 0
            else:
                ta_offset_x_jit = ta_offset_x
                ta_offset_y_jit = ta_offset_y
                passvalue_proper = None

            ni_im = cgisim_obj.generate_image(source_id=0,flag_return_contrast=True,
                                             jitter_sig_x=.3,jitter_sig_y=.1,
                                             x_ta_offset=ta_offset_x_jit,y_ta_offset=ta_offset_y_jit,
                                             passvalue_proper=passvalue_proper)
        
            r_arr,ni_curve = cgisim_obj.compute_contrast_curve(ni_im)
            me_ni = np.mean(ni_curve)
            
            # Figures
            sz_im = np.shape(ni_im);zoom_in_pix=9/0.435*1.2
            zoom_in_lim = np.array([np.round(sz_im[0]/2-zoom_in_pix),np.round(sz_im[0]/2+zoom_in_pix)])
            fig = plt.figure(figsize=(6,6))
            plt.imshow(ni_im, cmap='hot')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.gca().invert_yaxis()
            plt.title("TA err {:.3f} mas - Avg. NI {:.1e}".format(ta_offset_r,me_ni),fontsize=fntsz)
            plt.xlim(zoom_in_lim)
            plt.ylim(zoom_in_lim)
            plt.clim([1e-10, 1e-6])
            plt.savefig(outdir+'im_dh_'+label_extra+'_TAerr{:.1f}mas'.format(ta_offset_r)+'.png', dpi=1000)
        
            
            fig = plt.figure(figsize=(9,6))
            plt.semilogy(r_arr,ni_curve,color='k',linewidth=2)
            plt.ylim([1e-9, 5e-7])
            plt.xlim([3-0.5, 9+0.5])
            plt.grid(visible=True)
            plt.title("TA err {:.3f} mas - Avg. NI {:.1e}".format(ta_offset_r,me_ni),fontsize=fntsz)
            plt.xlabel('Angular Separation [lambda/D]',fontsize=fntsz)
            plt.ylabel('Normalized Intensity',fontsize=fntsz)
            plt.yticks(fontsize=15);plt.xticks(fontsize=15)
            plt.legend(fontsize=fntsz)
            plt.savefig(outdir+'contrast_curve_'+label_extra+'_TAerr{:.1f}mas'.format(ta_offset_r)+'.png', dpi=1000)
    
        #%% Create gifs
        from PIL import Image
        import glob
        import contextlib
        duration = 700
        
        # filepaths
        fp_in = outdir+"im_dh_"+label_extra+"*.png"
        fp_out = outdir+"gif_im_dh_ta_offset_"+label_extra+".gif"
        with contextlib.ExitStack() as stack:
        
            # lazily load images
            imgs = (stack.enter_context(Image.open(f))
                    for f in sorted(glob.glob(fp_in)))
        
            # extract  first image from iterator
            img = next(imgs)
        
            # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
            img.save(fp=fp_out, format='GIF', append_images=imgs,
                     save_all=True, duration=duration, loop=0)
    
        fp_in = outdir+"contrast_curve_"+label_extra+"*.png"
        fp_out = outdir+"gif_contrast_curve_ta_offset_"+label_extra+".gif"
        with contextlib.ExitStack() as stack:
        
            # lazily load images
            imgs = (stack.enter_context(Image.open(f))
                    for f in sorted(glob.glob(fp_in)))
        
            # extract  first image from iterator
            img = next(imgs)
        
            # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
            img.save(fp=fp_out, format='GIF', append_images=imgs,
                     save_all=True, duration=duration, loop=0)
