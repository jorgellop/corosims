#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:48:34 2024

@author: llopsayson
"""

import matplotlib.pylab as plt
from PIL import Image
import astropy.io.fits as pyfits
from utils import crop_data
import os
import glob
import contextlib
vmin_fig = None
vmax_fig = None

datadir = '/Users/llopsayson/Documents/Python/cgisim_sims/output/SpeckleSeries/example_pointSourceCompanions_rollsAndRefv6/'
outdir0 = datadir+'images/'
if not os.path.exists(outdir0):
    os.makedirs(outdir0)
# cubes for which you want individual gifs
flnm_cube_list = ['Ii_cube_emccd_batch0',
                  'Ii_cube_emccd_batch100',
                  'Ii_cube_emccd_batch101']
for flnm_cube in flnm_cube_list:
    outdir_images = outdir0+flnm_cube+'/'
    if not os.path.exists(outdir_images):
        os.makedirs(outdir_images)
    
    data = pyfits.open(datadir+flnm_cube+".fits")
    im_cube = data[0].data
    num_frames_limit = 120#len(im_cube)
    for II,imII in enumerate(im_cube):
        if II>num_frames_limit:
            #% Create figures & Write png images
            Ii_crop = crop_data(imII, nb_pixels=int(32))
            fig = plt.figure(figsize=(6,6))
            plt.imshow(Ii_crop, cmap='hot',vmin=vmin_fig,vmax=vmax_fig)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.gca().invert_yaxis()
            plt.title(flnm_cube)
            plt.savefig(outdir_images+'ni_im{}.png'.format(II+1))
            plt.close(fig)
        # sasdasdad
    #%% Create gifs

    # filepaths
    fp_in = outdir_images+"ni_im*.png"
    fp_out = outdir0+"gif_"+flnm_cube+".gif"
    
    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:
    
        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in sorted(glob.glob(fp_in)))
    
        # extract  first image from iterator
        img = next(imgs)
    
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=200, loop=0)
