#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:30:13 2024

@author: llopsayson
"""

import numpy as np
from skimage.draw import line_aa
import matplotlib.pylab as plt
from utils import make_circ_mask
from scipy.ndimage import convolve

im = np.zeros((256, 256))
rr, cc, val = line_aa(128, 128, 20, 20)
im[rr, cc] = val 

plt.figure(111)
plt.imshow(im)

npix = 256
X,Y = np.meshgrid(np.arange(-npix/2,npix/2),np.arange(-npix/2,npix/2));

jitter_sig_x = 12
jitter_sig_y = 5
W_jit = np.exp(-0.5*(X**2/jitter_sig_x**2 + Y**2/jitter_sig_y**2))


d_offset_max = 10 #mas
pix_scale = d_offset_max/npix

rad = 15
top_hat = make_circ_mask(npix,0,0,rad)

W_conv0 = convolve(W_jit,top_hat)
W_conv1 = convolve(W_conv0,im)
plt.figure(112)
plt.imshow(W_jit)
plt.figure(113)
plt.imshow(W_conv0)
plt.figure(114)
plt.imshow(W_conv1)
