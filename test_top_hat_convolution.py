#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:50:20 2024

@author: llopsayson
"""
import numpy as np    
import matplotlib.pyplot as plt  # To visualize
from utils import make_circ_mask
from scipy.ndimage import convolve
from scipy.interpolate import RegularGridInterpolator
npix = 128
X,Y = np.meshgrid(np.arange(-npix/2,npix/2),np.arange(-npix/2,npix/2));

jitter_sig_x = 12
jitter_sig_y = 5
W_jit = np.exp(-0.5*(X**2/jitter_sig_x**2 + Y**2/jitter_sig_y**2))


d_offset_max = 10 #mas
pix_scale = d_offset_max/npix

rad = 15
top_hat = make_circ_mask(npix,0,0,rad)

W_conv0 = convolve(top_hat,W_jit)

plt.figure(112)
plt.imshow(W_jit)
plt.figure(111)
plt.imshow(W_conv0)

interp = RegularGridInterpolator((X[0,:], Y[:,0]), W_conv0,
                              bounds_error=False, fill_value=None)

npix_new = 55
X,Y = np.arange(-npix_new/2,npix_new/2),np.arange(-npix_new/2,npix_new/2)
W_jit_new = np.zeros((npix_new,npix_new))
for II,x in enumerate(X):
    for JJ,y in enumerate(Y):
        W_jit_new[II,JJ] = interp((x,y))
plt.figure(113)
plt.imshow(W_jit_new)
