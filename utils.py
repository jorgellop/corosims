#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:33:14 2023

@author: llopsayson
"""
import numpy as np

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def make_circ_mask(npix,dx,dy,rad):
    X,Y = np.meshgrid(np.arange(-npix/2,npix/2),np.arange(-npix/2,npix/2)); 
    RHO,THETA = cart2pol(X-dx, Y-dy)
    mask = np.zeros((npix, npix))
    mask[RHO<rad]=1
    return mask

def crop_data(data, nb_pixels=50,centx=None,centy=None):
    if centx == None:
        NP = int(data.shape[0]/2)
        centx = NP
        centy = NP
    else:
        centx = int(centx)
        centy = int(centy)
        # centx = int(np.round(centx))
        # centy = int(np.round(centy))
    return data[centx-nb_pixels:centx+nb_pixels,centy-nb_pixels:centy+nb_pixels]

def degenPA(dx, dy):
    rat_pos = dx/dy
    if rat_pos>0:
        if dx>0:
            PA = (np.arctan((rat_pos)))#*180/np.pi
        else:
            PA = (np.arctan(np.abs(rat_pos))+np.pi)#*180/np.pi
    else:
        if dx>0:
            PA = (np.pi-np.arctan(np.abs(rat_pos)))#*180/np.pi
        else:
            PA = (2*np.pi-np.arctan(np.abs(rat_pos)))#*180/np.pi
    
    return (PA*180/np.pi )
