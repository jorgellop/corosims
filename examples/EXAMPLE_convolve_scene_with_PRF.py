import corosims
import matplotlib.pylab as plt
import numpy as np
import os
import astropy.io.fits as pyfits

if __name__ == '__main__':
    
    name_speckleSeries = 'example_speckleSeries_ScienceAndRef'
    obs_obj = corosims.Observation(name=name_speckleSeries, cor_type = 'hlc_band1', bandpass='1')

    flnm = '/Users/llopsayson/Documents/Roman/local_testing/data/toyModel_ring_norm_325mas_dr50mas.fits'
    data = pyfits.open(flnm)
    im_scene = data[0].data

    im_conv = obs_obj.convolve_image_with_prf(im_scene, 0.002)
    
    plt.figure()
    plt.imshow(im_conv)
    
    primary_hdu = pyfits.PrimaryHDU(data=im_conv)
    primary_hdu.writeto(os.path.join('/Users/llopsayson/Documents/Roman/local_testing/data/','toyModel_ring_norm_325mas_dr50mas_testconv.fits'),overwrite=True)
