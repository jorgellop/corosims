import corosims
import matplotlib.pylab as plt

if __name__ == '__main__':
    
    # Select coronagraph and bandpass
    cor_type = 'hlc_band1'
    # cor_type = 'hlc_band4'
    # cor_type = 'spc-wide'
    # cor_type = 'spc-wide_band1'
    
    bandpass='1'
    
    # Initialize object
    corgi = corosims.corosims_core(cor_type=cor_type, bandpass=bandpass)

    corgi.define_source('a0v', 2, # star_type,vmag
                        stellar_diam=2) #mas
    #%% Generate image
    im10 = corgi.generate_image(flag_return_contrast=True,
                                x_offset_mas=0,y_offset_mas=0,
                                use_fpm=1,zindex=None,zval_m=None,
                                jitter_sig_x=0,jitter_sig_y=0,
                                use_emccd=False)
    
    
    #%% Plot
    
    pixel_scale = corgi.options['pixel_scale']
    max_fov = corgi.sz_im* pixel_scale / 2

    fig = plt.figure(figsize=(6,6))
    plt.imshow(im10, cmap='hot', extent=[max_fov,-max_fov,max_fov,-max_fov])
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
    plt.clim(1e-10,1e-8)
    zoom_pix=80
    pixel_scale = 0.0218
    plt.xlim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
    plt.ylim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
    plt.xlabel('RA [arcsec]', fontsize = 16)
    plt.ylabel('Dec [arcsec]', fontsize = 16)

