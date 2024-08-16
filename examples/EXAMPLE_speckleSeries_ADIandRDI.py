import cgisim_sims
import matplotlib.pylab as plt
import numpy as np
import os

if __name__ == '__main__':
    
    name_speckleSeries = 'example_speckleSeries_ScienceAndRef'
    obs_obj = cgisim_sims.Observation(name=name_speckleSeries)
    
    #%% Define source and scene
    # Science target star
    star_vmag = 2.25
    star_type = 'a0v'
    name_source = 'referenceStar'
    obs_obj.create_source(star_type=star_type,vmag=star_vmag,name=name_source)
    print("You added a source: {}, {}, vmag={}".format(obs_obj.sources[0]['name'],obs_obj.sources[0]['star_type'],obs_obj.sources[0]['vmag']))
    
    
    # Ref target star
    star_vmag = 5.04
    star_type = 'g0v'
    name_source = '47UMa'
    obs_obj.create_source(star_type=star_type,vmag=star_vmag,name=name_source)
    print("You added a source: {}, {}, vmag={}".format(obs_obj.sources[1]['name'],obs_obj.sources[1]['star_type'],obs_obj.sources[1]['vmag']))
    
    # Planet
    vmag = 5.04+8*2.5
    name_source = 'planet'
    obs_obj.create_source(star_type='47UMa',vmag=vmag,name=name_source) # TODO: Planet spectra to be implemented
    print("You added a source: {}, {}, vmag={}".format(obs_obj.sources[0]['name'],obs_obj.sources[0]['star_type'],obs_obj.sources[0]['vmag']))

    # Create a scene, SCI
    obs_obj.create_scene(name='SCI')
    obs_obj.add_point_source_to_scene(scene_name='SCI',source_name='47UMa')
    obs_obj.add_point_source_to_scene(scene_name='SCI',source_name='planet',xoffset_mas=200,yoffset_mas=200)
    # Create another scene, REF
    obs_obj.create_scene(name='REF')
    obs_obj.add_point_source_to_scene(scene_name='REF',source_name='referenceStar')

    #%% Define jitter timeseries
    num_timesteps_batch0 = 3
    jitter_sig_x = 0.5 # mas
    jitter_sig_y = 0.5 # mas
    jitt_sig_x_arr_0 = jitter_sig_x + np.random.normal(0,jitter_sig_x/10,num_timesteps_batch0)
    jitt_sig_y_arr_0 = jitter_sig_y + np.random.normal(0,jitter_sig_y/10,num_timesteps_batch0)
    
    num_timesteps_batch1 = 5
    jitter_sig_x = 2 # mas
    jitter_sig_y = 1 # mas
    jitt_sig_x_arr_1 = jitter_sig_x + np.random.normal(0,jitter_sig_x/10,num_timesteps_batch1)
    jitt_sig_y_arr_1 = jitter_sig_y + np.random.normal(0,jitter_sig_y/10,num_timesteps_batch1)

    num_timesteps_batch2 = 5
    jitter_sig_x = 1.5 # mas
    jitter_sig_y = 1.5 # mas
    jitt_sig_x_arr_2 = jitter_sig_x + np.random.normal(0,jitter_sig_x/10,num_timesteps_batch2)
    jitt_sig_y_arr_2 = jitter_sig_y + np.random.normal(0,jitter_sig_y/10,num_timesteps_batch2)

    plt.figure(111)
    plt.plot(np.concatenate((jitt_sig_x_arr_0,jitt_sig_x_arr_1,jitt_sig_x_arr_2)))
    plt.plot(np.concatenate((jitt_sig_y_arr_0,jitt_sig_y_arr_1,jitt_sig_y_arr_2)))
    
    #%% Create batches
    # Batch 0
    obs_obj.create_batch(scene_name='REF',jitter_x=jitt_sig_x_arr_0,jitter_y=jitt_sig_y_arr_0)
    
    # Batch 1, Roll2: PA=0 deg
    V3PA_roll1 = 0
    obs_obj.create_batch(scene_name='SCI',jitter_x=jitt_sig_x_arr_1,jitter_y=jitt_sig_y_arr_1,V3PA=V3PA_roll1)
    
    # Batch 2, Roll2: PA=15 deg
    V3PA_roll2 = 15
    obs_obj.create_batch(scene_name='SCI',jitter_x=jitt_sig_x_arr_1,jitter_y=jitt_sig_y_arr_1,V3PA=V3PA_roll2)
    #%% Generate speckle series for scene
    obs_obj.generate_speckleSeries(num_images_printed=0,flag_return_contrast=False,
                                                 use_emccd=False,use_photoncount=False,flag_compute_normalization=True)
    
    #%% Add detector noise
    # Define detector
    obs_obj.corgisim.define_emccd(em_gain=5000.0)
    
    num_frames_interp_batch0 = 1000
    obs_obj.batches[0]['num_frames_interp'] =  num_frames_interp_batch0
    obs_obj.batches[0]['exptime'] = 2.0 # sec
    num_frames_interp_batch1 = 200
    obs_obj.batches[1]['num_frames_interp'] =  num_frames_interp_batch1
    obs_obj.batches[1]['exptime'] = 30.0 # sec
    num_frames_interp_batch2 = 200
    obs_obj.batches[2]['num_frames_interp'] =  num_frames_interp_batch2
    obs_obj.batches[2]['exptime'] = 30.0 # sec

    obs_obj.add_detector_noise_to_batches()
    
    #%% plot results
    # Load data
    obs_obj.load_batches_cubes()
    
    batches = obs_obj.batches
    fntsz = 10
    pixel_scale = obs_obj.corgisim.options['pixel_scale']
    max_fov = obs_obj.corgisim.sz_im* pixel_scale / 2
    zoom_pix = 30
    fig_ni, ax_ni = plt.subplots(1,1,figsize=(6, 6)) # For contrast curve
    for II,batch in enumerate(batches):
        # generate figures
        fig, axes = plt.subplots(2, 2,figsize=(2* 4.5, 2 * 3.75))#, dpi=300)
        # First Image:
        im = axes[0,0].imshow(batch['im_cube'][0], extent=[max_fov,-max_fov,max_fov,-max_fov], cmap='hot')
        axes[0,0].invert_yaxis()
        axes[0,0].set_xlim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        axes[0,0].set_ylim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        cb = fig.colorbar(im, ax=axes[0,0])#,label='Jy')
        # cb.ax.set_ylabel('Contrast')
        cb.ax.yaxis.label.set_size(fntsz)
        # axes[0,0].set_xlabel('RA [arcsec]', fontsize = fntsz)
        axes[0,0].set_ylabel('Dec [arcsec]', fontsize = fntsz)
        axes[0,0].set_title('First Image of Timeseries - No CCD Noise', fontsize = fntsz)

        # First Image:
        im = axes[0,1].imshow(batch['im_cube_emccd'][0], extent=[max_fov,-max_fov,max_fov,-max_fov], cmap='hot')
        axes[0,1].invert_yaxis()
        axes[0,1].set_xlim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        axes[0,1].set_ylim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        cb = fig.colorbar(im, ax=axes[0,1])#,label='Jy')
        cb.ax.set_ylabel('Contrast')
        cb.ax.yaxis.label.set_size(fntsz)
        # axes[0,0].set_xlabel('RA [arcsec]', fontsize = fntsz)
        # axes[0,1].set_ylabel('Dec [arcsec]', fontsize = fntsz)
        axes[0,1].set_title('First Image of Timeseries - With CCD Noise', fontsize = fntsz)

        # First Image:
        im = axes[1,0].imshow(batch['im_coadded'], extent=[max_fov,-max_fov,max_fov,-max_fov], cmap='hot')
        axes[1,0].invert_yaxis()
        axes[1,0].set_xlim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        axes[1,0].set_ylim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        cb = fig.colorbar(im, ax=axes[1,0])#,label='Jy')
        # cb.ax.set_ylabel('Contrast')
        cb.ax.yaxis.label.set_size(fntsz)
        axes[1,0].set_xlabel('RA [arcsec]', fontsize = fntsz)
        axes[1,0].set_ylabel('Dec [arcsec]', fontsize = fntsz)
        axes[1,0].set_title('Coadded Image - No CCD Noise', fontsize = fntsz)

        # First Image:
        im = axes[1,1].imshow(batch['im_coadded_emccd'], extent=[max_fov,-max_fov,max_fov,-max_fov], cmap='hot')
        axes[1,1].invert_yaxis()
        axes[1,1].set_xlim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        axes[1,1].set_ylim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
        cb = fig.colorbar(im, ax=axes[1,1])#,label='Jy')
        cb.ax.set_ylabel('Contrast')
        cb.ax.yaxis.label.set_size(fntsz)
        axes[1,1].set_xlabel('RA [arcsec]', fontsize = fntsz)
        # axes[1,1].set_ylabel('Dec [arcsec]', fontsize = fntsz)
        axes[1,1].set_title('Coadded Image - With CCD Noise', fontsize = fntsz)
    
        fig.suptitle('Batch {}'.format(batch['batch_id']), fontsize=fntsz+2)
        fig.savefig(os.path.join(obs_obj.paths["outdir"],'images_batch{}.png'.format(batch['batch_id'])))

        
        
        # Contrast Curve
        ni_im = batch['im_coadded']/ batch['maxI0_offaxis']
        sep_arr,ni_curve = obs_obj.corgisim.compute_contrast_curve(ni_im,iwa=3,owa=9,d_sep=0.5)
        
        # Plot
        ax_ni.plot(sep_arr,ni_curve, label='Batch {}'.format(batch['batch_id']))
    ax_ni.legend(fontsize=fntsz)
    ax_ni.set_xlabel('Angular Separation', fontsize=fntsz)
    ax_ni.set_ylabel('Norm. Int.', fontsize=fntsz)
    ax_ni.grid(True)

    fig_ni.savefig(os.path.join(obs_obj.paths["outdir"],'contrast_curves.png'))
    #%% roll subtract
    from scipy.ndimage import rotate

    im_ref = batches[0]['im_coadded_emccd']
    im_roll1 = batches[1]['im_coadded_emccd']
    im_roll2 = batches[2]['im_coadded_emccd']

    # Subtract ref
    im_sub1 = im_roll1 - im_ref
    im_sub2 = im_roll2 - im_ref
    
    # Derotate
    im1 = rotate(im_sub1,V3PA_roll1, reshape=False)
    im2 = rotate(im_sub2,V3PA_roll2, reshape=False)
    
    # Coadd
    im_fin = (im1+im2)/2
    
    # Plot:
    fig, axes = plt.subplots(1, 1,figsize=(1* 4.5, 1 * 3.75))#, dpi=300)
    im = axes.imshow(im_fin, extent=[max_fov,-max_fov,max_fov,-max_fov], cmap='hot')
    axes.invert_yaxis()
    axes.set_xlim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
    axes.set_ylim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
    cb = fig.colorbar(im, ax=axes)#,label='Jy')
    cb.ax.set_ylabel('counts')
    cb.ax.yaxis.label.set_size(fntsz)
    axes.set_xlabel('RA [arcsec]', fontsize = fntsz)

    axes.set_title('Roll Subtracted', fontsize=fntsz+2)
    fig.savefig(os.path.join(obs_obj.paths["outdir"],'image_subtr.png'))
