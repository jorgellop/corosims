import corosims
import matplotlib.pylab as plt
import numpy as np
import os

if __name__ == '__main__':
    
    name_speckleSeries = 'example_speckleSeries'
    obs_obj = corosims.Observation(name=name_speckleSeries, cor_type = 'hlc_band1', bandpass='1')
    
    #%% Define source and scene
    # Science target star
    star_vmag = 2.25
    star_type = 'a0v'
    name_source = 'a0vStar'
    obs_obj.create_source(star_type=star_type,vmag=star_vmag,name=name_source)
    print("You added a source: {}, {}, vmag={}".format(obs_obj.sources[0]['name'],obs_obj.sources[0]['star_type'],obs_obj.sources[0]['vmag']))
    
    
    # Create another scene, REF
    obs_obj.create_scene(name='Scene1')
    obs_obj.add_point_source_to_scene(scene_name='Scene1',source_name='a0vStar')

    #%% Define jitter timeseries
    num_timesteps_batch0 = 5
    jitter_sig_x = 0.5 # mas RMS
    jitter_sig_y = 0.5 # mas RMS
    jitt_sig_x_arr_0 = jitter_sig_x + np.random.normal(0,jitter_sig_x/10,num_timesteps_batch0)
    jitt_sig_y_arr_0 = jitter_sig_y + np.random.normal(0,jitter_sig_y/10,num_timesteps_batch0)
    
    num_timesteps_batch1 = 5
    jitter_sig_x = 1 # mas RMS
    jitter_sig_y = 1 # mas RMS
    jitt_sig_x_arr_1 = jitter_sig_x + np.random.normal(0,jitter_sig_x/10,num_timesteps_batch1)
    jitt_sig_y_arr_1 = jitter_sig_y + np.random.normal(0,jitter_sig_y/10,num_timesteps_batch1)

    plt.figure(111)
    plt.plot(np.concatenate((jitt_sig_x_arr_0,jitt_sig_x_arr_1)))
    plt.plot(np.concatenate((jitt_sig_y_arr_0,jitt_sig_y_arr_1)))
    plt.ylabel("Jitter mas RMS")
    plt.xlabel("Timestep")
    
    #%% Create batches
    # Batch 0
    obs_obj.create_batch(scene_name='Scene1',jitter_x=jitt_sig_x_arr_0,jitter_y=jitt_sig_y_arr_0)
    
    # Batch 1
    obs_obj.create_batch(scene_name='Scene1',jitter_x=jitt_sig_x_arr_1,jitter_y=jitt_sig_y_arr_1)
    
    #%% Generate speckle series for scene
    obs_obj.generate_speckleSeries(num_images_printed=0,flag_return_contrast=False,
                                                 use_emccd=False,use_photoncount=False,flag_compute_normalization=True)
    
    #%% plot results
    # Load data
    obs_obj.load_batches_cubes()
    
    batches = obs_obj.batches
    fntsz = 10
    pixel_scale = obs_obj.corosims.options['pixel_scale']
    max_fov = obs_obj.corosims.sz_im* pixel_scale / 2
    zoom_pix = 30
    
    # generate figures
    fig, axes = plt.subplots(1, 3,figsize=(3* 4.5, 1 * 3.75))#, dpi=300)
    # First Image:
    im = axes[0].imshow(batches[0]['im_coadded'], extent=[max_fov,-max_fov,max_fov,-max_fov], cmap='hot')
    axes[0].invert_yaxis()
    axes[0].set_xlim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
    axes[0].set_ylim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
    cb = fig.colorbar(im, ax = axes[0])#,label='Jy')
    # cb.ax.set_ylabel('Contrast')
    cb.ax.yaxis.label.set_size(fntsz)
    axes[0].set_xlabel('RA [arcsec]', fontsize = fntsz)
    axes[0].set_ylabel('Dec [arcsec]', fontsize = fntsz)
    axes[0].set_title('Batch 0', fontsize = fntsz)

    # First Image:
    im = axes[1].imshow(batches[1]['im_coadded'], extent=[max_fov,-max_fov,max_fov,-max_fov], cmap='hot')
    axes[1].invert_yaxis()
    axes[1].set_xlim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
    axes[1].set_ylim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
    cb = fig.colorbar(im, ax=axes[1])#,label='Jy')
    cb.ax.set_ylabel('')
    cb.ax.yaxis.label.set_size(fntsz)
    axes[1].set_xlabel('RA [arcsec]', fontsize = fntsz)
    # axes[0,1].set_ylabel('Dec [arcsec]', fontsize = fntsz)
    axes[1].set_title('Batch 1', fontsize = fntsz)

    # First Image:
    im = axes[2].imshow(batches[1]['im_coadded']-batches[0]['im_coadded'], extent=[max_fov,-max_fov,max_fov,-max_fov], cmap='hot')
    axes[2].invert_yaxis()
    axes[2].set_xlim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
    axes[2].set_ylim((- zoom_pix)*pixel_scale, ( + zoom_pix)*pixel_scale)
    cb = fig.colorbar(im, ax=axes[2])#,label='Jy')
    cb.ax.set_ylabel('e/s')
    cb.ax.yaxis.label.set_size(fntsz)
    axes[2].set_xlabel('RA [arcsec]', fontsize = fntsz)
    axes[2].set_title('Difference', fontsize = fntsz)

    fig.savefig(os.path.join(obs_obj.paths["outdir"],'images_batches_and_diff.png'))
