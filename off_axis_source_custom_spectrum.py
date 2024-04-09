# S. Hasler
# Run CGISim wrapper and generate scene with off-axis point source that has custom input spectrum
# Custom input spectrum must exist under the same name in the CGISim info_dir folder: 
# /Users/sammyh/.local/lib/python3.9/site-packages/cgisim-3.1-py3.9.egg/cgisim/cgisim_info_dir/

from cgisim_sims import cgisim_sims
import matplotlib.pylab as plt
import proper
import roman_phasec_proper
import numpy as np
import astropy.io.fits as pyfits
from scipy.ndimage import rotate

def run_OS(scene_name, source_type_filename, source_name, planet_xseparation, planet_yseparation):
    '''
    Run observing scenario with input custom source spectra for one off-axis point source.

    Parameters
    ----------
    scene_name : str
        Name for scene file directory
    source_type_filename : str
        Name of source type, which also corresponds to name of .dat spectrum file
    source_name : str
        Name of off-axis point source (can be any string)
    planet_xseparation : float
        x-direction offset for the PSF in mas
    planet_yseparation : float
        y-direction offset for the PSF in mas
    '''

    cgisim_obj = cgisim_sims()
    use_emccd_flag = False

    # Set DM solutions
    dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir + '/examples/hlc_best_contrast_dm1.fits')
    dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir + '/examples/hlc_best_contrast_dm2.fits')
    cgisim_obj.options['dm1'] = dm1
    cgisim_obj.options['dm2'] = dm2

    name_scene = scene_name

    # Science target star
    star_vmag = 5.0
    cgisim_obj.sources[0]['star_vmag'] = star_vmag
    cgisim_obj.sources[0]['star_type'] = 'g0v'
    cgisim_obj.sources[0]['name'] = '~Sun'

    # Reference star
    starref_vmag = 5.04
    cgisim_obj.sources[1]['star_vmag'] = starref_vmag
    cgisim_obj.sources[1]['star_type'] = 'g0v'
    cgisim_obj.sources[1]['name'] = 'rPup'

    # Planet 1
    planet_vmag = 25.56454656787905 # calculated with convert_albedo_spectra.py
    cgisim_obj.sources[2]['star_vmag'] = planet_vmag
    cgisim_obj.sources[2]['star_type'] = source_type_filename
    cgisim_obj.sources[2]['name'] = source_name

    # Read in LOWFE
    datadir_Z411 = "/Users/sammyh/Codes/cgisim_sims/data/hlc_os11_v2/" 
    flnm_Z411 = "hlc_os11_inputs.fits"
    inFile = pyfits.open(datadir_Z411+flnm_Z411)
    hlc_os11_inputs = inFile[0].data

    # Retrieve jitter values
    jitt_sig_x_arr = hlc_os11_inputs[:,78] * 1 # [masRMS]
    jitt_sig_y_arr = hlc_os11_inputs[:,79] * 1 # [masRMS]

    # Retrieve Z4-Z11 values
    z411_mat = hlc_os11_inputs[:,46:54]

    # Create scene
    # Batch IDs -- for naming
    batch_id_os11 = hlc_os11_inputs[:,2]

    # Create scene with LO errors
    cgisim_obj.generate_scene(name=name_scene,jitter_x=jitt_sig_x_arr,jitter_y=jitt_sig_y_arr,
                            zindex=np.arange(4,11+1),zval_m=z411_mat)

    # Initialize schedule_index_array
    cgisim_obj.scene['schedule']['schedule_index_array'] = []

    # Observe reference target
    index_batch_ref = 0
    batch_ID = 0
    num_frames_ref = len(np.where(batch_id_os11==batch_ID)[0])
    sourceid_ref = 1 # choose the star
    V3PA = 0         # roll angle
    exptime = 30     
    cgisim_obj.scene['schedule']['batches'][0] = {'num_timesteps':num_frames_ref,
                                                    'batch_ID':batch_ID,
                                                    'sourceid':sourceid_ref,
                                                    'exptime':exptime,
                                                    'V3PA':V3PA}
    cgisim_obj.scene['schedule']['schedule_index_array'].append(np.ones(num_frames_ref)*index_batch_ref)

    # Science observation roll 1
    index_batch_roll1 = 1
    batch_ID = 100
    num_frames_roll1 = len(np.where(batch_id_os11==batch_ID)[0])
    sourceid_sci = 0    # choose source star
    V3PA_roll1 = 13     # roll angle
    exptime = 30
    cgisim_obj.scene['schedule']['batches'].append({'num_timesteps':num_frames_roll1,
                                                    'batch_ID':batch_ID,
                                                    'sourceid':sourceid_sci,
                                                    'exptime':exptime,
                                                    'V3PA':V3PA_roll1})
    cgisim_obj.scene['schedule']['schedule_index_array'].append(np.ones(num_frames_ref)*index_batch_roll1)

    # Science observation roll 2
    index_batch_roll2 = 2
    batch_ID = 101
    # num_frames_roll2 = 3
    num_frames_roll2 = len(np.where(batch_id_os11==batch_ID)[0])
    sourceid_sci = 0 # what star?
    V3PA_roll2 = -13 #roll angle
    exptime = 30
    cgisim_obj.scene['schedule']['batches'].append({'num_timesteps':num_frames_roll2,
                                                    'batch_ID':batch_ID,
                                                    'sourceid':sourceid_sci,
                                                    'exptime':exptime,
                                                    'V3PA':V3PA_roll2})
    cgisim_obj.scene['schedule']['schedule_index_array'].append(np.ones(num_frames_ref)*index_batch_roll2)

    # Add point source companion to scene
    cgisim_obj.add_point_source_to_scene(sourceid=2, central_sourceid=0, xoffset=planet_xseparation, yoffset=planet_yseparation) # offset in mas

    # Run scene simulation:
    cgisim_obj.generate_speckleSeries_from_scene(num_images_printed=0,flag_return_contrast=True,
                                                use_emccd=use_emccd_flag,use_photoncount=False)

    # Post-processing
    # Get where images are located and set path for outputting images
    datadir = cgisim_obj.scene['outdir']
    outdir_images = datadir

    # Get reference image
    flnm = 'Ii_coadded_batch0.fits'
    data = pyfits.open(datadir + flnm)
    im_ref = data[0].data

    # Get roll 1
    flnm = 'Ii_coadded_batch100.fits'
    data = pyfits.open(datadir + flnm)
    im_roll1 = data[0].data

    # Get roll 2
    flnm = 'Ii_coadded_batch101.fits'
    data = pyfits.open(datadir+flnm)
    im_roll2 = data[0].data

    # Subtract reference from rolls
    im_sub1 = im_roll1 - im_ref
    im_sub2 = im_roll2 - im_ref

    # Derotate
    im1 = rotate(im_sub1, V3PA_roll1)
    im2 = rotate(im_sub2, V3PA_roll2)

    # Co-add
    im_final = (im1 + im2) / 2

    # Write to file
    hdulist = pyfits.PrimaryHDU(im_final)
    hdulist.writeto(outdir_images+'post_processed_wcontrast.fits',overwrite=True)

    # Take a look at post-processed image output
    plt.figure(figsize=(7,3))
    plt.subplot(1,2,1)
    plt.imshow(im_final, origin='lower')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    scene_name = 'custom_spectrum_companion_Jupiter5au_rollsAndRef_wEMCCD'
    source_type_filename = 'Jupiter_1x_5AU_0deg_Angstroms_flam'
    source_name = 'Jupiter_1x_5AU_0deg_Angstroms_flam'

    # Location of point source in DH: For planet 5 AU from its star in a system that is 15 pc away, separation is 0.333304"
    planet_xseparation = 300.3 # [mas] # TODO: change back to 333.3 mas in x,y and add emccd noise to compare to 5AU jupiter w/o emccd 
    planet_yseparation = 300.3 # [mas]

    run_OS(scene_name, source_type_filename, source_name, planet_xseparation, planet_yseparation)
    # TODO: something did not work when I ran this -- output is not correct 