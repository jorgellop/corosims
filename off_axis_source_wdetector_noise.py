# S. Hasler
# Run CGISim wrapper and generate scene with off-axis point source that has custom input spectrum
# Custom input spectrum must exist under the same name in the CGISim info_dir folder: 
# /Users/sammyh/.local/lib/python3.9/site-packages/cgisim-3.1-py3.9.egg/cgisim/cgisim_info_dir/

from cgisim_sims import cgisim_sims
import matplotlib.pylab as plt
import numpy as np
import astropy.io.fits as pyfits
from scipy.ndimage import rotate
from PhotonCount.corr_photon_count import get_count_rate

def add_emccd_to_OS(scene_name, label_out):
    cgisim_obj = cgisim_sims()
    cgisim_obj.scene['name'] = scene_name

    # Reference observation
    batch_ID = 0
    num_frames_interp = 49
    exptime=4.8
    cgisim_obj.scene['schedule']['batches'][0] = {'num_frames_interp':num_frames_interp,
                                                    'batch_ID':batch_ID,
                                                    'exptime':exptime}

    # Science observation roll 1
    batch_ID = 100
    num_frames_interp = 49
    exptime=4.8
    V3PA_roll1 = 13     # roll angle
    cgisim_obj.scene['schedule']['batches'].append({'num_frames_interp':num_frames_interp,
                                                    'batch_ID':batch_ID,
                                                    'exptime':exptime})

    # Science observation roll 2
    batch_ID = 101
    num_frames_interp = 49
    exptime=4.8
    V3PA_roll2 = -13 #roll angle
    cgisim_obj.scene['schedule']['batches'].append({'num_frames_interp':num_frames_interp,
                                                    'batch_ID':batch_ID,
                                                    'exptime':exptime})

    # Read in cube
    cgisim_obj.load_batches_cubes()

    cgisim_obj.define_emccd(em_gain=1000)
    cgisim_obj.add_detector_noise_to_batches()

    # Roll subtract
    # Get where images are located and set path for outputting images
    datadir = cgisim_obj.scene['outdir']
    outdir_images = datadir

    # Get reference image
    flnm = 'Ii_coadded_emccd_batch0.fits'
    data = pyfits.open(datadir + flnm)
    im_ref = data[0].data

    # Get roll 1
    flnm = 'Ii_coadded_emccd_batch100.fits'
    data = pyfits.open(datadir + flnm)
    im_roll1 = data[0].data

    # Get roll 2
    flnm = 'Ii_coadded_emccd_batch101.fits'
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
    hdulist.writeto(outdir_images+'im_roll_subtracted_emccd_'+label_out+'.fits',overwrite=True)

    flnm = 'Ii_cube_emccd_batch0.fits'
    data = pyfits.open(datadir+flnm)
    im_roll1 = data[0].data

    flnm = 'Ii_cube_emccd_batch101.fits'
    data = pyfits.open(datadir + flnm)
    im_roll2 = data[0].data

    # Photon counting
    photoncount_ref = get_count_rate(im_ref * cgisim_obj.emccd.eperdn - cgisim_obj.emccd.bias, 500, cgisim_obj.emccd.em_gain)
    photoncount_r1 = get_count_rate(im_roll1 * cgisim_obj.emccd.eperdn - cgisim_obj.emccd.bias, 500, cgisim_obj.emccd.em_gain)
    photoncount_r2 = get_count_rate(im_roll2 * cgisim_obj.emccd.eperdn - cgisim_obj.emccd.bias, 500, cgisim_obj.emccd.em_gain)

    # Subtract ref
    im_sub1 = photoncount_r1 - photoncount_ref
    im_sub2 = photoncount_r2 - photoncount_ref

    # Derotate
    im1 = rotate(im_sub1, V3PA_roll1)
    im2 = rotate(im_sub2, V3PA_roll2)

    # Co-add
    im_fin_pc = (im1+im2) / 2

    plt.figure(112)
    plt.imshow(im_fin_pc, origin='lower')
    plt.colorbar()
    plt.show()

    # save 
    hdulist = pyfits.PrimaryHDU(im_fin_pc)
    hdulist.writeto(outdir_images+'im_roll_subtracted_pc_'+label_out+'.fits', overwrite=True)

if __name__ == '__main__':
    scene_name = 'custom_spectrum_companion2_rollsandRef_Jup5au'
    label_out = "vEETCpc"

    add_emccd_to_OS(scene_name, label_out)