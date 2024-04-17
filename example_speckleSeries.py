# S. Hasler
# Run CGISim wrapper and generate scene with increased integration time

from cgisim_sims import cgisim_sims
# import matplotlib.pylab as plt
# import proper
# import roman_phasec_proper
import numpy as np
import astropy.io.fits as pyfits
# from scipy.ndimage import rotate

if __name__ == "__main__":
    cgisim_obj = cgisim_sims()

    name_speckleSeries = "example_speckleSeries_oneSource"

    flag_use_emccd = False

    # Science target star
    star_vmag = 5.04
    cgisim_obj.sources[0]['star_vmag'] = star_vmag
    cgisim_obj.sources[0]['star_type'] = 'g0v'
    cgisim_obj.sources[0]['name'] = '47Uma'

    # Read in jitter and Z4-11 timeseries from OS11 file
    datadir_Z411 = "/Users/sammyh/Codes/cgisim_sims/data/hlc_os11_v2/"
    flnm_Z411 = "hlc_os11_inputs.fits"

    inFile = pyfits.open(datadir_Z411+flnm_Z411)
    hlc_os11_inputs = inFile[0].data

    # Retrieve jitter values
    jitt_sig_x_arr = hlc_os11_inputs[:,78] * 1 # masRMS
    jitt_sig_y_arr = hlc_os11_inputs[:,79] * 1# masRMS

    # Retrieve Z4-11 values
    z411_mat = hlc_os11_inputs[:,46:54]
    
    # Create sccene with LO errors
    cgisim_obj.generate_scene(name=name_speckleSeries, jitter_x=jitt_sig_x_arr, jitter_y=jitt_sig_y_arr,
                              zindex=np.arange(4,11+1), zval_m=z411_mat)
    
    # Initialize schedule index array
    cgisim_obj.scene['schedule']['schedule_index_array'] = []

    # Reference observation
    batch_ID = 0
    num_frames_ref = 12
    sourceid = 0 # what star
    cgisim_obj.scene['schedule']['batches'][0] = {'num_timesteps':num_frames_ref,
                                                  'batch_ID':batch_ID,
                                                  'sourceid':sourceid}
    
    # Generae speckles series for scene
    cgisim_obj.generate_speckleSeries_from_scene(num_images_printed=0, flag_return_contrast=False,
                                                 use_emccd=flag_use_emccd, use_photoncount=False)