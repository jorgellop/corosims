import cgisim_sims
import matplotlib.pylab as plt
import numpy as np

if __name__ == '__main__':
    
    # Select coronagraph and bandpass
    cor_type = 'hlc_band1'
    # cor_type = 'hlc_band4'
    # cor_type = 'spc-wide'
    # cor_type = 'spc-wide_band1'
    
    bandpass='1'
    
    # Initialize object
    corgi = cgisim_sims.corgisims_core(cor_type = cor_type, bandpass=bandpass)

    corgi.define_source('a0v', 2)
    #%% Define Drift and TA errors
    ta_offset_r = 0 # mas
    ta_offset_pa = 0 # deg
    ta_offset_x = ta_offset_r*np.cos(ta_offset_pa*np.pi/180)
    ta_offset_y = ta_offset_r*np.sin(ta_offset_pa*np.pi/180)

    # Drift is defined as a vector
    # A line function is convolved with the jitter cloud of dEFs
    drift_vector = np.array([1,1])
        

    
    #%% Generate image
    im10 = corgi.generate_image(flag_return_contrast=True,
                                jitter_sig_x=3,jitter_sig_y=3,
                                x_offset_mas=ta_offset_x,
                                y_offset_mas=ta_offset_y,
                                drift_vector=drift_vector)
    
    fig = plt.figure(figsize=(6,6))
    plt.imshow(im10, cmap='hot')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.gca().invert_yaxis()
