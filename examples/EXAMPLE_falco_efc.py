import falco

import numpy as np
import os, sys
import copy
import matplotlib.pyplot as plt

import proper

sys.path.append('../../falco-python/roman/')

from EXAMPLE_config_Roman_CGI_HLC_NFOV_Band1 import mp
 # Which optical layout to use

if __name__ == '__main__':
    import corgisims
    
    mp.layout = 'cgisim' 
    mp.cgisim_obj = corgisims.corgisims_core(cor_type = 'hlc_band1', bandpass='1')
    
    mp.Nitr = 2
    # %% Define different directories for data output
    # #
    # # Location of minimal output files.
    # mp.path.brief =   # Default is mp.path.falco + '/data/brief/'
    # #
    # # (Mostly) complete workspace from end of trial.
    # mp.flagSaveWS = False  # Set to True to save the final mp object.
    # mp.path.ws =  # Default is mp.path.falco + '/data/ws/'
    
    
    # %% Overwrite values from config file if desired
    
    # ## Special Computational Settings
    mp.flagPlot = True
    mp.flagParallel = False  # whether to use multiprocessing to parallelize some large computations
    mp.flagParallelJacobian = False
    mp.flagParallel_efc = False
    # mp.Nthreads = 2  # Number of threads to use when using multiprocessing.
    
    # Record Keeping
    mp.TrialNum = 1
    mp.SeriesNum = 9
    
    
    # %% SETTINGS FOR QUICK RUN: SINGLE WAVELENGTH, SINGLE POLARIZATION, AND NO PROBING
    
    mp.fracBW = 0.06608695652173924  # fractional bandwidth of the whole bandpass (Delta lambda / lambda0)
    mp.Nsbp = 3  # Number of sub-bandpasses to divide the whole bandpass into for estimation and control
    mp.Nwpsbp = 1  # Number of wavelengths to used to approximate an image in each sub-bandpass
    mp.full.pol_conds = [10, ]
    mp.estimator = 'perfect'
    # mp.estimator = 'pwp-bp'
    
    
    #%% Use DM solution:
    
    # dm1 = proper.prop_fits_read( "~/Documents/Python/libraries/roman_phasec_v1.4/python/roman_phasec_proper/examples/hlc_best_contrast_dm1.fits" )
    # dm2 = proper.prop_fits_read( "~/Documents/Python/libraries/roman_phasec_v1.4/python/roman_phasec_proper/examples/hlc_best_contrast_dm2.fits" )
    dm1 = proper.prop_fits_read( "~/Documents/Python/libraries/roman_phasec_v1.4/python/roman_phasec_proper/examples/HLC_falco_S1T5_NI1.5e-09_dm1.fits" )
    dm2 = proper.prop_fits_read( "~/Documents/Python/libraries/roman_phasec_v1.4/python/roman_phasec_proper/examples/HLC_falco_S1T5_NI1.5e-09_dm2.fits" )
    
    
    # %% Keep only the central bandpasses's FPM if using just one wavelength with HLC
    
    if (mp.Nsbp == 1) and (mp.coro == 'HLC'):
        n = mp.compact.fpmCube.shape[0]
        nSlices = mp.compact.fpmCube.shape[2]
        mp.compact.fpmCube = mp.compact.fpmCube[:, :, nSlices//2].reshape((n, n, 1))
    
    
    # %% Perform an idealized phase retrieval (get the E-field directly)
    
    optval = copy.copy(mp.full)
    optval.source_x_offset = 0
    optval.use_dm1 = True
    optval.use_dm2 = True
    nout = 1024
    optval.output_dim = 1024
    optval.use_fpm = False
    optval.use_pupil_mask = False  # No SPM for getting initial phase
    optval.use_lyot_stop = False
    optval.use_field_stop = False
    optval.use_pupil_lens = True
    delattr(optval, 'final_sampling_lam0')
    
    # Use non-SPC flat maps for SPC since SPM has separate aberrations
    # downstream that can't be fully captured at entrance pupil with the SPM in
    # place. The SPM aberrations are flattened in a separate step not included
    # here.
    if 'sp' in mp.coro.lower():
        optval.dm1_m = mp.full.dm1.flatmapNoSPM
        optval.dm2_m = mp.full.dm2.flatmapNoSPM
    else:
        # optval.dm1_m = mp.full.dm1.flatmap
        # optval.dm2_m = mp.full.dm2.flatmap
        optval.dm1_m = dm1 + mp.full.dm1.flatmap
        optval.dm2_m = dm2 + mp.full.dm2.flatmap
    
    if mp.Nsbp == 1:
        lambdaFacs = np.array([1.])
    else:
        lambdaFacs = np.linspace(1-mp.fracBW/2, 1+mp.fracBW/2, mp.Nsbp)
    
    # Get the Input Pupil's E-field
    nCompact = falco.util.ceil_even(mp.P1.compact.Nbeam + 1)
    mp.P1.compact.E = np.ones((nCompact, nCompact, mp.Nsbp), dtype=complex)
    for iSubband in range(mp.Nsbp):
    
        lambda_um = 1e6*mp.lambda0*lambdaFacs[iSubband]
    
        # Get aberrations for the full optical train
        optval.pinhole_diam_m = 0  # 0 means don't use the pinhole at FPAM
        fieldFullAll, sampling = proper.prop_run('roman_phasec', lambda_um, nout, QUIET=True, PASSVALUE=optval.__dict__)
    
        # Put pinhole at FPM to get back-end optical aberrations
        optval.pinhole_diam_m = mp.F3.pinhole_diam_m;
        fieldFullBackEnd, sampling = proper.prop_run('roman_phasec', lambda_um, nout, QUIET=True, PASSVALUE=optval.__dict__)
        optval.pinhole_diam_m = 0  # 0 means don't use the pinhole at FPAM
    
        # Subtract off back-end phase aberrations from the phase retrieval estimate
        phFrontEnd = np.angle(fieldFullAll) - np.angle(fieldFullBackEnd)
        # swMask = ampthresh(fieldFullAll)
        # phFrontEnd, _ = removeZernikes(phFrontEnd, [0 1 1], [0 1 -1], swMask)  # Remove tip/tilt/piston
    
        # Put front-end E-field into compact model
        fieldFull = np.abs(fieldFullAll) * np.exp(1j*phFrontEnd)
        fieldCompactReal = falco.mask.rotate_shift_downsample_pupil_mask(
            np.real(fieldFull), mp.P1.full.Nbeam, mp.P1.compact.Nbeam, 0, 0, 0)
        fieldCompactImag = falco.mask.rotate_shift_downsample_pupil_mask(
            np.imag(fieldFull), mp.P1.full.Nbeam, mp.P1.compact.Nbeam, 0, 0, 0)
        fieldCompact = fieldCompactReal + 1j*fieldCompactImag
        fieldCompact = falco.util.pad_crop(fieldCompact, (nCompact, nCompact))
        mp.P1.compact.E[:, :, iSubband] = falco.prop.relay(fieldCompact, 1, centering=mp.centering)
    
        if mp.flagPlot:
            plt.figure(11); plt.imshow(np.angle(fieldCompact)); plt.colorbar(); plt.hsv(); plt.pause(1e-2)
            plt.figure(12); plt.imshow(np.abs(fieldCompact)); plt.colorbar(); plt.magma(); plt.pause(0.5)
    
    # Don't double count the pupil amplitude with the phase retrieval and a model-based mask
    mp.P1.compact.mask = np.ones_like(mp.P1.compact.mask)
    
    
    # %% Generate the label associated with this trial
    mp.runLabel = ('Series%04d_Trial%04d_%s' %
                   (mp.SeriesNum, mp.TrialNum, mp.coro))
    print(mp.runLabel)
    
    
    # %% Perform the Wavefront Sensing and Control
    mp.dm1.V = (dm1 - mp.full.dm1.flatmap)*1e9 #+ np.random.normal(0,0.1,(48,48))
    mp.dm2.V = (dm2 - mp.full.dm2.flatmap)*1e9      
    
    
    out = falco.setup.flesh_out_workspace(mp)
    
    # out.dm1.Vall[:,:,0] = dm1*1e9
    # out.dm2.Vall[:,:,0] = dm2*1e9
    
    falco.wfsc.loop(mp, out)
    
    
    # %% Plot the output
    falco.plot.plot_trial_output(out)
    
    # Or, load and plot the output data from pickled data
    fnPickle = os.path.join(mp.path.brief, (mp.runLabel + '_snippet.pkl'))
    falco.plot.plot_trial_output_from_pickle(fnPickle)
