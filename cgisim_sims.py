#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:35:42 2023

@author: llopsayson
"""
import roman_phasec_proper
import proper
import cgisim
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pylab as plt
import os
from datetime import date
from emccd_detect.emccd_detect import EMCCDDetectBase
from PhotonCount.corr_photon_count import get_count_rate
from scipy.ndimage import convolve
from scipy.interpolate import RegularGridInterpolator
from skimage.draw import line_aa

from utils import make_circ_mask,crop_data,degenPA

class cgisim_sims():
    # =============================================================================
    # cgisim_sims
    # 
    # class to simulate cgi
    # Only excam images
    # =============================================================================
    
    def __init__(self, cor_type = 'hlc_band1', bandpass='1'):
        
        # Define the 4 main options of cgisim as class attributes
        self.cor_type = cor_type
        self.bandpass = bandpass # 
        self.cgi_mode = 'excam_efield' # Always work in EF mode
        self.polaxis = -10 # Work on mean polarization mode, but no polarization is also acceptable.
        
        self.sz_im = 201 # This is replaced everytime an EF is computed
        
        # Define all other options in the options dictionary
        self.options = {}
             
        # Predefine options
        dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_mild_contrast_dm1.fits' )
        dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/hlc_mild_contrast_dm2.fits' )
        # SH Changed for SPC WFOV
        # dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/spc_wide_band4_best_contrast_dm1.fits' )
        # dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir+'/examples/spc_wide_band4_best_contrast_dm2.fits' )
        self.options['dm1'] = dm1
        self.options['dm2'] = dm2

        # Predefine sources
        source1 = {'name':'default_star1',
                'star_type': 'a0v',
                'star_vmag':2.0}
        source2 = {'name':'default_star2',
                'star_type': 'a0v',
                'star_vmag':2.0}
        source3 = {'name':'default_planet2',
                'star_type': None,
                'star_vmag':10.0}
        self.sources = [source1,
                        source2,
                        source3]

    def compute_EF(self,source_id,source_name=None,use_fpm=1,zindex=None,zval_m=None,passvalue_proper=None):
        # =============================================================================
        # compute_offaxis_normalization
        # 
        # Computes an offaxis PSF with the current DM config to be used for normalization
        # =============================================================================
        dm1 = self.options['dm1'] 
        dm2 = self.options['dm2'] 
        params = {'use_errors':1, 'use_dm1':1, 'dm1_m':dm1, 'use_dm2':1, 'dm2_m':dm2, 'use_fpm':use_fpm}
        if zindex is not None:
            params['zindex'] = zindex
            params['zval_m'] = zval_m
        if passvalue_proper is not None:
            params.update(passvalue_proper)
        EF, counts = cgisim.rcgisim( self.cgi_mode, self.cor_type, self.bandpass, self.polaxis, params) 
        
        # Source's spectrum
        if ~('counts_spectrum' in self.sources[source_id]):
            self.compute_spectrum(source_id)
        counts_spectrum = self.sources[source_id]['counts_spectrum']
        EF_new = []
        for II,counts in enumerate(counts_spectrum):
            # EF[II] = EF[II] * np.sqrt(counts)
            EF_new.append(EF[II] * np.sqrt(counts))
            # EF[II] = EF[II] * (counts)
            # import pdb 
            # pdb.set_trace()

        
        self.sz_im = np.shape(EF)[1]
        return EF_new

    def compute_offaxis_normalization(self,source_id=0,source_name=None):
        # =============================================================================
        # compute_offaxis_normalization
        # 
        # Computes an offaxis PSF with the current DM config to be used for normalization
        # =============================================================================
        print("Computing normalization factor")
        EF0 = self.compute_EF(source_id,use_fpm=0)
                    
        I0 = np.abs(EF0)**2 
        I0 = np.sum(I0,axis=0)
        self.sources[source_id]['maxI0_offaxis'] = np.max(I0)
        
        

    def compute_spectrum(self,source_id):
        # =============================================================================
        # compute_spectrum
        # 
        # Computes spectrum in counts for a given source
        # output of cgisim_integrate_spectrum is photons/cm^2/sec over bandpass
        # =============================================================================
        star_vmag = self.sources[source_id]['star_vmag']
        star_type = self.sources[source_id]['star_type']
        
        info_dir = cgisim.lib_dir + '/cgisim_info_dir/'
        mode_data, bandpass_data = cgisim.cgisim_read_mode( self.cgi_mode, self.cor_type, self.bandpass, info_dir )
    
        if self.polaxis != 10 and self.polaxis != -10 and self.polaxis != 0:
            polarizer_transmission = 0.45
        else:
            polarizer_transmission = 1.0
    
        nlam = bandpass_data["nlam"]
        lam_um = np.linspace( bandpass_data["minlam_um"], bandpass_data["maxlam_um"], nlam ) 
        
        nd = 0 # ND filter
        
        dlam_um = lam_um[1] - lam_um[0]
        # total_counts = 0.0
        counts_spectrum = []
        for ilam in range(nlam):
            lam_start_um = lam_um[ilam] - 0.5 * dlam_um
            lam_end_um = lam_um[ilam] + 0.5 * dlam_um
            bandpass_i = 'lam' + str(lam_start_um*1000) + 'lam' + str(lam_end_um*1000)
            counts = polarizer_transmission * cgisim.cgisim_get_counts( star_type, self.bandpass, bandpass_i, nd, star_vmag, 'V', 'excam', info_dir )
            # total_counts += counts
            counts_spectrum.append(counts)
        self.sources[source_id]['counts_spectrum'] = counts_spectrum

    def generate_image(self,source_id=0,use_fpm=1,jitter_sig_x=0,jitter_sig_y=0,zindex=None,zval_m=None,
                       passvalue_proper=None,
                       flag_return_contrast=True,use_emccd=False,exptime=1.0,x_ta_offset=0,y_ta_offset=0,
                       drift_vector=None):
        # =============================================================================
        # generate_image
        # 
        # Generate image wiht user defined aberrations
        # =============================================================================
        EF = self.compute_EF(source_id,use_fpm=use_fpm,zindex=zindex,zval_m=zval_m,passvalue_proper=passvalue_proper)
        
        # Get normalization factor for contrast
        if flag_return_contrast:
            if not ('maxI0_offaxis' in self.sources[source_id]):
                self.compute_offaxis_normalization(source_id=source_id)
            normalization = self.sources[source_id]['maxI0_offaxis']
        else:
            normalization = 1
        sz_im = self.sz_im
        
        # Add jitter if necessary
        if jitter_sig_x!=0 or jitter_sig_y!=0 or 'stellar_diameter' in self.sources[source_id]:
            
            # Read in the jitter dictonary if not yet done.
            if not ('jitter_dict' in self.options):
                try:
                    self.read_in_jitter_deltaEFs()
                except:
                    print("Error occured, did you read in the jitter dictionary from the right folder?")
            
            # Parameters from jitter_dict
            dEF_mat = self.options['jitter_dict']['dEF_mat']
            A_arr = self.options['jitter_dict']['A_arr']
            x_jitt_offset_mas_arr = self.options['jitter_dict']['x_jitt_offset_mas_arr']
            y_jitt_offset_mas_arr = self.options['jitter_dict']['y_jitt_offset_mas_arr']
            # num_jitt = self.options['jitter_dict']['num_jitt']
            
            WA_jit = np.zeros(len(x_jitt_offset_mas_arr))
            
            
            npix = 256
            X,Y = np.meshgrid(np.arange(-npix/2,npix/2),np.arange(-npix/2,npix/2));

            d_offset_max = np.max(np.array([np.max(np.abs(y_jitt_offset_mas_arr)),np.max(np.abs(x_jitt_offset_mas_arr))])) #mas
            pix_scale = (d_offset_max*2)/npix
            X,Y = X*pix_scale,Y*pix_scale
            W_jit = np.exp(-0.5*((X-x_ta_offset)**2/jitter_sig_x**2 + (Y-y_ta_offset)**2/jitter_sig_y**2))

            # Stellar diamter: top hat convolution
            if 'stellar_diameter' in self.sources[source_id]:    
                rad = self.sources[source_id]['stellar_diameter']/2
                top_hat = make_circ_mask(npix,0,0,rad/pix_scale)
                W_jit = convolve(top_hat,W_jit)
    
                # plt.figure(112)
                # plt.imshow(W_jit)
                # plt.figure(111)
                # plt.imshow(W_conv0)
                # import pdb 
                # pdb.set_trace()
            if drift_vector is not None:
                im_line_segment = np.zeros((npix, npix))
                start_vect_x = int(npix/2+x_ta_offset/pix_scale)
                start_vect_y = int(npix/2+y_ta_offset/pix_scale)
                fin_vect_x = int(npix/2+x_ta_offset/pix_scale + drift_vector[0]/pix_scale)
                fin_vect_y = int(npix/2+y_ta_offset/pix_scale + drift_vector[1]/pix_scale)
                rr, cc, val = line_aa(start_vect_x, start_vect_y, fin_vect_x, fin_vect_y)
                im_line_segment[rr, cc] = val 
                W_jit = convolve(im_line_segment,W_jit)
                # plt.figure(111)
                # plt.imshow(im_line_segment)
                # plt.figure(112)
                # plt.imshow(W_jit)
                # import pdb 
                # pdb.set_trace()

            interp = RegularGridInterpolator((X[0,:], Y[:,0]), W_jit,
                                          bounds_error=False, fill_value=None)

            
            for II,(x,y,A) in enumerate(zip(x_jitt_offset_mas_arr,y_jitt_offset_mas_arr,A_arr)):
                WA_jit[II] = interp((x,y)) * A
            # else:
            #     for II,(x,y,A) in enumerate(zip(x_jitt_offset_mas_arr,y_jitt_offset_mas_arr,A_arr)):
            #         WA_jit[II] = np.exp(-0.5*(x**2/jitter_sig_x**2 + y**2/jitter_sig_y**2)) * A
            
            # Normalize W*A function
            WA_jit_norm = np.sum(WA_jit)

            # Add jitter
            # Normalize W*A function
            # WA_jit_norm = 0
            # for x,y,A in zip(x_jitt_offset_mas_arr,y_jitt_offset_mas_arr,A_arr):
            #     WA_jit_norm = WA_jit_norm + np.exp(-0.5*(x**2/jitter_sig_x**2 + y**2/jitter_sig_y**2)) * A
            
            if WA_jit_norm!=0:
                # import pdb 
                # pdb.set_trace()
                counts_spectrum = self.sources[source_id]['counts_spectrum']
    
                # Loop over all jitter positions with W function
                Ii_sum = np.zeros((sz_im,sz_im))
                for KK,(x,y,A,dEF0) in enumerate(zip(x_jitt_offset_mas_arr,y_jitt_offset_mas_arr,A_arr,dEF_mat)):
                    WA_jit_fun =  WA_jit[KK]/ WA_jit_norm
                    
                    dEF = dEF0*0
                    for II,counts in enumerate(counts_spectrum):
                        dEF[II] = dEF0[II] * np.sqrt(counts)
                        
                    Ii_lam = np.abs(EF + dEF)**2 
                    Ii = np.sum(Ii_lam,axis=0)
                    Ii_sum = Ii_sum + Ii * WA_jit_fun 
            else: # jitter is very small
                Ii = np.abs(EF)**2
                Ii_sum = np.sum(Ii,axis=0)
        else: # No jitter:
            Ii = np.abs(EF)**2
            Ii_sum = np.sum(Ii,axis=0)
        Isum = Ii_sum/normalization
        # import pdb 
        # pdb.set_trace()

        if use_emccd:
            if not hasattr(self, "emccd"):
                print("Creating an emccd_detect object with default parameters")
                print("If you'd like other parameters use class function define_emccd()")
                self.define_emccd()
            if flag_return_contrast:
                print("Ignoring the fact that you requested contrast, will return emccd image")
                Isum = Isum*normalization
            Isum = self.add_detector_noise(Isum,exptime)

        return Isum
    
    def add_detector_noise(self,Im, exptime):
        # =============================================================================
        # add_detector_noise
        # 
        # Read in jitter params and EFs
        # =============================================================================
        # fix flux to ph/sec
        # pix_area = self.emccd_dict['pix_area'] * 10*10 #cm2
        # Im = Im#*pix_area
        Im_noisy = self.emccd.sim_sub_frame(Im, exptime).astype(float)
        return Im_noisy
    
    def read_in_jitter_deltaEFs(self,datadir_jitt0='data/jitter_EFs_and_data/'):
        # =============================================================================
        # read_in_jitter_deltaEFs
        # 
        # Read in jitter params and EFs
        # =============================================================================
        print("Reading in the jitter parameters and EFs into a cube")

        datadir_jitt = datadir_jitt0+'cor_type_'+self.cor_type+'_banpass'+self.bandpass+'_polaxis'+str(self.polaxis)+ '/fields/'

        # Read in files 
        inFile = pyfits.open(datadir_jitt+'offsets_mas.fits')
        offsets_mat = inFile[0].data
        x_jitt_offset_mas_arr = offsets_mat[0,:]
        y_jitt_offset_mas_arr = offsets_mat[1,:]
        num_jitt = len(y_jitt_offset_mas_arr)
        
        # Read the A (areas) list
        inFile = pyfits.open(datadir_jitt+'A_areas.fits')
        A_arr = inFile[0].data

        # Read in EF0
        inFile = pyfits.open(datadir_jitt+'EF0_real.fits')
        EF0_real = inFile[0].data
        inFile = pyfits.open(datadir_jitt+'EF0_imag.fits')
        EF0_imag = inFile[0].data
        EF0 = EF0_real * 1j*EF0_imag

        # Generate cube by reading in all jitters
        dEF_mat = []
        for II in range(num_jitt):
            # Read in EF0
            inFile = pyfits.open(datadir_jitt+'EF{}_real.fits'.format(II+1))
            EFII_real = inFile[0].data
            inFile = pyfits.open(datadir_jitt+'EF{}_imag.fits'.format(II+1))
            EFII_imag = inFile[0].data
            EFII = EFII_real + 1j*EFII_imag
            
            # Delta EF
            dEF = EFII-EF0
            dEF_mat.append(dEF)
        
        
        # Save all in the options dictionary
        self.options['jitter_dict'] = {'dEF_mat':dEF_mat,
                                       'A_arr':A_arr,
                                       'x_jitt_offset_mas_arr':x_jitt_offset_mas_arr,
                                       'y_jitt_offset_mas_arr':y_jitt_offset_mas_arr,
                                       'num_jitt':num_jitt}

    def compute_jitter_EFs(self,outdir0='data/jitter_EFs_and_data/'):
        # =============================================================================
        # compute_jitter_EFs
        # 
        # Compute jitter EF could and other parameters as in Krist et al. 2023
        # =============================================================================
        # Define sections
        # As in Krist+2023
        print('Computing all EFs for jitter a la Krist et al. 2023, this will take a lot of time!')

        d_ang1 = 0.15 # mas
        r1_lim = 0.6 # mas
        
        r_arr1 = np.arange(0,r1_lim,d_ang1)+d_ang1
        
        d_ang2 = np.logspace(0., 1.4, num=10)*0.15
        
        # Number of annuli
        num_annuli1 = len(r_arr1) 
        num_annuli2 = len(d_ang2)
        
        # init x_arr and y_arr
        x_arr = []
        y_arr = []
        A_arr = []
        
        # Loop over annuli and populate position arrays
        # Section 1
        th0 = 0
        for II,rII in enumerate(r_arr1):
            num_points = int(2*np.pi*rII / d_ang1)
            dth = 2*np.pi / num_points
            for JJ in range(num_points):
                x_arr.append(rII * np.sin(JJ*dth+th0)) 
                y_arr.append(rII * np.cos(JJ*dth+th0)) 
                if II==0:
                    A_arr.append(np.pi*(rII+(r_arr1[II+1]-rII)/2)**2/num_points)
                elif II<num_annuli1-1:
                    A_arr.append(np.pi*((rII+(r_arr1[II+1]-rII)/2)**2 - (rII-(rII-r_arr1[II-1])/2)**2)/num_points)
                else:
                    A_arr.append(np.pi*((rII+(d_ang2[0])/2)**2 - (rII-(rII-r_arr1[II-1])/2)**2)/num_points)
            th0 = th0 + dth/2
            
        # Section 2
        r0 = r_arr1[-1]
        for II,d_ang in enumerate(d_ang2):
            num_points = int(2*np.pi*(d_ang+r0) / d_ang)
            dth = 2*np.pi / num_points
            for JJ in range(num_points):
                x_arr.append((d_ang+r0) * np.sin(JJ*dth + th0)) 
                y_arr.append((d_ang+r0) * np.cos(JJ*dth + th0)) 
                rII = d_ang+r0
                if II<num_annuli2-1:
                    A_arr.append(np.pi*((rII+(d_ang2[II+1])/2)**2 - (rII-(rII-r0)/2)**2)/num_points)
                else:
                    A_arr.append(np.pi*((rII+(d_ang)/2)**2 - (rII-(rII-r0)/2)**2)/num_points)

            r0 = (d_ang+r0)
            th0 = th0+ dth/2
        
        
        cgi_mode = self.cgi_mode
        cor_type = self.cor_type
        bandpass = self.bandpass
        polaxis = self.polaxis
        outdir = outdir0+'cor_type_'+cor_type+'_banpass'+bandpass+'_polaxis'+str(polaxis)+'/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            os.makedirs(outdir+'fields/')

        # Save A array
        A_arr = A_arr/np.sum(A_arr)
        hdulist = pyfits.PrimaryHDU(A_arr)
        hdulist.writeto(outdir+'fields/A_areas.fits',overwrite=True)
        
        # Plot cloud like in Krist+2023
        fntsz = 18
        
        plt.figure(figsize=(6*1.5,6*1.5))
        plt.scatter(x_arr,y_arr,marker="+",s=15)
        plt.xlabel('X [mas]',fontsize=fntsz);plt.xticks(fontsize=fntsz)
        plt.ylabel('Y [mas]',fontsize=fntsz);plt.yticks(fontsize=fntsz)
        plt.grid()
        plt.title('Jitter Points {}'.format(len(x_arr)),fontsize=fntsz)
        plt.savefig(outdir+'jittercloud.png', dpi=500)

        #% Generate cube of delta-EFs
        # Generate EF0, centered, no errors
                
        dm1 = self.options['dm1'] 
        dm2 = self.options['dm2'] 

        
        
        print( "Computing on-axis PSF - No T/T error" )
        params = {'use_errors':1, 'use_dm1':1, 'dm1_m':dm1, 'use_dm2':1, 'dm2_m':dm2}
        EF0, counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params,
                                    output_file = outdir+'fields/EF0_on-axisPSF_noErrors') 
        
        # Loop for all cloud positions and populate cube
        source_x_offset_mas_arr = x_arr
        source_y_offset_mas_arr = y_arr
        for II,(source_x_offset_mas,source_y_offset_mas) in enumerate(zip(source_x_offset_mas_arr,source_y_offset_mas_arr)):
            print('Propagating jitter realization {0}/{1}.'.format(II+1,len(x_arr)))
            params = {'use_errors':1, 'use_dm1':1, 'dm1_m':dm1, 'use_dm2':1, 'dm2_m':dm2, 
                      'source_x_offset_mas':source_x_offset_mas, 'source_y_offset_mas':source_y_offset_mas} 
            EF, counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params,
                                        output_file = outdir+'fields/EF{}'.format(II+1))
        hdulist = pyfits.PrimaryHDU(np.array([source_x_offset_mas_arr,source_y_offset_mas_arr]))
        hdulist.writeto(outdir+'fields/offsets_mas.fits',overwrite=True)
        
        print('Done computing all EFs for jitter')
        
    def generate_scene(self,name=None,jitter_x=0,jitter_y=0,num_timesteps=None,zindex=None,zval_m=None,source_schedule=None,
                       passvalue_proper=None,exptime=None,bin_schedule=None):
        # =============================================================================
        # generate_scene
        # 
        # source_schedule: array of length len(jitter_x) with indexes for sources
        #
        # Generate
        # =============================================================================
        if num_timesteps is None:
            num_timesteps = np.size(jitter_x)
            
        self.scene = { 
            'num_timesteps' : num_timesteps,
            'jitter_x': jitter_x,
            'jitter_y': jitter_y}
        
        if zindex is not None:
            # Check lengths
            sz_zs = np.shape(zval_m)
            num_zs = sz_zs[0]
            if num_zs!=num_timesteps:
                print('Number of jitters and other errors should match!')
                return
            self.scene['zindex'] = zindex
            self.scene['zval_m'] = zval_m
        else:
            self.scene['zindex'] = None
            self.scene['zval_m'] = [None]*num_timesteps

        if passvalue_proper is not None:
            self.scene['passvalue_proper'] = passvalue_proper
        else:
            self.scene['passvalue_proper'] = None

        if source_schedule is None:
            self.scene['source_schedule'] = np.ones(num_timesteps)*0
        else:
            self.scene['source_schedule'] = source_schedule
            
        # if exptime is None:
        #     self.scene['exptime'] = np.ones(num_timesteps)*1.0
        # else:
        #     self.scene['exptime'] = exptime
        if exptime is None:
            exptime = 1.0
            
        if source_schedule is None:
            self.scene['bin_schedule'] = None
        else:
            self.scene['bin_schedule'] = bin_schedule
        
        # By default, there's only one batch with sourceid 0
        self.scene['schedule'] = {'schedule_index_array':np.zeros(num_timesteps),
                                  'batches':[]}
        self.scene['schedule']['batches'].append({'num_timesteps':num_timesteps,
                                                         'batch_ID':0,
                                                         'sourceid':0,
                                                         'exptime':exptime,
                                                         'V3PA':0})

        # TODO: how to add sources
        self.scene['count_other_sources'] = 0
        self.scene['other_sources'] = []
        
        if name is None:
            today = date.today()
            date_str = today.strftime("%Y%m%d")
            name='OS_'+date_str
        self.scene['name'] = name
        self.scene['outdir'] = 'output/SpeckleSeries/'+self.scene['name']+'/'
        
    def generate_speckleSeries_from_scene(self,outdir0='output/SpeckleSeries/',num_images_printed=0,
                                          vmin_fig=None,vmax_fig=None,title_fig='',
                                          use_emccd=False,use_photoncount=False,flag_return_contrast=True):
        # =============================================================================
        # generate_speckleSeries_from_scene
        # 
        # Generate speckle series based on scene as defined through class variable self.scene.
        # =============================================================================
        '''
        Generate speckle series based on scene as defined through self.scene class variable.

        Parameters
        ----------
        outdir0 : str, optional
            Directory to place output files, by default 'output/SpeckleSeries/'
        num_images_printed : int, optional
            Number of images to output, by default 0 ## TODO: confirm definition
        vmin_fig : float, optional
            vmin parameter for matplotlib.pyplot.imshow(), by default None
        vmax_fig : float, optional
            vmax parameter for matplotlib.pyplot.imshow(), by default None
        title_fig : str, optional
            Title for figure, by default ''
        use_emccd : bool, optional
            Whether or not to add emccd noise, by default False
        use_photoncount : bool, optional
            Whether or not to use photon counting mode (only when using emccd), by default False
        flag_return_contrast : bool, optional
            Whether or not to return the contrast of poitn source companion, by default True ## TODO: confirm definition

        '''
        outdir = self.scene['outdir']#outdir0+self.scene['name']+'/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outdir_images = outdir+'images/'
        if not os.path.exists(outdir_images):
            os.makedirs(outdir_images)

        if not hasattr(self,'scene'):
            print('You havent defined a scene')
            return []
        print("Starting the simulation of scene '"+self.scene['name']+"'")

        # num_timesteps = self.scene['num_timesteps']
        # source_schedule = self.scene['source_schedule'].astype(int)
        jitter_x = self.scene['jitter_x']
        jitter_y = self.scene['jitter_y']
        zindex = self.scene['zindex']
        zval_m = self.scene['zval_m']
        passvalue_proper = self.scene['passvalue_proper']
        # exptime = self.scene['exptime']
        # bin_schedule = self.scene['bin_schedule']
        
        schedule = self.scene['schedule']
        num_batches = len(schedule['batches'])
            
        flag_use_emccd = False # SH added
        if use_emccd:
            flag_return_contrast = False
            flag_use_emccd = True # SH added 
            
        sz_im = self.sz_im
        
        # Generate mask
        sampling_hlc = {'hlc_band1':{'1':0.435}} #TODO
        iwa = 3
        owa = 9
        iwa_mask = make_circ_mask(sz_im,0,0,iwa/sampling_hlc['hlc_band1']['1'])
        owa_mask = make_circ_mask(sz_im,0,0,owa/sampling_hlc['hlc_band1']['1'])
        mask_field = owa_mask-iwa_mask
        
        count_timestep = 0 # total timesteps for all batches
        for KK in range(num_batches):
            batch = schedule['batches'][KK]
            exptime = batch['exptime']     
            num_timesteps  = batch['num_timesteps']
            
            Ii_cube = np.zeros((num_timesteps,sz_im,sz_im))
            Ii_mask_cube = np.zeros((num_timesteps,sz_im,sz_im))
            
            # Generate other point sources if other sources in scene are defined, TODO: only for point sources
            if self.scene['count_other_sources']>0:
                flag_other_sources = True
                count_other_sources = self.scene['count_other_sources']
                
                Ii_othersources = np.zeros((sz_im,sz_im))
                # For each of the defined additional point sources
                for II in range(count_other_sources):
                    # Get the ID of the central source in the image
                    central_sourceid = self.scene['other_sources'][II]['central_sourceid']
                    if batch['sourceid']==central_sourceid:
                        V3PA  = batch['V3PA'] # eg, Roll angle
                        # Get separation of point source from central source
                        xoffset_othersources = self.scene['other_sources'][II]['xoffset']
                        yoffset_othersources = self.scene['other_sources'][II]['yoffset']
                        sep_source = np.sqrt(xoffset_othersources**2+yoffset_othersources**2)
                        PA = degenPA(-xoffset_othersources,yoffset_othersources)
                        xoffset_othersources = -sep_source*np.sin((PA+V3PA)*np.pi/180)
                        yoffset_othersources = sep_source*np.cos((PA+V3PA)*np.pi/180)
                        passvalue_proper_othersources = {'source_x_offset_mas':xoffset_othersources, 
                                                         'source_y_offset_mas':yoffset_othersources} 
    
                        sourceid = self.scene['other_sources'][II]['sourceid']
                        Ii_sourceII = self.generate_image(source_id=sourceid,use_fpm=1,
                                            jitter_sig_x=0,jitter_sig_y=0, #TODO: we assume no jitter
                                            passvalue_proper=passvalue_proper_othersources,
                                            use_emccd=flag_use_emccd,flag_return_contrast=flag_return_contrast)
                        if not ('maxI0_offaxis' in self.sources[central_sourceid]):
                            self.compute_offaxis_normalization(source_id=central_sourceid)
                        if flag_return_contrast:
                            Ii_sourceII =  Ii_sourceII*self.sources[sourceid]['maxI0_offaxis']/self.sources[central_sourceid]['maxI0_offaxis']
                        Ii_othersources = Ii_othersources + Ii_sourceII
                        # import pdb 
                        # pdb.set_trace()
            else:
                flag_other_sources = False

            
            for II in range(num_timesteps):
                print("Computing image num. {} out of {}".format(II+1,batch['num_timesteps']))
                Ii_sum = self.generate_image(source_id=batch['sourceid'],use_fpm=1,
                                    jitter_sig_x=jitter_x[count_timestep],jitter_sig_y=jitter_y[count_timestep],
                                    zindex=zindex,zval_m=zval_m[count_timestep],passvalue_proper=passvalue_proper,
                                    use_emccd=flag_use_emccd,flag_return_contrast=flag_return_contrast)
                if flag_other_sources:
                    # hdulist = pyfits.PrimaryHDU(Ii_sum)
                    # hdulist.writeto(outdir+'Ii_sum.fits',overwrite=True)
                    # hdulist = pyfits.PrimaryHDU(Ii_othersources)
                    # hdulist.writeto(outdir+'Ii_othersources.fits',overwrite=True)
                    Ii_sum = Ii_sum + Ii_othersources
                    # dasdasdads
                    
                # Add detector noise
                if use_emccd:
                    Ii_sum = self.add_detector_noise(Ii_sum,exptime)
                    
                Ii_cube[II,:,:] = Ii_sum
                Ii_mask_cube[II,:,:] = Ii_sum*mask_field
                count_timestep = count_timestep + 1 
                
                if II<num_images_printed:
                    Ii_crop = crop_data(Ii_sum*mask_field, nb_pixels=int(owa/sampling_hlc['hlc_band1']['1']*1.4))
                    fig = plt.figure(figsize=(6,6))
                    plt.imshow(Ii_crop, cmap='hot',vmin=vmin_fig,vmax=vmax_fig)
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.gca().invert_yaxis()
                    plt.title(title_fig)
                    plt.savefig(outdir_images+'ni_im{}.png'.format(II+1))
                    plt.close(fig)
        
        # if bin_schedule is not None:
        #     num_bins = len(bin_schedule)
        #     Ii_mask_cube_bin = np.zeros((num_bins,sz_im,sz_im))
        #     Ii_cube_bin = np.zeros((num_bins,sz_im,sz_im))
        #     init_frameII = 0
        #     for II in range(num_bins):
        #         Ii_II = np.mean(Ii_cube[init_frameII:(init_frameII+bin_schedule[II])],axis=0)
        #         Ii_cube_bin[II] = Ii_II
        #         Ii_mask_II = np.mean(Ii_mask_cube[init_frameII:(init_frameII+bin_schedule[II])],axis=0)
        #         Ii_mask_cube_bin[II] = Ii_mask_II

        #         if II<num_images_printed:
        #             Ii_crop = crop_data(Ii_mask_II, nb_pixels=int(owa/sampling_hlc['hlc_band1']['1']*1.4))
        #             fig = plt.figure(figsize=(6,6))
        #             plt.imshow(Ii_crop, cmap='hot',vmin=vmin_fig,vmax=vmax_fig)
        #             plt.colorbar(fraction=0.046, pad=0.04)
        #             plt.gca().invert_yaxis()
        #             plt.title(title_fig)
        #             plt.savefig(outdir_images+'ni_im_mask_{}_binned.png'.format(II+1))
        #             plt.close(fig)
        #             fig = plt.figure(figsize=(6,6))
        #             plt.imshow(Ii_II, cmap='hot',vmin=vmin_fig,vmax=vmax_fig)
        #             plt.colorbar(fraction=0.046, pad=0.04)
        #             plt.gca().invert_yaxis()
        #             plt.title(title_fig)
        #             plt.savefig(outdir_images+'ni_im{}_binned.png'.format(II+1))
        #             plt.close(fig)
        #     hdulist = pyfits.PrimaryHDU(Ii_cube_bin)
        #     hdulist.writeto(outdir+'Ii_cube_bin.fits',overwrite=True)
        #     hdulist = pyfits.PrimaryHDU(Ii_mask_cube_bin)
        #     hdulist.writeto(outdir+'Ii_mask_cube_bin.fits',overwrite=True)
     
            if use_photoncount:
                if 'photoncount_thresh' in self.scene:
                    thresh = self.scene['photoncount_thresh']
                else:
                    thresh = 500.  
                if not use_emccd:
                   print('PhotonCount is only good for emccd images')
                photoncount_im = get_count_rate(Ii_cube * self.emccd.eperdn - self.emccd.bias, thresh, self.emccd.em_gain)
                # Save
                hdulist = pyfits.PrimaryHDU(photoncount_im)
                hdulist.writeto(outdir+'Ii_photoncount.fits',overwrite=True)
                fig = plt.figure(figsize=(6,6))
                plt.imshow(photoncount_im, cmap='hot',vmin=vmin_fig,vmax=vmax_fig)
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.gca().invert_yaxis()
                plt.title(title_fig)
                plt.savefig(outdir_images+'photoncount_im.png')
                plt.close(fig)
            
            # Write FITS file
            hdulist = pyfits.PrimaryHDU(Ii_cube)
            hdulist.writeto(outdir+'Ii_cube_batch{}.fits'.format(batch['batch_ID']),overwrite=True)
            hdulist = pyfits.PrimaryHDU(Ii_mask_cube)
            hdulist.writeto(outdir+'Ii_mask_cube_batch{}.fits'.format(batch['batch_ID']),overwrite=True)
            Ii_coadded = np.mean(Ii_cube,axis=0)
            Ii_coadded_mask = np.mean(Ii_mask_cube,axis=0)
            hdulist = pyfits.PrimaryHDU(Ii_coadded)
            hdulist.writeto(outdir+'Ii_coadded_batch{}.fits'.format(batch['batch_ID']),overwrite=True)
            hdulist = pyfits.PrimaryHDU(Ii_coadded_mask)
            hdulist.writeto(outdir+'Ii_coadded_mask_batch{}.fits'.format(batch['batch_ID']),overwrite=True)
        
    def add_point_source_to_scene(self,central_sourceid=0,sourceid=None,vmag=5,xoffset=1,yoffset=1,name='noname',
                                  spectral_type='a0v'):
        # =============================================================================
        # add_point_source_to_scene
        # 
        # Add a point source to the scene
        # =============================================================================
        '''
        Adds a point source to the scene

        Parameters
        ----------
        central_sourceid : int, optional
            ID for the host star, by default 0
        sourceid : int, optional
            ID for the planet, by default None
        vmag : int, optional
            Vmag of the host star, by default 5
        xoffset : int, optional
            x-direction offset in [mas], by default 1 (TODO: confirm units)
        yoffset : int, optional
            y-direction offset in [mas], by default 1 (TODO: confirm units)
        name : str, optional
            Name of point source companion, by default 'noname'
        spectral_type : str, optional
            Host star spectral type, by default 'a0v'
        '''
        self.scene['count_other_sources']+=1
        if sourceid is None: # Then source is defined in this function
            self.sources.append({'name':name,
                                'star_type': spectral_type,
                                'star_vmag':vmag})
            sourceid = len(self.sources)-1
            
        self.scene['other_sources'].append({'sourceid':sourceid,
                                            'central_sourceid':central_sourceid,
                                            'is_point_source':True,
                                            'xoffset':xoffset,
                                            'yoffset':yoffset})
        
        
    def define_emccd(self,em_gain=1000.0, full_well_image=60000.0, full_well_serial=100000.0,
                     dark_rate=0.00056, cic_noise=0.01, read_noise=100.0, bias=0,qe=1.0, cr_rate=0, 
                     pixel_pitch=13e-6, e_per_dn=1.0, numel_gain_register=604, nbits=14,
                     use_traps = False,date4traps=2028.0):
        # =============================================================================
        # define_emccd
        # 
        # Define the emccd object
        # =============================================================================
        emccd = EMCCDDetectBase( em_gain=em_gain, full_well_image=full_well_image, full_well_serial=full_well_serial,
                             dark_current=dark_rate, cic=cic_noise, read_noise=read_noise, bias=bias,
                             qe=qe, cr_rate=cr_rate, pixel_pitch=pixel_pitch, eperdn=e_per_dn,
                             numel_gain_register=numel_gain_register, nbits=nbits )

        
        if use_traps: 
            from cgisim.rcgisim import model_for_Roman
            from arcticpy.roe import ROE
            from arcticpy.ccd import CCD
            traps = model_for_Roman( date4traps )  
            ccd = CCD(well_fill_power=0.58, full_well_depth=full_well_image)
            roe = ROE()
            emccd.update_cti( ccd=ccd, roe=roe, traps=traps, express=1 )
        
        self.emccd = emccd
        self.emccd_dict = {'pix_area':pixel_pitch**2,
                           'em_gain':em_gain}
        
    def load_batches_cubes(self):
        # =============================================================================
        # load_batches_cubes
        # 
        # Load pre-computed image cubes for scene
        # =============================================================================
        datadir = 'output/SpeckleSeries/'+self.scene["name"]+'/'
        for II,batch in enumerate(self.scene['schedule']['batches']):
            data = pyfits.open(datadir+"Ii_cube_batch{}.fits".format(batch['batch_ID']))
            im_cube = data[0].data
            self.scene['schedule']['batches'][II]['im_cube'] = im_cube
            
    def add_detector_noise_to_batches(self,label_out=''):
        from scipy.interpolate import interp1d
        for II,batch in enumerate(self.scene['schedule']['batches']):
            im_cube = self.scene['schedule']['batches'][II]['im_cube']
            num_im = len(im_cube)
            fn = interp1d(np.linspace(0,1,num_im),im_cube,axis=0)
            num_frames_interp = batch['num_frames_interp']
            exptime = batch['exptime']
            # print(exptime)
            im_cube_interp = fn(np.linspace(0,1,num_frames_interp))
            im_cube_interp_emccd = []
            for JJ in range(num_frames_interp):
                im_cube_interp_emccd.append(self.add_detector_noise(im_cube_interp[JJ],exptime))
            hdulist = pyfits.PrimaryHDU(im_cube_interp_emccd)
            hdulist.writeto(self.scene['outdir']+'Ii_cube_emccd_batch{}{}.fits'.format(batch['batch_ID'],label_out),overwrite=True)
            im_coadded_interp_emccd = np.mean(im_cube_interp_emccd,axis=0)
            hdulist = pyfits.PrimaryHDU(im_coadded_interp_emccd)
            hdulist.writeto(self.scene['outdir']+'Ii_coadded_emccd_batch{}{}.fits'.format(batch['batch_ID'],label_out),overwrite=True)
            # hdulist = pyfits.PrimaryHDU(im_cube_interp)
            # hdulist.writeto(self.scene['outdir']+'test{}.fits'.format(batch['batch_ID']),overwrite=True)

    def compute_contrast_curve(self,ni_im,iwa=3,owa=9,d_sep=0.5):
        # =============================================================================
        # compute_contrast_curve
        # 
        # return contrast at an array of separations
        # iwa owa d_sep in arcsec
        # =============================================================================

        sampling_hlc = {'hlc_band1':{'1':0.435}} #TODO

        sz_im = np.shape(ni_im)
        sep_arr = np.arange(iwa,owa+0.5,d_sep)
        num_samp = len(sep_arr)
        ni_curve = np.zeros(num_samp)
        for II in range(num_samp):
            r_ring = sep_arr[II]/(sampling_hlc['hlc_band1']['1'])

            # Mask
            rin_mask = make_circ_mask(sz_im[0],0,0,r_ring-0.5/sampling_hlc['hlc_band1']['1'])
            rout_mask = make_circ_mask(sz_im[0],0,0,r_ring+0.5/sampling_hlc['hlc_band1']['1'])
            mask_ring = rout_mask-rin_mask
            
            ni_avg_ring = np.mean(ni_im[np.where(mask_ring==1)])
            ni_curve[II] = ni_avg_ring
        return sep_arr,ni_curve
    
    def compute_avarage_contrast(self,ni_im,iwa=3,owa=9):
        # =============================================================================
        # compute_contrast_curve
        # 
        # return contrast at an array of separations
        # iwa owa d_sep in arcsec
        # =============================================================================

        sampling_hlc = {'hlc_band1':{'1':0.435}} #TODO
        sz_im = np.shape(ni_im)
        rout_mask = make_circ_mask(sz_im[0],0,0,owa/sampling_hlc['hlc_band1']['1'])
        rin_mask = make_circ_mask(sz_im[0],0,0,iwa/sampling_hlc['hlc_band1']['1'])
        mask_ring = rout_mask-rin_mask
        ni_avg_ring = np.mean(ni_im[np.where(mask_ring==1)])
        
        return ni_avg_ring