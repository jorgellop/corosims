"""Optical Propagations."""

import roman_preflight_proper
import proper
import cgisim
import numpy as np
from scipy.ndimage import convolve
from scipy.interpolate import RegularGridInterpolator
import astropy.io.fits as pyfits
import matplotlib.pylab as plt
import os
import warnings
import importlib

from emccd_detect.emccd_detect import EMCCDDetectBase

from utils import make_circ_mask#,crop_data,degenPA

class corosims_core():
    """
    Core simulator of Roman Coronagraph images. A wrapper over cgisim, by John Krist.


    Parameters
    ----------
    cor_type : string
        coronagraph type: HLC, SPC
    bandpass : string
        string defining the Roman Coronagraph bandpass

    Examples
    --------
    """
    
    def __init__(self, cor_type = 'hlc_band1', bandpass='1'):
        # Define the 4 main options of cgisim as class attributes
        self.cor_type = cor_type
        self.bandpass = bandpass # 
        self.cgi_mode = 'excam_efield' # Always work in EF mode
        self.polaxis = -10 # Work on mean polarization mode, but no polarization is also acceptable.
        
        self.sz_im = 201 # This is replaced everytime an EF is computed
        
        # Define all other options in the options dictionary
        self.options = {}
        
        localpath = os.path.dirname(os.path.abspath(__file__))
        head, tail = os.path.split(localpath)
        self.paths = {"datadir_jitter": os.path.join(head, 'data', 'jitter_EFs_and_data'),
                      "dm_maps": os.path.join(head, 'data', 'dm_maps')}

        # Predefine options
        # DMs taken from the roman_phasec library or computed by FALCO
        if cor_type=='hlc_band1' and bandpass=='1':
            dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_2e-9_dm1_v.fits' )
            dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_2e-9_dm2_v.fits' )
        elif cor_type=='spc-wide' and bandpass=='4':
            dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/spc-wide_ni_3e-9_dm1_v.fits' )
            dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/spc-wide_ni_3e-9_dm2_v.fits' )
        elif cor_type=='hlc_band4' and bandpass=='4':
            dm1 = proper.prop_fits_read( os.path.join(self.paths["dm_maps"],'HLC4_Band4_falco_S8T3_NI6.5e-09_dm1.fits' ))
            dm2 = proper.prop_fits_read( os.path.join(self.paths["dm_maps"],'HLC4_Band4_falco_S8T3_NI6.5e-09_dm2.fits' ))
        elif cor_type=='spc-wide_band1' and bandpass=='1':
            dm1 = proper.prop_fits_read( os.path.join(self.paths["dm_maps"],'SPLC_Band1_falco_S7T1_NI5.0e-09_dm1.fits' ))
            dm2 = proper.prop_fits_read( os.path.join(self.paths["dm_maps"],'SPLC_Band1_falco_S7T1_NI5.0e-09_dm2.fits' ))
        self.options['dm1'] = dm1
        self.options['dm2'] = dm2
        self.options['dm1_xc_act'] = 23.5
        self.options['dm2_xc_act'] = 23.5 - 0.1
        self.options['dm1_yc_act'] = 23.5
        self.options['dm2_yc_act'] = 23.5
        
        # Some other predefined parameters
        self.options['no_integrate_pixels'] = False
        self.options['pixel_scale'] = 0.0218
        self.options['sampling'] = {'1':500/573.8*0.5,
                                    '2':500/659.4*0.5,
                                    '3':500/729.3*0.5,
                                    '4':500/825.5*0.5}
        
        # Define a default source
        self.define_source('a0v',2.0)
        

    def define_source(self,star_type,vmag):
        """
        Define source of which we'll take images of.
    
        A corosims object will have a source defined; default is set in the __init__.     

        Parameters
        ----------
        star_type : Float
            Stellar type
        vmag : Float
            V-Magnitude of the source
    
        """
        self.source = {'star_type': star_type,
                            'vmag':vmag}

    def compute_EF(self,x_offset_mas=0,y_offset_mas=0,use_fpm=1,zindex=None,zval_m=None,
                   dm1_shear_x=0,dm2_shear_x=0,dm1_shear_y=0,dm2_shear_y=0,
                   lyot_shift_x=0,lyot_shift_y=0,
                   cgi_shift_x=0,cgi_shift_y=0,
                   passvalue_proper={}):
        """
        Compute the electric field.
    
        This is where all images are computed.     

        Parameters
        ----------
        use_fpm : float
            0 or 1, 0: don't use focal plane mask, 1: use focal plane mask
        zindex : ModelVariables
            Structure containing temporary optical model variables
        isNorm : bool
            If False, return an unnormalized image. If True, return a
            normalized image with the currently stored norm value.
    
        Returns
        -------
        EF_new : numpy ndarray
            2D electric field in image plane
        """
        dm1 = self.options['dm1'] 
        dm2 = self.options['dm2'] 
        dm1_xc_act = self.options['dm1_xc_act'] + dm1_shear_x/0.9906e-3
        dm2_xc_act = self.options['dm2_xc_act'] + dm2_shear_x/0.9906e-3
        dm1_yc_act = self.options['dm1_yc_act'] + dm1_shear_y/0.9906e-3
        dm2_yc_act = self.options['dm2_yc_act'] + dm2_shear_y/0.9906e-3
        params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 'use_fpm':use_fpm, 
                   'dm1_xc_act': dm1_xc_act,'dm2_xc_act': dm2_xc_act,'dm1_yc_act': dm1_yc_act,'dm2_yc_act': dm2_yc_act,
                  'lyot_x_shift_m':lyot_shift_x,'lyot_y_shift_m':lyot_shift_y,
                  'cgi_x_shift_m':cgi_shift_x,'cgi_y_shift_m':cgi_shift_y}
        if 'source_x_offset' not in passvalue_proper:
            params['source_x_offset_mas'] = x_offset_mas
            params['source_y_offset_mas'] = y_offset_mas
            # import pdb 
            # pdb.set_trace()

        if zindex is not None:
            params['zindex'] = zindex
            params['zval_m'] = zval_m
        
        # Proper passvalue is treated internally in cgisim
        params.update(passvalue_proper)
        EF, counts = cgisim.rcgisim( self.cgi_mode, self.cor_type, self.bandpass, self.polaxis, params,
                                    no_integrate_pixels=self.options['no_integrate_pixels']) 
        
        # Source's spectrum
        if ~('counts_spectrum' in self.source):
            self.compute_spectrum()
        counts_spectrum = self.source['counts_spectrum']
        EF_new = []
        for II,counts in enumerate(counts_spectrum):
            EF_new.append(EF[II] * np.sqrt(counts))
        
        self.sz_im = np.shape(EF)[1]
        return EF_new

    def compute_spectrum(self):
        """
        Compute the spectrum over the current bandpass.
    
        Called by compute_EF if self.source['counts_spectrum'] is not initialized    
    
        Returns
        -------
        normI : float
            normalization factor equal to the peak off-axis PSF
        """
        
        star_type = self.source["star_type"]
        star_vmag = self.source["vmag"]

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
            if star_type is not None:
                counts = polarizer_transmission * cgisim.cgisim_get_counts( star_type, self.bandpass, bandpass_i, nd, star_vmag, 'V', 'excam', info_dir )
            else:
                counts = 1
            # total_counts += counts
            counts_spectrum.append(counts)
        self.source['counts_spectrum'] = counts_spectrum
        return counts_spectrum

    def compute_offaxis_normalization(self):
        """
        Compute the normalization factor.
    
        Used to normalize images to units of contrast.     
    
        Returns
        -------
        normI : float
            normalization factor equal to the peak off-axis PSF
        """
        print("Computing normalization factor")
        EF0 = self.compute_EF(use_fpm=0)
                    
        I0 = np.abs(EF0)**2 
        I0 = np.sum(I0,axis=0)
        normI = np.max(I0)
        self.maxI0_offaxis = normI
        
        return normI

    def generate_image(self,x_offset_mas=0,y_offset_mas=0,use_fpm=1,jitter_sig_x=0,jitter_sig_y=0,
                       zindex=None,zval_m=None,
                       dm1_shear_x=0,dm2_shear_x=0,dm1_shear_y=0,dm2_shear_y=0,
                       lyot_shift_x=0,lyot_shift_y=0,
                       cgi_shift_x=0,cgi_shift_y=0,
                       stellar_diameter=None,
                       passvalue_proper={},
                       flag_return_contrast=False,use_emccd=False,exptime=1.0,
                       drift_vector=None):
        """
        Generate image.
    
        Used to generate images in units of contrast.     

        Parameters
        ----------
        use_fpm : int
            1: use focal plane mask, 0: don't use focalplane mask
        use_fpm : int
            1: use focal plane mask, 0: don't use focalplane mask
    
        Returns
        -------
        Isum : numpy ndarray
            2 D image of the intensity of the simulated source
        """
        EF = self.compute_EF(x_offset_mas=x_offset_mas,y_offset_mas=y_offset_mas,use_fpm=use_fpm,zindex=zindex,zval_m=zval_m,
                             dm1_shear_x=dm1_shear_x,dm2_shear_x=dm2_shear_x,dm1_shear_y=dm1_shear_y,dm2_shear_y=dm2_shear_y,
                             lyot_shift_x=lyot_shift_x,lyot_shift_y=lyot_shift_y,
                             cgi_shift_x=cgi_shift_x,cgi_shift_y=cgi_shift_y,
                             passvalue_proper=passvalue_proper)
        
        # Get normalization factor for contrast
        if flag_return_contrast:
            if not hasattr(self, 'maxI0_offaxis'):
                self.compute_offaxis_normalization()
            normalization = self.maxI0_offaxis
        else:
            normalization = 1
        sz_im = self.sz_im
        
        # Add jitter if necessary
        if (jitter_sig_x!=0 or jitter_sig_y!=0 or stellar_diameter is not None) and (np.sqrt(x_offset_mas**2+y_offset_mas**2)<10):
            # If one (only one) of the jitter axis is zero, add just a little bit to avoid numerical problems
            if jitter_sig_x!=0 or jitter_sig_y!=0:
                if jitter_sig_x==0:
                    jitter_sig_x = jitter_sig_y*1e-2 # Just a little bit to avoid dividing by zero
                if jitter_sig_y==0:
                    jitter_sig_y = jitter_sig_x*1e-2 # Just a little bit to avoid dividing by zero

            # Read in the jitter dictonary if not yet done.
            if not ('jitter_dict' in self.options):
                try:
                    self.read_in_jitter_deltaEFs()
                except:
                    warnings.warn("Did you read in the jitter dictionary from the right folder?")
            
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
            W_jit = np.exp(-0.5*((X-x_offset_mas)**2/jitter_sig_y**2 + (Y-y_offset_mas)**2/jitter_sig_x**2))

            # Stellar diamter: top hat convolution
            if stellar_diameter is not None:    
                rad = stellar_diameter/2
                top_hat = make_circ_mask(npix,0,0,rad/pix_scale)
                W_jit = convolve(top_hat,W_jit)
    
            if drift_vector is not None:
                # Import the tool to draw a line:
                line_aa = importlib.import_module('skimage.draw').line_aa
                im_line_segment = np.zeros((npix, npix))
                start_vect_x = int(npix/2+x_offset_mas/pix_scale)
                start_vect_y = int(npix/2+y_offset_mas/pix_scale)
                fin_vect_x = int(npix/2+x_offset_mas/pix_scale + drift_vector[0]/pix_scale)
                fin_vect_y = int(npix/2+y_offset_mas/pix_scale + drift_vector[1]/pix_scale)
                rr, cc, val = line_aa(start_vect_x, start_vect_y, fin_vect_x, fin_vect_y)
                im_line_segment[rr, cc] = val 
                W_jit = convolve(im_line_segment,W_jit)

            interp = RegularGridInterpolator((X[0,:], Y[:,0]), W_jit,
                                          bounds_error=False, fill_value=None)

            
            for II,(x,y,A) in enumerate(zip(x_jitt_offset_mas_arr,y_jitt_offset_mas_arr,A_arr)):
                WA_jit[II] = interp((x,y)) * A
            
            # Normalize W*A function
            WA_jit_norm = np.sum(WA_jit)

            # Add jitter
            if WA_jit_norm!=0:
                if 'counts_spectrum' in self.source:
                    counts_spectrum = self.source['counts_spectrum']
                else:
                    counts_spectrum = np.ones(len(EF))

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
            else: # if jitter is very small
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
                warnings.warn("Ignoring the fact that you requested contrast, will return emccd image")
                Isum = Isum*normalization
            Isum = self.add_detector_noise(Isum,exptime)

        return Isum

    def read_in_jitter_deltaEFs(self,datadir_jitt0=None):
        """
        Read teh jitter products needed for simulating jitter.
    
        Parameters
        ----------
    
        """
        print("Reading in the jitter parameters and EFs into a cube")
        
        if datadir_jitt0 is None:
            datadir_jitt0 = self.paths["datadir_jitter"]
        datadir_jitt = os.path.join(datadir_jitt0, 'cor_type_'+self.cor_type+'_band'+self.bandpass+'_polaxis'+str(self.polaxis), 'fields')
        
        # import pdb 
        # pdb.set_trace()

        # Read in files 
        inFile = pyfits.open(os.path.join(datadir_jitt,'offsets_mas.fits'))
        offsets_mat = inFile[0].data
        x_jitt_offset_mas_arr = offsets_mat[0,:]
        y_jitt_offset_mas_arr = offsets_mat[1,:]
        num_jitt = len(y_jitt_offset_mas_arr)
        
        # Read the A (areas) list
        inFile = pyfits.open(os.path.join(datadir_jitt,'A_areas.fits'))
        A_arr = inFile[0].data

        # Read in EF0
        inFile = pyfits.open(os.path.join(datadir_jitt,'EF0_real.fits'))
        EF0_real = inFile[0].data
        inFile = pyfits.open(os.path.join(datadir_jitt,'EF0_imag.fits'))
        EF0_imag = inFile[0].data
        EF0 = EF0_real * 1j*EF0_imag

        # Generate cube by reading in all jitters
        dEF_mat = []
        for II in range(num_jitt):
            # Read in EF0
            inFile = pyfits.open(os.path.join(datadir_jitt,'EF{}_real.fits'.format(II+1)))
            EFII_real = inFile[0].data
            inFile = pyfits.open(os.path.join(datadir_jitt,'EF{}_imag.fits'.format(II+1)))
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

    def compute_jitter_EFs(self,d_ang1=0.15,r1_lim=0.6,outdir0=None):
        """
        Compute the jitter products needed to efficiently simulate jitter.
    
        This needs to be done once per mode.     

        Parameters
        ----------
        d_ang1 : float
            delta_r in mas in section 1
        r1_lim : float
            outer radious in mas for section 1
    
        Returns
        -------
        Isum : numpy ndarray
            2 D image of the intensity of the simulated source
        """
        print('Computing all EFs for jitter a la Krist et al. 2023, this will take a lot of time!')

        if outdir0 is None:
            outdir0 = self.paths["datadir_jitter"]
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
        outdir = os.path.join(outdir0,'cor_type_'+cor_type+'_band'+bandpass+'_polaxis'+str(polaxis))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            os.makedirs(os.path.join(outdir,'fields'))

        # Save A array
        A_arr = A_arr/np.sum(A_arr)
        hdulist = pyfits.PrimaryHDU(A_arr)
        hdulist.writeto(os.path.join(outdir,'fields','A_areas.fits'),overwrite=True)
        
        # Plot cloud like in Krist+2023
        fntsz = 18
        
        plt.figure(figsize=(6*1.5,6*1.5))
        plt.scatter(x_arr,y_arr,marker="+",s=15)
        plt.xlabel('X [mas]',fontsize=fntsz);plt.xticks(fontsize=fntsz)
        plt.ylabel('Y [mas]',fontsize=fntsz);plt.yticks(fontsize=fntsz)
        plt.grid()
        plt.title('Jitter Points {}'.format(len(x_arr)),fontsize=fntsz)
        plt.savefig(os.path.join(outdir,'jittercloud.png'), dpi=500)

        #% Generate cube of delta-EFs
        # Generate EF0, centered, no errors
                
        dm1 = self.options['dm1'] 
        dm2 = self.options['dm2'] 

        
        
        print( "Computing on-axis PSF - No T/T error" )
        params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
        EF0, counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params,
                                    output_file = os.path.join(outdir,'fields','EF0')) 
        
        # Loop for all cloud positions and populate cube
        source_x_offset_mas_arr = x_arr
        source_y_offset_mas_arr = y_arr
        for II,(source_x_offset_mas,source_y_offset_mas) in enumerate(zip(source_x_offset_mas_arr,source_y_offset_mas_arr)):
            print('Propagating jitter realization {0}/{1}.'.format(II+1,len(x_arr)))
            params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 
                      'source_x_offset_mas':source_x_offset_mas, 'source_y_offset_mas':source_y_offset_mas} 
            EF, counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params,
                                        output_file = os.path.join(outdir,'fields','EF{}'.format(II+1)))
        hdulist = pyfits.PrimaryHDU(np.array([source_x_offset_mas_arr,source_y_offset_mas_arr]))
        hdulist.writeto(os.path.join(outdir,'fields','offsets_mas.fits'),overwrite=True)
        
        print('Done computing all EFs for jitter')

    def define_emccd(self,em_gain=1000.0, full_well_image=60000.0, full_well_serial=100000.0,
                     dark_rate=0.00056, cic_noise=0.01, read_noise=100.0, bias=0,qe=1.0, cr_rate=0, 
                     pixel_pitch=13e-6, e_per_dn=1.0, numel_gain_register=604, nbits=14,
                     use_traps = False,date4traps=2028.0):
        """
        Create emccd object.
    
        Used to return images in units of contrast.     

        Parameters
        ----------
        use_fpm : int
            1: use focal plane mask, 0: don't use focalplane mask
        use_fpm : int
            1: use focal plane mask, 0: don't use focalplane mask
    
        """
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
        
    def add_detector_noise(self,Im, exptime):
        """
        Generate image.
    
        Used to return images in units of contrast.     

        Parameters
        ----------
        use_fpm : int
            1: use focal plane mask, 0: don't use focalplane mask
        use_fpm : int
            1: use focal plane mask, 0: don't use focalplane mask
    
        Returns
        -------
        Im_noisy : numpy ndarray
            2 D image of the intensity of the simulated source with detector noise
        """
        Im_noisy = self.emccd.sim_sub_frame(Im, exptime).astype(float)
        
        return Im_noisy
    
    def compute_contrast_curve(self,ni_im,iwa=3,owa=9,d_sep=0.5):
        """
        Compute contrast curve.
    
        Used to return images in units of contrast.     

        Parameters
        ----------
        use_fpm : int
            1: use focal plane mask, 0: don't use focalplane mask
        use_fpm : int
            1: use focal plane mask, 0: don't use focalplane mask
    
        Returns
        -------
        Im_noisy : numpy ndarray
            2 D image of the intensity of the simulated source with detector noise
        """

        sampling = self.options['sampling'][self.bandpass]

        sz_im = np.shape(ni_im)
        sep_arr = np.arange(iwa,owa+0.5,d_sep)
        num_samp = len(sep_arr)
        ni_curve = np.zeros(num_samp)
        for II in range(num_samp):
            r_ring = sep_arr[II]/(sampling)

            # Mask
            rin_mask = make_circ_mask(sz_im[0],0,0,r_ring-0.5/sampling)
            rout_mask = make_circ_mask(sz_im[0],0,0,r_ring+0.5/sampling)
            mask_ring = rout_mask-rin_mask
            
            ni_avg_ring = np.mean(ni_im[np.where(mask_ring==1)])
            ni_curve[II] = ni_avg_ring
        return sep_arr,ni_curve
