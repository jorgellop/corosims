"""Observations with the Roman Coronagraph."""

import numpy as np
from datetime import date
import os
import matplotlib.pylab as plt
import astropy.io.fits as pyfits
import warnings

from corosims.corosims_core import corosims_core
from utils import make_circ_mask,degenPA,crop_data

class Observation():
    """
    Observation class to define a simulation.


    Parameters
    ----------
    cor_type : string
        coronagraph type: HLC, SPC
    bandpass : string
        string defining the Roman Coronagraph bandpass

    Examples
    --------
    """
    
    def __init__(self,name=None, cor_type = 'hlc_band1', bandpass='1'):
        
        # An Observation has a set of scenes, here initialized as zero
        self.num_scenes = 0
        self.scenes = []
        
        # An Observation has a set of observation batches, here initialized as zero
        self.num_batches = 0
        self.batches = []

        # An Observation has a set of sources, here initialized as zero
        self.sources = []
        self.num_sources = 0
        
        # Define the instrument with a "core" instance
        self.corosims = corosims_core(cor_type=cor_type, bandpass=bandpass)
        
        # If name is not given, just call it OS_{date}
        if name is None:
            today = date.today()
            date_str = today.strftime("%Y%m%d")
            name='OS_'+date_str
        self.name_OS = name
        # scene['name'] = name
        
        # Paths
        localpath = os.path.dirname(os.path.abspath(__file__))
        head, tail = os.path.split(localpath)

        self.paths = {'outdir': os.path.join('output','SpeckleSeries',name),
                      'datadir': os.path.join(head, 'data')}
        # if not os.path.exists(self.paths['outdir']):
        #     os.makedirs(self.paths['outdir'])

    def create_source(self,name=None,vmag=None,star_type='a0v',spectrum=None,stellar_diam=None):
    
        """
        Create a point source and add it to the observation.
        self.sources will carry a list of dictionaries with all the information about the user-defined sources.     

        Parameters
        ----------
        name : str, optional
            Name of the source.
        vmag : float, optional
            V-band magnitude of the source.
        star_type : str, optional
            Spectral type of the star (default: 'a0v').
        spectrum : np.ndarray, optional
            Spectrum of the source.
        stellar_diam : float, optional
            Stellar diameter in milliarcseconds.
        """
        if name is None:
            name = 'source{}'.format(self.num_sources)

        source = {'source_index_id':self.num_sources,
                'name':name,
                'star_type': star_type, # TODO: planet spectrum
                'vmag':vmag,
                'spectrum':spectrum,
                'stellar_diam':stellar_diam}
        
        # append source dictionary to the sources list and update number of sources
        self.sources.append(source)
        self.num_sources = self.num_sources + 1
        
        
    def create_scene(self,name=None):
        """
        Generate astrophysical scene.
    
        self.scenes will carry a list of dictionaries with all the information about the user-defined scenes.     

        Parameters
        ----------
        name : str, optional
            Name of the scene.
    
        """
                    
        if name is None:
            name = 'Scene{}'.format(self.num_scenes)
            
        scene = { 
            'scene_index_id' : self.num_scenes,
            "name":name,
            "sources_in_scene":[],
            "num_sources_in_scene":0}
        
        
        self.scenes.append(scene)
        self.num_scenes = self.num_scenes + 1

        
    def convolve_image_with_prf(self,image_scene, image_pixel_scale,thresh=1e-20):
        """
        Convolve an image with a grid of PSFs while handling different samplings.
        
        Parameters
        ----------
        image_scene : np.ndarray
            Input scene image.
        image_pixel_scale : float
            Pixel scale of the input image in mas/pixel.
        thresh : float, optional
            Threshold for image values to be considered in convolution (default: 1e-20).
        
        Returns
        -------
        np.ndarray
            Convolved image.
        """
        from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
        
        n=201
        
        # Inner region
        dx_list = [22.2,5.3,1.1]
        owa_list = [500,160,35]
        label_out = '_maxSep{}mas_res{:.1f}mas'
        
        prf_interpolator_list = []
        for dx,owa in zip(dx_list,owa_list):
            # Load FITS data
            flnm = os.path.join(self.paths['datadir'],'psf_grids_for_convolution','im_cube_PRF'+label_out.format(owa,dx)+'.fits')
            data = pyfits.open(flnm)
            psf_cube = data[0].data
            flnm = os.path.join(self.paths['datadir'],'psf_grids_for_convolution','sep_x_PRF'+label_out.format(owa,dx)+'.fits')
            data = pyfits.open(flnm)
            sep_offset_x = data[0].data/1000
            flnm = os.path.join(self.paths['datadir'],'psf_grids_for_convolution','sep_y_PRF'+label_out.format(owa,dx)+'.fits')
            data = pyfits.open(flnm)
            sep_offset_y = data[0].data/1000
            psf_pixel_scale = 0.0218
            # Flatten the PSF grid positions
            psf_positions = np.c_[sep_offset_x.ravel(), sep_offset_y.ravel()]
            
    
            # Create an interpolator for PSFs in the cube
            # prf_interpolator = LinearNDInterpolator(psf_positions, psf_cube)
            prf_interpolator = RegularGridInterpolator((range(n),range(n),sep_offset_x.ravel(), sep_offset_y.ravel()), psf_cube)
            prf_interpolator_list.append(prf_interpolator)



        # import pdb 
        # pdb.set_trace()


        # # Define original image grid in physical units
        im_scene_extent = (image_scene.shape[0] // 2) * image_pixel_scale
        x_im_scene = np.linspace(-im_scene_extent, im_scene_extent, image_scene.shape[0])
        y_im_scene = np.linspace(-im_scene_extent, im_scene_extent, image_scene.shape[0])
        # interp_func = RegularGridInterpolator((x_im_scene, y_im_scene), image_scene, bounds_error=False, fill_value=0)

        # # PSF interp
        # target_extent = (psf_cube.shape[1] // 2) * psf_pixel_scale
        # x_target = np.linspace(-target_extent, target_extent, psf_cube.shape[1])
        # y_target = np.linspace(-target_extent, target_extent, psf_cube.shape[1])
        # x_target_grid, y_target_grid = np.meshgrid(x_target, y_target)

        # points = np.c_[y_target_grid.ravel(), x_target_grid.ravel()]
        # resampled_image = interp_func(points).reshape(psf_cube.shape[1], psf_cube.shape[1])

        grid_for_interp=np.meshgrid(range(n),range(n))
        flattened_grid = np.vstack([grid_for_interp[0].flatten(),grid_for_interp[1].flatten()])

        convolved_image = np.zeros((n,n))
        for II in range(len(x_im_scene)):
            for JJ in range(len(y_im_scene)):
                sep = np.sqrt(x_im_scene[II]**2+y_im_scene[JJ]**2)
                if sep<0.5 and image_scene[II,JJ]>thresh:
                    if sep<owa_list[2]/1000:
                        prf_interpolator = prf_interpolator_list[2]
                    elif sep<owa_list[1]/1000:
                        prf_interpolator = prf_interpolator_list[1]
                    elif sep<owa_list[0]/1000:
                        prf_interpolator = prf_interpolator_list[0]
                    # Interpolate the PSF at this location
                    points_interp = np.vstack([ flattened_grid , x_im_scene[II]*np.ones(len(grid_for_interp[0].flatten())), y_im_scene[JJ]*np.ones(len(grid_for_interp[0].flatten())) ]).T
                    psf = prf_interpolator(points_interp).reshape(n,n)

                    # Ensure the PSF is valid
                    if psf is not None and np.sum(psf) > 0:
                        # Convolve the local patch with the interpolated PSF
                        # convolved_image[II,JJ] = np.sum(resampled_image * psf[:resampled_image.shape[0], :resampled_image.shape[1]])
                        convolved_image = convolved_image + image_scene[II,JJ] * psf#[:resampled_image.shape[0], :resampled_image.shape[1]]
                        # if resampled_image[JJ,II]>3.9e-5:
                        #     import pdb 
                        #     pdb.set_trace()

    
        # Normalize to conserve energy
        # convolved_image *= resampled_image.sum() / convolved_image.sum()

        return convolved_image

    def add_point_source_to_scene(self,source_index_id=None,scene_index_id=None,
                                  source_name=None,scene_name=None,
                                  xoffset_mas=0,yoffset_mas=0):
        """
        Add a point source to a previously generated scene.
        
        Parameters
        ----------
        source_index_id : int, optional
            Index of the source to be added.
        scene_index_id : int, optional
            Index of the scene to which the source is added.
        source_name : str, optional
            Name of the source.
        scene_name : str, optional
            Name of the scene.
        xoffset_mas : float, optional
            X-offset of the source in milliarcseconds (default: 0).
        yoffset_mas : float, optional
            Y-offset of the source in milliarcseconds (default: 0).
        """
        if self.num_scenes==0:
            warnings.warn("No scene has been defined")

        # Checking that IDs are valid for scene
        if source_index_id is not None and source_name is not None:
            if self.sources[source_index_id]['name'] != source_name:
                warnings.warn("Source: Name and Index ID don't match, going for name")
        if source_name is not None:
            source = [s for s in self.sources if s['name']==source_name]
            if len(source)!=1:
                warnings.warn("Source Name not valid")
            source = source[0]
            source_index_id = source["source_index_id"]

        # Checking that IDs are valid for scene
        if scene_index_id is not None and scene_name is not None:
            if self.scenes[scene_index_id]['name'] != scene_name:
                warnings.warn("Scene: Name and Index ID don't match, going for name")
        if scene_index_id is None: # Pick scene 0 if no scene is defined
            scene_index_id = 0
        if scene_name is not None:
            for index, sc in enumerate(self.scenes):
                if sc.get("name") == scene_name:
                    scene_index_id = index
        
        # Dictionary to store this source in scene information
        source_in_scene ={}
        # Is the source on-axis?
        sep = np.sqrt(xoffset_mas**2+yoffset_mas**2)
        max_sep_onaxis = 100 #TODO: where do I put this kind of variables
        if sep<max_sep_onaxis:
            flag_onaxis=True
        else:
            flag_onaxis=False
        
        source_in_scene["source_index_id"] = source_index_id
        source_in_scene["flag_onaxis"] = flag_onaxis
        source_in_scene["x_y_separation_mas"] = np.array([xoffset_mas,yoffset_mas])
        
        # Append source to scene             
        self.scenes[scene_index_id]["sources_in_scene"].append(source_in_scene)
        self.scenes[scene_index_id]["num_sources_in_scene"] = self.scenes[scene_index_id]["num_sources_in_scene"] + 1

    def create_batch(self,batch_id=None,scene_index_id=None,scene_name=None,jitter_x=[0],jitter_y=[0],num_timesteps=None,zindex=None,zval_m=None,
                       dm1_shear_x=None,dm2_shear_x=None,dm1_shear_y=None,dm2_shear_y=None,
                       lyot_shift_x=None,lyot_shift_y=None,
                       cgi_shift_x=None,cgi_shift_y=None,
                       passvalue_proper=None,exptime=None,
                       V3PA=0):
        """
        Create an observation batch.
        
        Parameters
        ----------
        batch_id : int, optional
            Identifier for the batch.
        scene_index_id : int, optional
            Index of the scene associated with this batch.
        scene_name : str, optional
            Name of the scene associated with this batch.
        jitter_x : list, optional
            List of X-jitter values.
        jitter_y : list, optional
            List of Y-jitter values.
        num_timesteps : int, optional
            Number of timesteps in the batch.
        V3PA : float, optional
            Position angle of the telescope (default: 0).
        """
        # Checking that IDs are valid for scene
        if scene_index_id is not None and scene_name is not None:
            if self.scenes[scene_index_id]['name'] != scene_name:
                warnings.warn("Scene: Name and Index ID don't match, going for name")
        if scene_index_id is None: # Pick scene 0 if no scene is defined
            scene_index_id = 0
        if scene_name is not None:
            for index, sc in enumerate(self.scenes):
                if sc.get("name") == scene_name:
                    scene_index_id = index

        if batch_id is None:
            batch_id=self.num_batches
        batch = {'batch_id':batch_id,
                 'batch_scene_id':scene_index_id}
        
        # Bunber of timesteps: the number of times we'll propagate through the instrument
        if num_timesteps is None:
            num_timesteps = len(jitter_x)
        else:
            # Check that numbers add up
            if num_timesteps!=len(jitter_x) or num_timesteps!=len(jitter_y):
                warnings.warn("num_timesteps and length of jitter_x")
            if len(jitter_x)==1:
                jitter_x = np.ones(num_timesteps)*jitter_x
            if len(jitter_y)==1:
                jitter_y = np.ones(num_timesteps)*jitter_y
        
        # Define jitter
        batch['jitter_x'] = jitter_x
        batch['jitter_y'] = jitter_y
        batch['num_timesteps'] = num_timesteps
        batch['V3PA'] = V3PA
        
        if zindex is not None:
            # Check lengths
            sz_zs = np.shape(zval_m)
            num_zs = sz_zs[0]
            if num_zs!=num_timesteps:
                warnings.warn('Number of jitters and other errors should match!')
                return
            batch['zindex'] = zindex
            batch['zval_m'] = zval_m
        else:
            batch['zindex'] = None
            batch['zval_m'] = [None]*num_timesteps

        if dm1_shear_x is not None:
            sz_zs = np.shape(dm1_shear_x)
            num_zs = sz_zs[0]
            if num_zs!=num_timesteps:
                warnings.warn('Number of jitters and shears should match!')
                return
            batch['dm1_shear_x'] = dm1_shear_x
        else:
            batch['dm1_shear_x'] = np.ones(num_timesteps)*0
        if dm2_shear_x is not None:
            sz_zs = np.shape(dm2_shear_x)
            num_zs = sz_zs[0]
            if num_zs!=num_timesteps:
                warnings.warn('Number of jitters and shears should match!')
                return
            batch['dm2_shear_x'] = dm2_shear_x
        else:
            batch['dm2_shear_x'] = np.ones(num_timesteps)*0

        if dm1_shear_y is not None:
            sz_zs = np.shape(dm1_shear_y)
            num_zs = sz_zs[0]
            if num_zs!=num_timesteps:
                warnings.warn('Number of jitters and shears should match!')
                return
            batch['dm1_shear_y'] = dm1_shear_y
        else:
            batch['dm1_shear_y'] = np.ones(num_timesteps)*0
        if dm2_shear_y is not None:
            sz_zs = np.shape(dm2_shear_y)
            num_zs = sz_zs[0]
            if num_zs!=num_timesteps:
                warnings.warn('Number of jitters and shears should match!')
                return
            batch['dm2_shear_y'] = dm2_shear_y
        else:
            batch['dm2_shear_y'] = np.ones(num_timesteps)*0

        if lyot_shift_x is not None:
            sz_zs = np.shape(lyot_shift_x)
            num_zs = sz_zs[0]
            if num_zs!=num_timesteps:
                warnings.warn('Number of jitters and shears should match!')
                return
            batch['lyot_shift_x'] = lyot_shift_x
        else:
            batch['lyot_shift_x'] = np.ones(num_timesteps)*0
        if lyot_shift_y is not None:
            sz_zs = np.shape(lyot_shift_y)
            num_zs = sz_zs[0]
            if num_zs!=num_timesteps:
                warnings.warn('Number of jitters and shears should match!')
                return
            batch['lyot_shift_y'] = lyot_shift_y
        else:
            batch['lyot_shift_y'] = np.ones(num_timesteps)*0

        if cgi_shift_x is not None:
            sz_zs = np.shape(cgi_shift_x)
            num_zs = sz_zs[0]
            if num_zs!=num_timesteps:
                warnings.warn('Number of jitters and shears should match!')
                return
            batch['cgi_shift_x'] = cgi_shift_x
        else:
            batch['cgi_shift_x'] = np.ones(num_timesteps)*0
        if cgi_shift_y is not None:
            sz_zs = np.shape(cgi_shift_y)
            num_zs = sz_zs[0]
            if num_zs!=num_timesteps:
                warnings.warn('Number of jitters and shears should match!')
                return
            batch['cgi_shift_y'] = cgi_shift_y
        else:
            batch['cgi_shift_y'] = np.ones(num_timesteps)*0

        if passvalue_proper is not None:
            batch['passvalue_proper'] = passvalue_proper
        else:
            batch['passvalue_proper'] = None

        if exptime is None:
            exptime = 1.0
            
        self.batches.append(batch)
        self.num_batches = self.num_batches + 1

    def generate_speckleSeries(self,batch_id_list=None,outdir0='output/SpeckleSeries/',num_images_printed=0,
                                    vmin_fig=None,vmax_fig=None,title_fig='',
                                    use_emccd=False,use_photoncount=False,flag_return_contrast=False,flag_compute_normalization=False):
        """
        Generate a speckle series.
        
        Parameters
        ----------
        batch_id_list : list, optional
            List of batch IDs for which to generate speckle series.
        outdir0 : str, optional
            Output directory (default: 'output/SpeckleSeries/').
        num_images_printed : int, optional
            Number of images to print (default: 0).
        """
        outdir = self.paths['outdir']#outdir0+self.scene['name']+'/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if num_images_printed>0:
            outdir_images = outdir+'images/'
            if not os.path.exists(outdir_images):
                os.makedirs(outdir_images)
        
        # TODO below
        if not hasattr(self,'scenes'):
            print('You havent defined a scene')
            return []
        print("Starting the simulation of this observation, '"+self.name_OS+"'")

        if batch_id_list is None:
            batch_list = self.batches
        else:
            batch_list = [b for b in self.batches if any(batch_match==b['batch_id'] for batch_match in batch_id_list)]

        
        sz_im = self.corosims.sz_im
        
        # Generate mask
        sampling = self.corosims.options['sampling'][self.corosims.bandpass]
        iwa = 3
        owa = 9
        iwa_mask = make_circ_mask(sz_im,0,0,iwa/sampling)
        owa_mask = make_circ_mask(sz_im,0,0,owa/sampling)
        mask_field = owa_mask-iwa_mask
        
        for batch in batch_list:

            batch_id = batch['batch_id']
            batch_scene_id = batch['batch_scene_id']
            batch_scene = self.scenes[batch_scene_id]
            jitter_x = batch['jitter_x']
            jitter_y = batch['jitter_y']
            zindex = batch['zindex']
            zval_m = batch['zval_m']
            dm1_shear_x = batch['dm1_shear_x']
            dm2_shear_x = batch['dm2_shear_x']
            dm1_shear_y = batch['dm1_shear_y']
            dm2_shear_y = batch['dm2_shear_y']
            lyot_shift_x = batch['lyot_shift_x']
            lyot_shift_y = batch['lyot_shift_y']
            cgi_shift_x = batch['cgi_shift_x']
            cgi_shift_y = batch['cgi_shift_y']
            passvalue_proper = batch['passvalue_proper']
            V3PA = batch['V3PA']
            
            print("Starting Batch {}".format(batch_id))

            num_timesteps  = batch['num_timesteps']
                                    
            sources_in_scene = batch_scene["sources_in_scene"]
            sources_offaxis = [s for s in sources_in_scene if not s["flag_onaxis"]]
            sources_onaxis = [s for s in sources_in_scene if s["flag_onaxis"]]
            
            Ii_offaxis = np.zeros((sz_im,sz_im))
            for source_offaxis_dict in sources_offaxis:
                source_offaxis = self.sources[source_offaxis_dict["source_index_id"]]
                # X and Y separation:
                x_y_separation_mas = source_offaxis_dict["x_y_separation_mas"]
                sep_source = np.sqrt(x_y_separation_mas[0]**2+x_y_separation_mas[1]**2)
                PA = degenPA(-x_y_separation_mas[0],x_y_separation_mas[1])
                xoffset = -sep_source*np.sin((PA+V3PA)*np.pi/180)
                yoffset = sep_source*np.cos((PA+V3PA)*np.pi/180)
                passvalue_proper = {'source_x_offset_mas':xoffset, 
                                    'source_y_offset_mas':yoffset} 
                # import pdb 
                # pdb.set_trace()

                # # Define source in the corosims_core object, #TODO: i don't know if I like this way of doing it...
                # self.corosims.compute_spectrum(source_offaxis_dict["star_type"],source_offaxis_dict["vmag"])
                # Define source in the corosims_core object, #TODO: i don't know if I like this way of doing it...
                self.corosims.source = source_offaxis

                # Generate image
                Ii_offaxis = Ii_offaxis + self.corosims.generate_image(use_fpm=1,
                                                                        jitter_sig_x=0,jitter_sig_y=0, # we assume no jitter for an off-axis source
                                                                        passvalue_proper=passvalue_proper,
                                                                        use_emccd=False,flag_return_contrast=False)
            
            Ii_onaxis = np.zeros((num_timesteps,sz_im,sz_im))
            for KK,source_onaxis_dict in enumerate(sources_onaxis):
                source_onaxis = self.sources[source_onaxis_dict["source_index_id"]]
                # X and Y separation:
                x_y_separation_mas = source_onaxis_dict["x_y_separation_mas"]
                sep_source = np.sqrt(x_y_separation_mas[0]**2+x_y_separation_mas[1]**2)
                PA = degenPA(-x_y_separation_mas[0],x_y_separation_mas[1])
                xoffset = -sep_source*np.sin((PA+V3PA)*np.pi/180)
                yoffset = sep_source*np.cos((PA+V3PA)*np.pi/180)
                passvalue_proper = {'source_x_offset_mas':xoffset, 
                                    'source_y_offset_mas':yoffset} 
                
                # Define source in the corosims_core object, #TODO: i don't know if I like this way of doing it...
                self.corosims.source = source_onaxis

                # Compute the off-axis normalization to put it on the header
                if flag_return_contrast or flag_compute_normalization:
                    if len(sources_onaxis)>1:
                        print("More than one source is on-axis and you asked for contrast, how do we normalize?")
                        print("We'll assume that we normalize with the first on-axis source")
                    if KK==0:
                        maxI0_offaxis = self.corosims.compute_offaxis_normalization()
                        
                Ii_onaxis_onesource = np.zeros((num_timesteps,sz_im,sz_im))
                for II in range(num_timesteps):
                    print("Computing image num. {} out of {}".format(II+1,num_timesteps))

                    # Generate image
                    Ii_onaxis_II = self.corosims.generate_image(use_fpm=1,
                                                            jitter_sig_x=jitter_x[II],jitter_sig_y=jitter_y[II],
                                                            zindex=zindex,zval_m=zval_m[II],
                                                            dm1_shear_x=dm1_shear_x[II],dm2_shear_x=dm2_shear_x[II],
                                                            dm1_shear_y=dm1_shear_y[II],dm2_shear_y=dm2_shear_y[II],
                                                            lyot_shift_x=lyot_shift_x[II],lyot_shift_y=lyot_shift_y[II],
                                                            cgi_shift_x=cgi_shift_x[II],cgi_shift_y=cgi_shift_y[II],
                                                            passvalue_proper=passvalue_proper,
                                                            use_emccd=False,flag_return_contrast=False)
                    Ii_onaxis_onesource[II] = Ii_onaxis_II
                    
                    
                    # Produce Images if required by user
                    if II<num_images_printed:
                        Ii_crop = crop_data(Ii_onaxis_II*mask_field, nb_pixels=int(owa/sampling*1.4)) 
                        fig = plt.figure(figsize=(6,6))
                        plt.imshow(Ii_crop, cmap='hot',vmin=vmin_fig,vmax=vmax_fig)
                        plt.colorbar(fraction=0.046, pad=0.04)
                        plt.gca().invert_yaxis()
                        plt.title(title_fig)
                        plt.savefig(outdir_images+'ni_im{}.png'.format(II+1))
                        plt.close(fig)
                
                    # import pdb 
                    # pdb.set_trace()

                Ii_onaxis = Ii_onaxis + Ii_onaxis_onesource
                
            Ii_cube = Ii_onaxis + Ii_offaxis        
            if flag_return_contrast:
                Ii_cube_contrast = (Ii_onaxis + Ii_offaxis) / maxI0_offaxis 
            
            # Create header
            header_for_batch = self.create_header_for_batch(batch,maxI0_offaxis=maxI0_offaxis)
            # Write FITS file
            primary_hdu = pyfits.PrimaryHDU(data=Ii_cube,header=header_for_batch)
            primary_hdu.writeto(os.path.join(outdir,'Ii_cube_batch{}.fits'.format(batch_id)),overwrite=True)

            Ii_coadded = np.mean(Ii_cube,axis=0)
            primary_hdu = pyfits.PrimaryHDU(data=Ii_coadded,header=header_for_batch)
            primary_hdu.writeto(os.path.join(outdir,'Ii_coadded_batch{}.fits'.format(batch_id)),overwrite=True)

            if flag_return_contrast:
                primary_hdu = pyfits.PrimaryHDU(data=Ii_cube_contrast,header=header_for_batch)
                primary_hdu.writeto(os.path.join(outdir,'Ii_cube_contrast_batch{}.fits'.format(batch_id)),overwrite=True)
                Ii_coadded = np.mean(Ii_cube_contrast,axis=0)
                primary_hdu = pyfits.PrimaryHDU(data=Ii_coadded,header=header_for_batch)
                primary_hdu.writeto(os.path.join(outdir,'Ii_coadded_contrast_batch{}.fits'.format(batch_id)),overwrite=True)

    def create_header_for_batch(self,batch,maxI0_offaxis=None):
        header = pyfits.Header()

        # Add header keywords for Batch Information
        # header.insert(10,('', "====================="))
        # header.append('COMMENT', " BATCH INFORMATION ")
        # header.insert('', "=====================")
        # header.add_blank("=========frt============")
        header['BATCH_ID'] = batch['batch_id']
        header.set('JITTX', np.mean(batch['jitter_x']), 'Mean of jitter_x')
        header.set('JITTY', np.mean(batch['jitter_y']), 'Mean of jitter_y')
        zindex = batch['zindex']
        zval_m = batch['zval_m']
        if zindex is not None:
            for II,zer in enumerate(zindex):
                zv = np.mean(zval_m[:,II])
                header['Z{}_AVG'.format(zer)] = zv
                header.set('Z{}_AVG'.format(zer), zv, 'Average WFE of Zernike Noll #{}'.format(zer))
        header.set('DM1SHX', np.mean(batch['dm1_shear_x']), 'Mean of dm1_shear_x')
        header.set('DM1SHY', np.mean(batch['dm1_shear_y']), 'Mean of dm1_shear_y')
        header.set('DM2SHX', np.mean(batch['dm2_shear_x']), 'Mean of dm2_shear_x')
        header.set('DM2SHY', np.mean(batch['dm2_shear_y']), 'Mean of dm2_shear_y')
        header.set('LYSHX', np.mean(batch['lyot_shift_x']), 'Mean of lyot_shift_x')
        header.set('LYSHY', np.mean(batch['lyot_shift_y']), 'Mean of lyot_shift_y')
        header.set('CGISHX', np.mean(batch['cgi_shift_x']), 'Mean of cgi_shift_x')
        header.set('CGISHY', np.mean(batch['cgi_shift_y']), 'Mean of cgi_shift_y')
    
        # Add a blank line to separate sections
        # header['COMMENT'] = " "
    
        scene = self.scenes[batch['batch_scene_id']]
        sources_in_scene = scene["sources_in_scene"]
        sources_offaxis = [s for s in sources_in_scene if not s["flag_onaxis"]]
        sources_onaxis = [s for s in sources_in_scene if s["flag_onaxis"]]
        
        # Add header keywords for Scene Information
        # header['COMMENT'] = "====================="
        # header['COMMENT'] = " SCENE INFORMATION "
        # header['COMMENT'] = "====================="
        header['SCN_ID'] = batch['batch_scene_id']
        header['SCN_NAME'] = scene['name']
        header.set('NONAX', len(sources_onaxis), 'Number of sources on-axis')
        header.set('NOFFAX', len(sources_offaxis), 'Number of sources off-axis')
        if len(sources_onaxis)>0:
            # header['COMMENT'] = " ON-AXIS SOURCES "
            for II,source_in_scene in enumerate(sources_onaxis):
                header.set('SRCID', source_in_scene["source_index_id"], 'Source index ID')
                source = self.sources[source_in_scene["source_index_id"]]
                header.set('SRCNAME', source['name'], 'Source name')
                header.set('SRCTYPE', source['star_type'], 'Source spectral type')
                header.set('SRCVMAG', source['vmag'], 'source vmag')
                header.set('DX_MAS', source_in_scene["x_y_separation_mas"][0], 'X Separation from center [mas]')
                header.set('DY_MAS', source_in_scene["x_y_separation_mas"][1], 'Y Separation from center [mas]')
        if len(sources_offaxis)>0:
            # header['COMMENT'] = " OFF-AXIS SOURCES "
            for II,source_in_scene in enumerate(sources_offaxis):
                header.set('SRCID', source_in_scene["source_index_id"], 'Source index ID')
                source = self.sources[source_in_scene["source_index_id"]]
                header.set('SRCNAME', source['name'], '')
                header.set('SRCTYPE', source['star_type'], '')
                header.set('SRCVMAG', source['vmag'], '')
                header.set('DX_MAS', source_in_scene["x_y_separation_mas"][0], 'X Separation from center [mas]')
                header.set('DY_MAS', source_in_scene["x_y_separation_mas"][1], 'Y Separation from center [mas]')
        # Normalization for contrast
        if maxI0_offaxis is not None:
            # header['COMMENT'] = " "
            header.set('NORMI', maxI0_offaxis, 'Off-axis peak for contrast normalization')
        
        return header
            
    def load_batches_cubes(self):
        """
        Load precomputed image cubes for the scene.
        """
        datadir = self.paths['outdir']
        if len(self.batches)==0:
            warnings.warn("No batches!")

        for II,batch in enumerate(self.batches):
            flnm = os.path.join(datadir,'Ii_cube_batch{}.fits'.format(batch["batch_id"]))
            data = pyfits.open(flnm)
            im_cube = data[0].data
            flnm = os.path.join(datadir,'Ii_coadded_batch{}.fits'.format(batch["batch_id"]))
            data = pyfits.open(flnm)
            im_coadded = data[0].data
            hdr = data[0].header
            maxI0_offaxis = hdr['NORMI']
            batch['im_cube'] = im_cube
            batch['im_coadded'] = im_coadded
            batch['maxI0_offaxis'] = maxI0_offaxis
            flnm = os.path.join(datadir,'Ii_cube_emccd_batch{}.fits'.format(batch["batch_id"]))
            if os.path.exists(flnm):
                data = pyfits.open(flnm)
                im_cube = data[0].data
                batch['im_cube_emccd'] = im_cube
                flnm = os.path.join(datadir,'Ii_coadded_emccd_batch{}.fits'.format(batch["batch_id"]))
                data = pyfits.open(flnm)
                im_cube = data[0].data
                batch['im_coadded_emccd'] = im_cube

            self.batches[II] = batch
            
            
    def add_detector_noise_to_batches(self,label_out=''):
        """
        Add detector noise to precomputed image batches.
        
        Parameters
        ----------
        label_out : str, optional
            Label to append to output filenames (default: '').
        """
        from scipy.interpolate import interp1d
        
        # Load cubes into batches
        self.load_batches_cubes()
        
        # Loop over batches and add detector noise
        for II,batch in enumerate(self.batches):
            # Load cube
            im_cube = batch['im_cube']
            num_im = len(im_cube)
            
            # Define interpolator function
            fn = interp1d(np.linspace(0,1,num_im),im_cube,axis=0)
            
            # Number of frames to interpolate
            if 'num_frames_interp' not in batch:
                num_frames_interp = num_im
            else:
                num_frames_interp = batch['num_frames_interp']
            
            # Exposure time is uniform for each batch
            exptime = batch['exptime']
            
            # Do the interpolation
            im_cube_interp = fn(np.linspace(0,1,num_frames_interp))
            
            # Add detector noise to this new interpolated cube
            im_cube_interp_emccd = []
            for JJ in range(num_frames_interp):
                im_cube_interp_emccd.append(self.corosims.add_detector_noise(im_cube_interp[JJ],exptime))
                
            # Save FITS files
            hdulist = pyfits.PrimaryHDU(im_cube_interp_emccd)
            flnm_out = os.path.join(self.paths['outdir'],'Ii_cube_emccd_batch{}{}.fits'.format(batch['batch_id'],label_out))
            hdulist.writeto(flnm_out,overwrite=True)
            # Coadd and save
            im_coadded_interp_emccd = np.mean(im_cube_interp_emccd,axis=0)
            hdulist = pyfits.PrimaryHDU(im_coadded_interp_emccd)
            flnm_out = os.path.join(self.paths['outdir'],'Ii_coadded_emccd_batch{}{}.fits'.format(batch['batch_id'],label_out))
            hdulist.writeto(flnm_out,overwrite=True)
        
