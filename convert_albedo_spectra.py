#!/usr/bin/env python
# S. Hasler
# Code to convert custom source spectra of a specific format for CGISim use
# Source spectra must be placed into CGISim info_dir to be read into CGISim
# The spectrum units can be one of four types for CGISim:
# 
# |   Name   |   Units               | 
# |:--------:|:---------------------:|
# |  FLAM    |  erg / s / cm^2 / A   | 
# |    JY    |     Jansky            | 
# |   FNU    | erg / s / cm^2 / Hz   |  
# | PHOTLAM  | photon / s / cm^2 / A |

import numpy as np

def convert_custom_spectrum(source_path, source_file, stellar_spectrum_file, planet_separation, 
                            R_p, output_dir, output_dir2, source_wavelength='um', stellar_flux_units='flam'):
    '''
    Function to convert custom source albedo spectrum file into a file usable by CGISim

    Parameters
    ----------
    source_path : str
        Path to spectrum file
    source_file : str
        Spectrum file name to use in source_path
    stellar_spectrum_file : str
        Path to stellar spectrum file, including file name. Wavelength values must be in units
        of Angstroms.
    planet_separation : float
        Planet-star separation in AU
    R_p : float
        Planet radius in km
    output_dir : str
        Output directory to save new spectral file to
    output_dir2 : str
        Second output directory to save new spectral file (one needs to go in code directory, the other in cgisim.lib_dir)
    source_wavelength : str, optional
        Wavelength in original spectrum file, by default 'um'
    stellar_flux_units : str, optional
        Units of stellar flux spectrum from file. Should be either 'flam', 'fnu', 'jy', or 'photlam'
    
    '''
    source_spectrum_file = source_path + source_file 

    if source_wavelength == 'um':
        conversion = 1e4 # convert um to Angstroms
    if source_wavelength == 'nm':
        conversion = 10 # convert nm to A
    if source_wavelength == 'cm':
        conversion = 1e8 # convert cm to A
    if source_wavelength == 'A':
        conversion = 1 

    # Load in spectrum -- must have wavelength in first column, albedo spectrum in the second
    wavelength, alb_spec = np.loadtxt(source_spectrum_file, unpack=True)

    # Convert the planet wavelengths to Angstroms:
    wavelength = np.array([x*conversion for x in wavelength])

    # Get stellar spectrum
    star_wavelength, star_spec = np.loadtxt(stellar_spectrum_file, unpack=True)

    # Adjust stellar spectrum for distance of planet from star
    planet_separation = planet_separation * 1.496e8 # convert to km
    star_spec_atplanet = star_spec / planet_separation**2
    # Interpolate solar spectrum on planet wavelength grid
    stellar_flux_atplanet = np.interp(wavelength, star_wavelength, star_spec_atplanet)

    # Convert source spectrum from albedo to flux space
    planet_flux = alb_spec * stellar_flux_atplanet * (R_p / planet_separation)**2 # A_g(lambda) * F_star(lambda) * (Rp / r)^2
    fpfs = planet_flux / stellar_flux_atplanet

    # Write converted source spectrum to file:
    #   Header should contain only units of spectrum/wavelength
    wave_units = 'angstroms'
    spectrum_data = np.column_stack((wavelength, planet_flux))
    filename = source_file[:-4] + f"_Angstroms_{stellar_flux_units}.dat"
    output_path = output_dir + filename
    output_path2 = output_dir2 + filename
    np.savetxt(output_path, spectrum_data, fmt='%.4E', header=f'{wave_units},{stellar_flux_units}', 
               delimiter='\t', comments='')
    np.savetxt(output_path2, spectrum_data, fmt='%.4E', header=f'{wave_units},{stellar_flux_units}', 
               delimiter='\t', comments='')
    
    return alb_spec#, stellar_flux_atplanet, planet_flux, fpfs, wavelength

def get_source_mag(m_star, R_p, d, alb_spec, wavelength, phi_lambda=1.0, band_center=5738.0):
    '''
    Get magnitude of source given albedo spectrum, magnitude of host star, distance/size of source, 
    phase function, and center of bandpass.

    Parameters
    ----------
    m_star : float
        magntiude of host star in band
    R_p : float
        radius of source in km
    d : float
        source-star separation in km 
    alb_spec : array of floats
        Array of albedo spectrum of source
    wavelength : array of floats
        Array of wavelengths corresponding to albedo spectrum
    phi_lambda : float
        Value of phase function 
    band_center : float
        Wavelength value at center of bandpass in Angstroms

    Returns
    -------
    float
        Magnitude of source at bandpass center
    '''

    # Compute planet magnitude as a function of wavelength
    m_planet = m_star - 2.5 * np.log10(alb_spec * phi_lambda * (R_p / d)**2)

    # Get index of center of wavelength band to grab planet magnitude
    center_id = np.abs(wavelength - band_center).argmin()
    planet_mag = m_planet[center_id]

    return planet_mag
        
if __name__ == '__main__':
    # set paths
    source_path = "/Users/sammyh/Documents/Projects/planet_modeling/Cahoy_et_al_2010_Albedo_Spectra/albedo_spectra/"
    source_file = "Jupiter_1x_5AU_0deg.dat"
    stellar_spectrum_file = "/Users/sammyh/Documents/Projects/planet_modeling/Cahoy_et_al_2010_Albedo_Spectra/SOLARSPECTRUM.DAT"
    output_dir = "/Users/sammyh/Codes/cgisim_v3.1/cgisim/cgisim_info_dir/"
    output_dir2 = "/Users/sammyh/.local/lib/python3.9/site-packages/cgisim-3.1-py3.9.egg/cgisim/cgisim_info_dir/"

    # set constant planet properties
    planet_separation = 5 # AU
    R_p = 69911 # km
    d = planet_separation * 1.496e8 # convert to km
    wavelength = 5738 # wavelength center of band 1 in Angstroms
    m_star = 5.0 # apparent magnitude of host star

    alb_spec = convert_custom_spectrum(source_path=source_path, source_file=source_file, 
                                       stellar_spectrum_file=stellar_spectrum_file,
                                       planet_separation=planet_separation, R_p=R_p, output_dir=output_dir, 
                                       output_dir2=output_dir2)
    planet_mag = get_source_mag(m_star, R_p, d, alb_spec=alb_spec, wavelength=wavelength)
    print('Planet magnitude = ', planet_mag)
    