#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:40:43 2024
Modified exrensively on Sat Sep 27 10:27:00 2025 over spain

@author: nclark
"""

'''
IMPORTING MODULES
'''

#standard stuff
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import  AutoMinorLocator
import random
from time import time

#saving imagaes as PDFs
from PIL import Image  # install by > python3 -m pip install --upgrade Pillow  # ref. https://pillow.readthedocs.io/en/latest/installation.html#basic-installation

#used for fits file handling
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

#Import needed scipy libraries for curve_fit
import scipy.optimize

#Import needed sklearn libraries for RANSAC
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

#needed for fringe remover
import pickle 
from astropy.units import UnitBase, Unit, Quantity
from astropy.constants import h, c, k_B

#needed for unit_changer
import astropy.units as u

#needed for els' region function
import regions
from astropy.wcs import wcs
from astropy.stats import sigma_clip

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# needed for bethanys continuum code
import pandas
import scipy.io as sio
from astropy.io import ascii
from scipy.interpolate import UnivariateSpline
import statistics
from matplotlib.backends.backend_pdf import PdfPages
import copy
from math import floor, ceil

'''
TO DO
'''

# TODO

# make sure everything is indexed as data[:,y,x]
# rename the 'flux aligner' functions 

# set up these functions to work as class nocules per subchannel, maybe also
# a seperate stitched class?



####################################################################



'''
FUNCTIONS
'''



def planck_wl(
    x: Quantity, T: Quantity, output_unit: UnitBase | str = Unit("erg / (s cm^2 um sr)")
) -> Quantity:
    """
    The planck function.

    Parameters
    ----------
    x : Quantity
        Wavelength.
    T : Quantity
        Temperature.
    output_unit : Unit | str, optional
        Unit of the output. Default is "erg / (s cm^2 um sr)".

    Returns
    -------
    Quantity
        The planck function for temperature T, evaluated at the given wavelength(s).
    """
    return (
        2 * h * c**2 / x**5 / (np.exp((h * c / k_B / x / T).decompose()) - 1) / Unit("sr")
    ).to(output_unit) # fmt: skip


def modified_planck_wl(
    x: Quantity,
    T: Quantity,
    amp: float,
    gamma: float,
    output_unit: UnitBase | str = Unit("erg / (s cm^2 um sr)"),
) -> Quantity:
    """
    A modified blackbody, which is a planck function multiplied by a power law.

    Parameters
    ----------
    x : Quantity
        Wavelength.
    T : Quantity
        Temperature.
    amp : float
        amplitude (unitless)
    gamma: float
        Exponent of power law.

    Returns
    -------
    Quantity
        The planck function for temperature T, evaluated at the given wavelength(s).
    """
    return amp * planck_wl(x, T, output_unit).value * x.to("um").value ** (gamma + 2) # +2 converts to per freq units



def mean_from_wave(data, wavelengths, shape, wave_list):
    new = np.newaxis
    N = len(wave_list[:,0,0])
    M = len(wave_list[0,:,0])
    
    # array indices (case, index, upper or lower)
    temp_index = np.argmin(abs(wavelengths[new, :, new, new] - wave_list[:, new, :, :]), axis=1)

    means = np.ones((N, M, shape[0], shape[1]))
    for i in range(N):
        for j in range(M):
            # short form wavelength indices
            w1 = temp_index[i, j, 0]
            w2 = temp_index[i, j, 1]
            means[i, j] = np.nanmean(data[w1 : w2, :, :], axis=0)
        
    return means



def chi_plotter(i, y, x, chi_squared, chi_squared2, minA, minT, minA2, minT2, resolution):
    ax = plt.figure(f'{i}: {y}, {x}', figsize=(18,10)).add_subplot(121)
    plt.title(f'{i}: {y}, {x}, case 1 anchors')
    plt.imshow(np.log10(chi_squared), vmin=0.1, vmax=10)
    plt.scatter([minA], [minT], color='blue', edgecolors='yellow')
    ax.invert_yaxis()
    ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
    ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
    ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
    ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel('Amplitude index', fontsize=16)
    plt.ylabel('Temp index', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    ax = plt.figure(f'{i}: {y}, {x}', figsize=(18,10)).add_subplot(122)
    plt.title(f'{i}: {y}, {x}, case 2 anchors')
    plt.imshow(np.log10(chi_squared2), vmin=0.1, vmax=10)
    plt.scatter([minA2], [minT2], color='blue', edgecolors='yellow')
    ax.invert_yaxis()
    ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
    ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
    ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
    ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel('Amplitude index', fontsize=16)
    plt.ylabel('Temp index', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.savefig(f'PDFtime/overview/temp/{i}__y{y}_x{x}.png', dpi=resolution)
    plt.show()
    plt.close()
    
    
    
def remove_overlap(datasquare, index): # XXX update docstring
    
    wavelengths = datasquare[0]
    data = datasquare[1]
    error = datasquare[2]
    segment = datasquare[3]

    wave_a = wavelengths[:index]
    wave_b = wavelengths[index:]
    
    data_a = data[:index]
    data_b = data[index:]
    
    datasquare_a = datasquare[:, :index]
    datasquare_b = datasquare[:, index:]
    
    #check if wavelength interval is the same or different
    check_a = np.round(wave_a[-1] - wave_a[-2], 4)
    check_b = np.round(wave_b[1] - wave_b[0], 4)

    if check_a == check_b:
    
        #check where the overlap is
        overlap = np.argmin(abs(wave_a - wave_b[0]))
        if wave_a[overlap] - wave_b[0] < 0:
            overlap += 1
            
        # overlap = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]

        #find how many entries are overlapped, subtract 1 for index
        overlap_length = len(wave_a) -1 - overlap
                
        #combine arrays such that the first half of one is used, and the second half
        #of the other is used. This way data at the end of the wavelength range is avoided
        
        split_index = overlap_length/2
        
        #check if even or odd, do different things depending on which
        if overlap_length%2 == 0: #even
            lower_index = overlap + split_index
            upper_index = split_index
            #print(lower_index, upper_index)
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap + split_index + 0.5
            upper_index = split_index - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        
    else:
        #check where the overlap is, only works for wave_a
        overlap_a = np.argmin(abs(wave_a - wave_b[0]))
        # overlap_a = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        if wave_a[overlap_a] - wave_b[0] < 0:
            overlap_a += 1
        
        #find how many microns the overlap is
        overlap_micron = wave_a[-1] - wave_a[overlap_a]
        
        #find how many entries of wave_a are overlapped, subtract 1 for index
        overlap_length_a = len(wave_a) -1 - overlap_a
        split_index_a = overlap_length_a/2
        
        #number of indices in wave_B over the wavelength range
        overlap_length_b = int(overlap_micron/check_b)
        split_index_b = overlap_length_b/2
        
        #check if even or odd, do different things depending on which
        if overlap_length_a%2 == 0: #even
            lower_index = overlap_a + split_index_a
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap_a + split_index_a + 0.5
            
        if overlap_length_b%2 == 0: #even
            upper_index = split_index_b
        else: #odd, so split_index is a number of the form int+0.5
            upper_index = split_index_b - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)

    # wavelengths_new = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
    # data_new = np.concatenate((data_a[:lower_index], data_b[upper_index:]), axis=0)
    datasquare_new = np.concatenate((datasquare_a[:, :lower_index], datasquare_b[:, upper_index:]), axis=1)
    if datasquare_new[0, lower_index-1] >= datasquare_new[0, lower_index]:
        datasquare_new = np.delete(datasquare_new, lower_index, axis=1)
        
    return datasquare_new, lower_index
    
    

def print_time(msg, start_time=None):
    current_time = round(time(), 2)
    
    if start_time is not None:
        print(f'{msg} {current_time - start_time} s')
        
    return current_time



####################################################################

# XXX

'''
FUNCTIONS THAT HAVE ANALOGS IN CLASS
'''



def loading_function(file_loc): # removed header_index
    '''
    This function loads in JWST MIRI and NIRSPEC fits data cubes, and extracts wavelength 
    data from the header and builds the corresponding wavelength array.
    
    Parameters
    ----------
    file_loc
        TYPE: string
        DESCRIPTION: where the fits file is located.

    Returns
    -------
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns.
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [k,i,j] k is wavelength index, i and j are position index.
    error_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data.
                for [k,i,j] k is wavelength index, i and j are position index.
    '''
    
    #load in the data
    image_file = get_pkg_data_filename(file_loc)
    
    #header data
    science_header = fits.getheader(image_file, 1)
    
    #wavelength data from header
    number_wavelengths = science_header["NAXIS3"]
    wavelength_increment = science_header["CDELT3"]
    wavelength_start = science_header["CRVAL3"]
    
    #constructing the ending point using given data
    #subtracting 1 so wavelength array is the right size.
    wavelength_end = wavelength_start + (number_wavelengths - 1)*wavelength_increment

    #making wavelength array, in micrometers
    wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_increment)
    
    #extracting image data
    image_data = fits.getdata(image_file, ext=1)
    error_data = fits.getdata(image_file, ext=2)
    
    #sometimes wavelength array is 1 element short, this will fix that
    if len(wavelengths) != len(image_data):
        wavelength_end = wavelength_start + number_wavelengths*wavelength_increment
        wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_increment)

    return wavelengths, image_data, error_data



def single_emission_line_remover(wavelengths, image_data, wave_list, special=None):
    '''
    removes a single emission line occupying a specified wavelength range, by 
    replacing the line with a linear function. The slope is calculated by 
    using points below the blue end and above the red end.
    
    Parameters
    ----------
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_a and data_b joined 
            together as described above.
    image_data
        TYPE: 1d array of floats
        DESCRIPTION: a spectra, with the line to be removed.
        wave_list
            TYPE: list of floats
            DESCRIPTION: the wavelengths in microns, corresponding to the 
                beginning and ending of the line to be removed
    special
        TYPE: kwarg nonetype
        DESCRIPTION: making this not none will trigger special slope calculation 
            for troublesome lines

    Returns
    -------
    new_data 
        TYPE: 1d array of floats
        DESCRIPTION: spectra with emission line removed.
    '''
    
    #defining this within my function so i can make more advanced rounding code
    #without cluttering my function with repetition
    def temp_index_generator(wave):
        temp_index = np.where(np.round(wavelengths, 3) == wave)[0]
        if len(temp_index) == 0:
            temp_index = np.where(np.round(wavelengths, 2) == np.round(wave, 2))[0][0]
        else:
            temp_index = temp_index[0]
            
        return temp_index
        
    temp_index_1 = temp_index_generator(wave_list[0])
    temp_index_2 = temp_index_generator(wave_list[1])
    
    #calculating the slope of the line to use
    pah_slope_1 = np.median(image_data[temp_index_1 - 5:temp_index_1])
    if temp_index_1 < 5:
        pah_slope_1 = np.median(image_data[:temp_index_1])
    pah_slope_2 = np.median(image_data[temp_index_2:5+temp_index_2])
    if int(len(wavelengths)) - temp_index_2 < 5:
        pah_slope_2 = np.median(image_data[temp_index_2:])
        
    # special case when medians are dangerous 
    if special != None:
        pah_slope_1 = image_data[temp_index_1]
        pah_slope_2 = image_data[temp_index_2]
    
    pah_slope = (pah_slope_2 - pah_slope_1)/\
        (wavelengths[temp_index_2] - wavelengths[temp_index_1])

    #putting it all together
    no_line = pah_slope*(wavelengths[temp_index_1:temp_index_2] - wavelengths[temp_index_1]) + pah_slope_1
    
    #removing line in input array
    new_data = np.copy(image_data)
    
    for i in range(len(no_line)):
        new_data[temp_index_1 + i] = no_line[i]

    return new_data



def emission_line_remover_wrapper(wavelengths, image_data, wave_list):
    '''
    Wrapper function for the emission line remover, allowing it to work over the entire data cube and not
    just a single spectra.
    
    Parameters
    ----------
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_a and data_b joined together as described above.
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: a spectra, indices [k,i,j] where k is spectral index and i,j are position index
        wave_list
            TYPE: list of floats
            DESCRIPTION: the wavelengths in microns, corresponding to the beginning and ending of the line to be removed

    Returns
    -------
    new_data 
        TYPE: 3d array of floats
        DESCRIPTION: spectra with emission line removed.
    '''
    
    new_data = np.copy(image_data)
    for i in range(len(image_data[0,:,0])):
        for j in range(len(image_data[0,0,:])):
            new_data[:,i,j] = single_emission_line_remover(wavelengths, image_data[:,i,j], wave_list)
            
    return new_data



def nan_replacer(wavelengths, image_data):
    '''
    iterates through image_data, and replaces any nan values with the value preceding it
    
    Parameters
    ----------
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_a and data_b joined 
            together as described above.
    image_data
        TYPE: 1d array of floats
        DESCRIPTION: a spectra, with the line to be removed.

    Returns
    -------
    new_data 
        TYPE: 1d array of floats
        DESCRIPTION: spectra with NaNs replaced
    '''
    
    #left edge case, just set to 0
    if np.isnan(image_data[0]) == True:
        image_data[0] = 0
    
    for i in range(len(wavelengths)):
        if np.isnan(image_data[i]) == True:
            image_data[i] = image_data[i-1]
    
    return image_data



'''
def correct_units_astropy(cube, wavelengths):                                   # XXX does this need to exist
        """
        Corrects the units of cubes by changing them from MJy/sr to W/m^2/um/str (with astropy)
        
        Parameters
        ----------
        cube [str or subtract_fits]: the cube(1d array now) whose units need to be corrected (in MJy/sr)
        
        directory_cube_data [str]: the directory of a .fits spectral cube file (in MJy/sr)
    
        directory_wave [str]: the directory of the file with the wavelengths (in micrometers)
    
        function_wave [function]: the function used to read in the wavelengths
                                  get_data: for .xdr files
                                  get_wavlengths: for .tbl files
                                  get_wave_fits: for .fits files
    
        Returns
        -------
        np.array : an array in W/m^2/sr/um
    
        """
        final_cube = np.zeros(cube.shape)
        cube_with_units = (cube*10**6)*(u.Jy/u.sr)
        print('cube with units')
        for i in range(len(wavelengths)):
            final_cube[i] = cube_with_units[i].to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavelengths[i]*u.micron))
        return(final_cube)
'''



def linear_continuum(wavelengths, data, wave_list, tight=None):           # XXX MAKE SURE IT WORKS WITH CUBES AND NOT ONLY SPECTRA
                                                                                  # XXX need docstring

        array_y, array_x = 1, 1

        # determining number of anchor points from wave_list
        N = len(wave_list)
        
        temp_index = np.ones(N+2).astype(np.int64)
        temp_index[0] = 0
        temp_index[-1] = len(wavelengths) - 1
        
        for i in range(N):
            temp_index[i+1] = np.argmin(abs(wavelengths - wave_list[i]))
            
        # value used in slope median
        median_val = 15
        if tight != None:
            median_val = tight
        
        # calculating slope y vals
        pah_slope_y_vals = np.ones((N+2, array_y, array_x))
        
        for i in range(N+2):
            if i == 0:
                pah_slope_y_vals[i] = np.median(data[temp_index[i] : temp_index[i] + median_val], axis=0)
            elif i < N+1:
                pah_slope_y_vals[i] = np.median(data[temp_index[i] - median_val : temp_index[i] + median_val], axis=0)  
            else:
                pah_slope_y_vals[i] = np.median(data[temp_index[i] - median_val : temp_index[i]], axis=0)
        
        # calculating slopes
        pah_slope = np.ones((N+1, array_y, array_x))
        
        #need wavelengths[i] to have 2d shape
        wavelengths_cube = np.ones((len(wavelengths), array_y, array_x))
        for i in range(len(wavelengths)):
            wavelengths_cube[i] = wavelengths[i]
        
        for i in range(N+1):
            # short form wavelength indices        
            w1 = temp_index[i]
            w2 = temp_index[i+1]
            pah_slope[i] = (pah_slope_y_vals[i+1] - pah_slope_y_vals[i])/\
                (wavelengths_cube[w2] - wavelengths_cube[w1])
                    
        # calculating continuum
        continuum = 0*np.copy(data)
        
        j = 0
        for i in range(N+1):            
            # short form wavelength indices        
            w1 = temp_index[i]
            w2 = temp_index[i+1]
            
            while j < w2:
                continuum[j] = pah_slope[i]*(wavelengths_cube[j] - wavelengths_cube[w1]) + pah_slope_y_vals[i]
                j += 1
        
        continuum[-1] = pah_slope[-1]*(wavelengths_cube[-1] - wavelengths_cube[w1]) + pah_slope_y_vals[-2]
                
        return continuum



def pah_feature_integrator(wavelengths, data):                                    # XXX need docstring

    # changing units 
    final_cube = np.zeros(data.shape)
    cube_with_units = (data*10**6)*(u.Jy/u.sr)

    final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies =\
                                    u.spectral_density(wavelengths*u.micron))
        
    # final_cube = final_cube*(u.micron)                                       # XXX test units, try to optimize
    # final_cube = final_cube*(u.m)
    # final_cube = final_cube*(u.m)
    # final_cube = final_cube*(u.sr/u.W)
    
    final_cube = final_cube*((u.m**2)*u.micron*u.sr/u.W)
    
    integrand_temp = np.copy(data)
    for i in range(len(data)):
        integrand_temp[i] = float(final_cube[i])

    # performing numberical integration
    odd_sum = 0

    for i in range(1, len(integrand_temp), 2):
        odd_sum += integrand_temp[i] 

    even_sum = 0    

    for i in range(2, len(integrand_temp), 2):
        even_sum += integrand_temp[i] 
    
    #NOTE THAT THIS WILL NOT WORK IF WAVELENGTH CONTAINS MULTIPLE H; WILL NEED TO INTEGRATE
    #ALONG WAVELENGTH CHANNELS AND ADD THEM TOGETHER
    
    h = wavelengths[1] - wavelengths[0]
    
    integral = (h/3)*(integrand_temp[0] + integrand_temp[-1] + 4*odd_sum + 2*even_sum)
    
    return integral



def CalculateR(wavelength):                                                    # XXX need docstring, does nirspec exist?
    
    # 1A
    if 4.9 <= wavelength <= 5.74:
        coeff = [8.4645410e+03, -2.4806001e+03, 2.9600000e+02]
    # 1B
    elif 5.66 <= wavelength <= 6.63:
        coeff = [1.3785873e+04, -3.8733003e+03, 3.6100000e+02]
    # 1C
    elif 6.53 <= wavelength <= 7.65 :
        coeff = [9.0737793e+03, -2.0355999e+03, 1.7800000e+02]
    # 2A
    elif 7.51 <= wavelength <= 8.76:
        coeff = [-1.3392804e+04, 3.8513999e+03, -2.1800000e+02]
    # 2B
    elif 8.67 <= wavelength <= 10.15:
        coeff = [-3.0707996e+03, 1.0530000e+03, -4.0000000e+01]
    # 2C
    elif 10.01 <= wavelength <= 11.71:
        coeff = [-1.4632270e+04, 3.0245999e+03, -1.2700000e+02]
    # 3A
    elif 11.55 <= wavelength <= 13.47:
        coeff = [-6.9051500e+04, 1.1490000e+04, -4.5800000e+02]
    # 3B
    elif 13.29 <= wavelength <= 15.52:
        coeff = [3.2627500e+03, -1.9200000e+02, 9.0000000e+00]
    # 3C
    elif 15.41 <= wavelength <= 18.02:
        coeff = [-1.2368500e+04, 1.4890000e+03, -3.6000000e+01]
    # 4A
    elif 17.71 <= wavelength <= 20.94:
        coeff = [-1.1510681e+04, 1.2088000e+03, -2.7000000e+01]
    # 4B
    elif 20.69 <= wavelength <= 24.44:
        coeff = [-4.5252500e+03, 5.4800000e+02, -1.2000000e+01]
    # 4C
    elif 23.22 <= wavelength <= 28.1:
        coeff = [-4.9578794e+03, 5.5819995e+02, -1.2000000e+01]
                
    R = coeff[0] + coeff[1]*wavelength + coeff[2]*wavelength**2
        
    return(R)



def error_finder(wavelengths, data, feature_wavelength, feature_indices, error_index): # XXX removed integral
    #calculates error of assosiated integrals
    
    # changing units 
    
    data_temp = data[error_index-25:error_index+25]
    wavelengths_temp = wavelengths[error_index-25:error_index+25]
    
    final_cube = np.zeros(data_temp.shape)
    cube_with_units = (data_temp*10**6)*(u.Jy/u.sr)

    final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies =\
                                    u.spectral_density(wavelengths_temp*u.micron))
        
    # final_cube = final_cube*(u.micron)                                       # XXX test units, try to optimize
    # final_cube = final_cube*(u.m)
    # final_cube = final_cube*(u.m)
    # final_cube = final_cube*(u.sr/u.W)
    
    final_cube = final_cube*((u.m**2)*u.micron*u.sr/u.W)
    
    rms_data = np.copy(data_temp)
    for i in range(len(data_temp)):
        rms_data[i] = float(final_cube[i])

    # calculating RMS    
    rms = (np.var(rms_data))**0.5
    
    resolution = CalculateR(feature_wavelength)
    
    delta_wave = feature_wavelength/resolution
    
    num_points = (wavelengths[feature_indices[1]] - wavelengths[feature_indices[0]])/delta_wave
    
    error = rms*delta_wave*(num_points)**0.5
    
    return error



def spectra_stitcher_no_offset(wave_a, wave_b, data_a, data_b):
    '''
    This function takes in 2 adjacent wavelength and image data arrays, presumably 
    from the same part of the image fov (field of view), so they correspond to 
    the same location in the sky. It then finds which indices in the lower wavelength 
    data overlap with the beginning of the higher wavelength data, and combines 
    the 2 data arrays in the middle of this region. For this function, no scaling is done.
    
    It needs to work with arrays that may have different intervals, so it is split into 2
    to take a longer running but more careful approach if needed.
    
    Note that in the latter case, the joining is not perfect, and currently a gap
    of ~0.005 microns is present; this is much better than the previous gap of ~0.05 microns,
    as 0.005 microns corresponds to 2-3 indices.
    
    Parameters
    ----------
    wave_a
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the smaller wavelengths.
        wave_b
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the larger wavelengths.
    data_a
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_a.
            for [k,i,j] k is wavelength index, i and j are position index.
    data_b
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_b.
            for [k,i,j] k is wavelength index, i and j are position index.
            
    Returns
    -------
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, data_a and data_b joined together as described above.
            for [k,i,j] k is wavelength index, i and j are position index.
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_a and data_b joined together as described above.
    overlap
        TYPE: integer (index) OR tuple (index)
        DESCRIPTION: index of the wavelength value in wave_a that equals the first element in wave_b. In the 
        case of the two wavelength arrays having different intervals, overlap is instead a tuple of the regular
        overlap, followed by the starting index in the 2nd array.
        
        UPDATE: now returns lower_index and upper_index, as opposed to the indices where wave_a first meets wave_b,
        i.e. it now returns the indices where the connection happens.
        
    '''
    
    #check if wavelength interval is the same or different
    check_a = np.round(wave_a[1] - wave_b[0], 4)
    check_b = np.round(wave_b[1] - wave_b[0], 4)
    
    if check_a == check_b:
    
        #check where the overlap is
        overlap = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        
        #find how many entries are overlapped, subtract 1 for index
        overlap_length = len(wave_a) -1 - overlap
        
        #making a temp array to scale
        temp = np.copy(data_b)
                
        #combine arrays such that the first half of one is used, and the second half
        #of the other is used. This way data at the end of the wavelength range is avoided
        
        split_index = overlap_length/2
        
        #check if even or odd, do different things depending on which
        if overlap_length%2 == 0: #even
            lower_index = overlap + split_index
            upper_index = split_index
            #print(lower_index, upper_index)
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap + split_index + 0.5
            upper_index = split_index - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        
    else:
        #check where the overlap is, only works for wave_a
        overlap_a = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        
        #find how many microns the overlap is
        overlap_micron = wave_a[-1] - wave_a[overlap_a]
        
        #find how many entries of wave_a are overlapped, subtract 1 for index
        overlap_length_a = len(wave_a) -1 - overlap_a
        split_index_a = overlap_length_a/2
        
        #number of indices in wave_B over the wavelength range
        overlap_length_b = int(overlap_micron/check_b)
        split_index_b = overlap_length_b/2
        
        #making a temp array to scale
        temp = np.copy(data_b)
        
        #check if even or odd, do different things depending on which
        if overlap_length_a%2 == 0: #even
            lower_index = overlap_a + split_index_a
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap_a + split_index_a + 0.5
            
        if overlap_length_b%2 == 0: #even
            upper_index = split_index_b
        else: #odd, so split_index is a number of the form int+0.5
            upper_index = split_index_b - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        #print(lower_index, upper_index)

        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        #overlap = (overlap_a, overlap_length_b)
        overlap = (lower_index, upper_index)
    
    return image_data, wavelengths, overlap



def spectra_stitcher(wave_a, wave_b, data_a, data_b, offset=None):
    '''
    This function takes in 2 adjacent wavelength and image data arrays, presumably 
    from the same part of the image fov (field of view), so they correspond to 
    the same location in the sky. It then finds which indices in the lower wavelength 
    data overlap with the beginning of the higher wavelength data, and combines 
    the 2 data arrays in the middle of this region.
    
    It needs to work with arrays that may have different intervals, so it is split into 2
    to take a longer running but more careful approach if needed.
    
    Note that in the latter case, the joining is not perfect, and currently a gap
    of ~0.005 microns is present; this is much better than the previous gap of ~0.05 microns,
    as 0.005 microns corresponds to 2-3 indices.
    
    Parameters
    ----------
    wave_a
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the smaller wavelengths.
    wave_b
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the larger wavelengths.
    data_a
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_a.
            for [k,i,j] k is wavelength index, i and j are position index.
    data_b
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_b.
            for [k,i,j] k is wavelength index, i and j are position index.
            
    Returns
    -------
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, data_a and data_b joined together as described above.
            for [k,i,j] k is wavelength index, i and j are position index.
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_a and data_b joined together as described above.
    offset
        TYPE: float
        DESCRIPTION: offset applied to data
    overlap
        TYPE: tuple of indices
        DESCRIPTION: the index of the lower and upper array where the stitching occurs.
        
    '''
    
    #check if wavelength interval is the same or different
    check_a = np.round(wave_a[-1] - wave_a[-2], 4)
    check_b = np.round(wave_b[1] - wave_b[0], 4)

    if check_a == check_b:
    
        #check where the overlap is
        overlap = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]

        #find how many entries are overlapped, subtract 1 for index
        overlap_length = len(wave_a) -1 - overlap

        #making a temp array to scale
        temp = np.copy(data_b)
                
        #combine arrays such that the first half of one is used, and the second half
        #of the other is used. This way data at the end of the wavelength range is avoided
        
        split_index = overlap_length/2
        
        #check if even or odd, do different things depending on which
        if overlap_length%2 == 0: #even
            lower_index = overlap + split_index
            upper_index = split_index
            #print(lower_index, upper_index)
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap + split_index + 0.5
            upper_index = split_index - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        
        #apply offset
        distance = 10
        if overlap_length < 20:
            distance = 5
        if offset is None:
            offset1 = data_a[lower_index] - temp[upper_index]
            offset2 = data_a[lower_index - distance] - temp[upper_index - distance]
            offset3 = data_a[lower_index + distance] - temp[upper_index + distance]
            
            offset = (offset1 + offset2 + offset3)/3
        
        #offset = data_a[lower_index, i] - temp[upper_index, i]
        temp = temp + offset
        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        
    else:
        #check where the overlap is, only works for wave_a
        overlap_a = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        
        #find how many microns the overlap is
        overlap_micron = wave_a[-1] - wave_a[overlap_a]
        
        #find how many entries of wave_a are overlapped, subtract 1 for index
        overlap_length_a = len(wave_a) -1 - overlap_a
        split_index_a = overlap_length_a/2
        
        #number of indices in wave_B over the wavelength range
        overlap_length_b = int(overlap_micron/check_b)
        split_index_b = overlap_length_b/2

        #making a temp array to scale
        temp = np.copy(data_b)
        
        #check if even or odd, do different things depending on which
        if overlap_length_a%2 == 0: #even
            lower_index = overlap_a + split_index_a
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap_a + split_index_a + 0.5
            
        if overlap_length_b%2 == 0: #even
            upper_index = split_index_b
        else: #odd, so split_index is a number of the form int+0.5
            upper_index = split_index_b - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        #print(lower_index, upper_index)
        
        #apply offset
        distance = 10
        if overlap_length_a < 20 or overlap_length_b < 20:
            distance = 5
        
        if offset is None:
            offset1 = data_a[lower_index] - temp[upper_index]
            offset2 = data_a[lower_index - distance] - temp[upper_index - distance]
            offset3 = data_a[lower_index + distance] - temp[upper_index + distance]
            
            offset = (offset1 + offset2 + offset3)/3



        temp = temp + offset
        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        overlap = (lower_index, upper_index)
    
    return image_data, wavelengths, overlap, offset



def spectra_stitcher_special(wave_a, wave_b, data_a, data_b, offset=None):
    '''
    This function takes in 2 adjacent wavelength and image data arrays, presumably 
    from the same part of the image fov (field of view), so they correspond to 
    the same location in the sky. It then finds which indices in the lower wavelength 
    data overlap with the beginning of the higher wavelength data, and combines 
    the 2 data arrays in the middle of this region. For this function, no scaling is done.
    
    It needs to work with arrays that may have different intervals, so it is split into 2
    to take a longer running but more careful approach if needed.
    
    Note that in the latter case, the joining is not perfect, and currently a gap
    of ~0.005 microns is present; this is much better than the previous gap of ~0.05 microns,
    as 0.005 microns corresponds to 2-3 indices.
    
    Parameters
    ----------
    wave_a
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the smaller wavelengths.
        wave_b
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the larger wavelengths.
    data_a
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_a.
            for [k,i,j] k is wavelength index, i and j are position index.
    data_b
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_b.
            for [k,i,j] k is wavelength index, i and j are position index.
            
    Returns
    -------
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, data_a and data_b joined together as described above.
            for [k,i,j] k is wavelength index, i and j are position index.
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_a and data_b joined together as described above.
    overlap
        TYPE: integer (index) OR tuple (index)
        DESCRIPTION: index of the wavelength value in wave_a that equals the first element in wave_b. In the 
        case of the two wavelength arrays having different intervals, overlap is instead a tuple of the regular
        overlap, followed by the starting index in the 2nd array.
        
        UPDATE: now returns lower_index and upper_index, as opposed to the indices where wave_a first meets wave_b,
        i.e. it now returns the indices where the connection happens.
        
    '''
    
    #check if wavelength interval is the same or different
    check_a = np.round(wave_a[1] - wave_b[0], 4)
    check_b = np.round(wave_b[1] - wave_b[0], 4)
    
    if check_a == check_b:
    
        #check where the overlap is
        overlap = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        
        #find how many entries are overlapped, subtract 1 for index
        overlap_length = len(wave_a) -1 - overlap
        
        #making a temp array to scale
        temp = np.copy(data_b)
                
        #combine arrays such that the first half of one is used, and the second half
        #of the other is used. This way data at the end of the wavelength range is avoided
        
        split_index = overlap_length/2
        
        #check if even or odd, do different things depending on which
        if overlap_length%2 == 0: #even
            lower_index = overlap + split_index
            upper_index = split_index
            #print(lower_index, upper_index)
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap + split_index + 0.5
            upper_index = split_index - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        
    else:
        #check where the overlap is, only works for wave_a
        overlap_a = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        
        #find how many microns the overlap is
        overlap_micron = wave_a[-1] - wave_a[overlap_a]
        
        #find how many entries of wave_a are overlapped, subtract 1 for index
        overlap_length_a = len(wave_a) -1 - overlap_a
        split_index_a = overlap_length_a/2
        
        #number of indices in wave_B over the wavelength range
        overlap_length_b = int(overlap_micron/check_b)
        split_index_b = overlap_length_b/2
        
        #making a temp array to scale
        temp = np.copy(data_b)
        
        #check if even or odd, do different things depending on which
        if overlap_length_a%2 == 0: #even
            lower_index = overlap_a + split_index_a
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap_a + split_index_a + 0.5
            
        if overlap_length_b%2 == 0: #even
            upper_index = split_index_b
        else: #odd, so split_index is a number of the form int+0.5
            upper_index = split_index_b - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        #print(lower_index, upper_index)

        #hard coded because the offset is weird around 7.58, the usual strat wont work
            
            #using a wavelength of 7.525 roughly
        if offset is not None:
            pass
        else:
            offset = data_a[1243] - temp[11]
        temp = temp + offset

        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        #overlap = (overlap_a, overlap_length_b)
        overlap = (lower_index, upper_index)
    
    return image_data, wavelengths, offset # overlap


# used to be extract_spectra_from_regions_one_pointing_no_bkg
def region_to_mask(fname_cube, data, fname_region):                            # docstring, standardize names
    
    reg = regions.Regions.read(fname_region, format='ds9')
    fits_cube = fits.open(fname_cube)
    w = wcs.WCS(fits_cube[1].header).dropaxis(2)

    region_indicator = np.zeros((len(data[:,0]), len(data[0,:])))

    # loop over regions in .reg file
    for i in range(len(reg)):
        regmask = reg[i].to_pixel(w).to_mask(mode='subpixels', subpixels=1).to_image(shape=data.shape[0:])
        if regmask is not None:
            for ix in range(data.shape[0]):
                for iy in range(data.shape[1]):
                    if regmask[ix, iy] == 1:
                        region_indicator[ix, iy] = 1

    return region_indicator
    


def weighted_mean_finder_rms(data, rms):
    '''
    This function takes a weighted mean of the (assumed background-subtracted, 
    in the case of JWST cubes) data, for 3 dimensional arrays.
    The mean is taken over the 1st and 2nd indicies (not the 0th), i.e. the spacial dimensions.
    For the weights, the RMS of the nearby continua is used.
    
    Parameters
    ----------
    data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    rms
        TYPE: 2d array of floats
        DESCRIPTION: RMS values corresponding to area near where weighted mean is found, for each spacial pixel
    
    Returns
    -------
    weighted_mean
        TYPE: 1d array of floats
        DESCRIPTION: weighted mean of the background-subtracted input data 
            spacial dimensions, as a spectra.
    '''
    
    #replacing nans with 0, as data has nans on border
    where_are_NaNs = np.isnan(data) 
    data[where_are_NaNs] = 0
    
    #weighted mean array
    weighted_mean_temp = np.copy(data)
    
    #big rms means noisy data so use the inverse as weights
    weights = 1/rms
    
    # weights equal to 1.0 are set to 0
    for y in range(len(data[0,:,0])):
        for x in range(len(data[0,0,:])):
            if weights[y,x] == 1.0:
                weights[y,x] = 0
    
    for y in range(len(data[0,:,0])):
        for x in range(len(data[0,0,:])):
            #adding single components of weighted mean to lists, to sum later
            weighted_mean_temp[:,y,x] = (weights[y,x])*(data[:, y, x])

    #summing to get error and weighted mean for this wavelength
    weighted_mean = (np.sum(weighted_mean_temp, axis=(1,2)))/(np.sum(weights))
    
    return weighted_mean



def weighted_mean_finder_rms_template(data, rms, y_points, x_points):
    '''
    This function takes a weighted mean of the (assumed background-subtracted, 
    in the case of JWST cubes) data, for 3 dimensional arrays.
    The mean is taken over lists of provided spacial indices
    For the weights, the RMS of the nearby continua is used.
    
    Parameters
    ----------
    data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    rms
        TYPE: 2d array of floats
        DESCRIPTION: RMS values corresponding to area near where weighted mean is found, for each spacial pixel
    
    Returns
    -------
    weighted_mean
        TYPE: 1d array of floats
        DESCRIPTION: weighted mean of the background-subtracted input data 
            spacial dimensions, as a spectra.
    '''
    
    #replacing nans with 0, as data has nans on border
    where_are_NaNs = np.isnan(data) 
    data[where_are_NaNs] = 0
    
    #big rms means noisy data so use the inverse as weights
    weights = 1/rms
    
    # weights equal to 1.0 are set to 0
    for y in range(len(data[0,:,0])):
        for x in range(len(data[0,0,:])):
            if weights[y,x] == 1.0:
                weights[y,x] = 0
    
    # setting all weights not specified by the points to 0
    
    weights_temp = np.zeros(rms.shape)
    for i in range(len(y_points)):
        weights_temp[y_points[i], x_points[i]] = 1
    
    for y in range(len(data[0,:,0])):
        for x in range(len(data[0,0,:])):
            if weights_temp[y,x] == 0:
                weights[y,x] = 0
    
    #weighted mean array
    weighted_mean_temp = np.copy(data)

    for y in range(len(data[0,:,0])):
        for x in range(len(data[0,0,:])):
            #adding single components of weighted mean to lists, to sum later
            weighted_mean_temp[:,y,x] = (weights[y,x])*(data[:, y, x])

    
    #all values not in list have rms to 0 and therefore dont contribute

    #summing to get error and weighted mean for this wavelength
    weighted_mean = (np.sum(weighted_mean_temp, axis=(1,2)))/(np.sum(weights))
    
    return weighted_mean



def regrid(data, rms, N):
    '''
    This function regrids a data cube, such that its pixel size goes from 1x1 to NxN, where N is specified.
    This is done by taking a weighted mean. Note that if the size of the array is not
    divisible by N, the indices at the end are discarded. 
    This should be ok since edge pixels are usually ignored.
    
    Parameters
    ----------
    data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data cube to be rebinned.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    rms
        TYPE: 2d array of floats
        DESCRIPTION: RMS values corresponding to area near where weighted mean is found, for each spacial pixel
    N
        TYPE: positive integer
        DESCRIPTION: the value N such that the number of pixels that go into the new pixel are N^2,
            i.e. before eacch pixel is 1x1, after its NxN per pixel.

    Returns
    -------
    rebinned_data
        TYPE: 3d array of floats
        DESCRIPTION: new data, on a smaller grid size in the positional dimensions.
    '''
    
    #defining current size
    size_y = len(data[0,:,0])
    size_x = len(data[0,0,:])
    
    #Figure out if any indices need to be discarded, so that the current size will
    #be divisible by N
    remainder_y = size_y % N
    remainder_x = size_x % N
    
    if remainder_y != 0:
        size_y = size_y - remainder_y
        
    if remainder_x != 0:
        size_x = size_x - remainder_x

    #building new arrays
    size_wavelength = int(len(data[:,0,0]))
    
    rebinned_data = np.zeros((size_wavelength, int(size_y/N), int(size_x/N)))
    
    for y in range(0, size_y, N):
        for x in range(0, size_x, N):
            #note that y:y+N will have y+1,...,y+N, with length N, so want to subtract 1 from these to include y
            
            #taking weighted mean over the pixels to be put in 1 bin
            temp_data =\
                weighted_mean_finder_rms(
                data[:, y:y + N, x:x + N], rms[y:y + N, x:x + N])
            
            #adding new pixel to array. y/N and x/N should always be integers, because the remainder was removed above.
            rebinned_data[:, int(y/N), int(x/N)] = temp_data
            
    return rebinned_data



def regrid_undoer(list_x, list_y):
    index = len(list_x)
    
    new_list_x = []
    new_list_y = []
    
    for i in range(index):
        new_list_x.append(2*list_x[i])
        new_list_y.append(2*list_y[i])
        
        new_list_x.append(2*list_x[i] + 1)
        new_list_y.append(2*list_y[i])
        
        new_list_x.append(2*list_x[i])
        new_list_y.append(2*list_y[i] + 1)
        
        new_list_x.append(2*list_x[i] + 1)
        new_list_y.append(2*list_y[i] + 1)

    return new_list_x, new_list_y



'''
BETHANYS CONTINUUM CODE
'''



class Anchor_points_and_splines():
    """
    A class that finds the anchor ponits and splines for the continua.
    """

    def __init__(self, spectral_cube, directory_ipac, wavelengths, length = None):
        """
        The constructor for Anchor_points_and_splines
        
        Parameters
        ----------
        spectral_cube [array-like]: the array with the spectral cube

        directory_ipac [str]: the directory of the ipac file with the anchor info

        wavelengths [array-like]: the array with the wavelengths

        length [int or NoneType]: How many consecutive continuum points need to be above the data
                                  to be an issue. If None, no continua overshoot correcting occurs.

        """

        self.wavelengths = wavelengths

        self.spectral_cube = spectral_cube
        anchor_ipac_data = ascii.read(directory_ipac, format = 'ipac')
        low_to_high_inds = np.array(anchor_ipac_data['x0']).argsort()
        # Because UnivariateSpline needs ordered values
        self.Pl_inds = [i for i, x in enumerate(anchor_ipac_data['on_plateau'][low_to_high_inds]) if x == "True"]
        if len(self.Pl_inds) > 0:
            self.Pl_starting_waves = anchor_ipac_data['x0'][low_to_high_inds][self.Pl_inds]
        self.moments = anchor_ipac_data['moment'][low_to_high_inds]
        self.starting_waves = anchor_ipac_data['x0'][low_to_high_inds]
        self.x0_mins = anchor_ipac_data['x0_min'][low_to_high_inds]
        self.x0_maxes = anchor_ipac_data['x0_max'][low_to_high_inds]
        self.bumps = anchor_ipac_data['bumps'][low_to_high_inds]
        self.starting_waves = anchor_ipac_data['x0'][low_to_high_inds]
        self.length = length



    def starting_anchors_x(self):
        """
        Gets the desired wavelengths for the starting anchor points.

        Returns
        -------
        The anchors points and their indices within wavelengths
        """
        indices = []
        for wave in self.starting_waves:
            indices.append(np.abs(np.asarray(self.wavelengths) - wave).argmin())
        anchors = list(self.wavelengths[indices])
        return(anchors, indices)



    def find_min_brightness(self, wave, wave_min, wave_max, pixel):
        """
        Finds the initial x and y values of an anchor point to use when the user wants the lowest brightness.

        Parameters
        ---------

        wave [float]: a wavelength between wave_min and wave_max

        wave_min, wave_max [float or int]: the minimum and maximum wavelength values to look for a minimum brightness
                                           between

        pixel [lst]: a pixel within the data cube (e.g. [10, 15])

        Returns
        -------

        The minimum brightness value and its corresponding wavelength
        """
        wavelengths = self.wavelengths
        wanted_wavelength_inds = np.where(np.logical_and(wavelengths > wave_min, wavelengths < wave_max))[0]
        brightness_list = list(self.spectral_cube[wanted_wavelength_inds, pixel[1], pixel[0]])
        brightness_anchor = min(brightness_list)
        ind_lowest_brightness = wanted_wavelength_inds[brightness_list.index(min(brightness_list))]
        wavelength_to_change = wavelengths[ind_lowest_brightness]
        return(brightness_anchor, wavelength_to_change)

    def new_anchor_points(self, pixel):
        """
        Finds more accurate lists of x and y values for anchor points (based on the 'moment' column of
        the anchor ipac table)

        Parameters
        ---------
        pixel [lst]: a pixel within the data cube (e.g. [10, 15])

        Returns
        -------
        A list of brightness and wavelength values for anchor points

        """
        cube = self.spectral_cube
        moments = self.moments
        wavelength_anchors_data =  self.starting_anchors_x()
        wavelength_anchors = wavelength_anchors_data[0]
        anchor_indices = wavelength_anchors_data[1]
        # To get desired list size - will change elements of list in code
        brightness_anchors = list(range(len(wavelength_anchors)))
        for ind in range(len(anchor_indices)):
            if moments[ind] == 1: # Find the mean between nearby points
                list_for_mean = [cube[anchor_indices[ind]][pixel[1]][pixel[0]]]
                brightness_anchors[ind] = statistics.mean(list_for_mean)
            elif moments[ind] == 2: # Find the mean between more nearby points
                list_for_mean = cube[anchor_indices[ind] - 2 : anchor_indices[ind] + 2, pixel[1], pixel[0]]
                brightness_anchors[ind] = np.nanmean(list_for_mean)
            elif moments[ind] == 3:
             # Find the average of two anchor point means and places the x location between them
             # Currently, the second anchor point for the average will be 0.3 microns behind the first
             # Find indice of anchor point not given:
                 wavelengths = self.wavelengths
                 wave = wavelength_anchors[ind] + 0.3
                 location = np.abs(np.asarray(wavelengths) - wave).argmin()
                 list_for_mean_1 = [cube[anchor_indices[ind]][pixel[1]][pixel[0]],
                                    cube[anchor_indices[ind]-1][pixel[1]][pixel[0]],
                                    cube[anchor_indices[ind]+1][pixel[1]][pixel[0]]]
                 list_for_mean_2 = [cube[location][pixel[1]][pixel[0]],
                                    cube[location-1][pixel[1]][pixel[0]],
                                    cube[location+1][pixel[1]][pixel[0]]]
                 brightness_anchors[ind] = (statistics.mean(list_for_mean_1) + statistics.mean(list_for_mean_2))/2
                 wavelength_anchors[ind] = (wave + wavelength_anchors[ind])/2
            elif moments[ind] == 4:
                pt_inds = np.where(np.logical_and(self.wavelengths > self.x0_mins[ind],
                                                  self.wavelengths < self.x0_maxes[ind]))[0]
                brightness_anchors[ind] = np.average(cube[pt_inds, pixel[1], pixel[0]])
                wavelength_anchors[ind] = (self.x0_mins[ind] + self.x0_maxes[ind])/2
            elif moments[ind] == 5: # Find the mean between lots of points (deal with fringes)
                list_for_mean = cube[anchor_indices[ind] - 20 : anchor_indices[ind] + 20, pixel[1], pixel[0]]
                brightness_anchors[ind] = np.nanmean(list_for_mean)
            elif moments[ind] == 0 or self.bumps[ind] == "True":
                # Find the min brightness within a wavelength region
                wave = wavelength_anchors[ind]
                wave_min = self.x0_mins[ind]
                wave_max = self.x0_maxes[ind]
                brightness_anchor, wavelength_to_change = self.find_min_brightness(wave, wave_min,
                                                                                   wave_max, pixel)
                brightness_anchors[ind] = brightness_anchor
                wavelength_anchors[ind] = wavelength_to_change
        return(brightness_anchors, wavelength_anchors)



    def get_anchors_for_all_splines(self, pixel):
        """
        Creates the GS, LS, and plateaus' anchor points' values (brightnesses and wavelengths)

        Parameters
        ---------
        pixel [lst]: a pixel within the data cube (e.g. [10, 15])

        Returns
        -------
        A new list of brightness and wavelength values for anchor points

        """
        if len(self.Pl_inds) > 0:
            plateau_waves = self.Pl_starting_waves
        anchor_info = self.new_anchor_points(pixel)
        brightness_anchors = anchor_info[0]
        new_wavelength_anchors = anchor_info[1]
        if "True" in self.bumps:
        # We need a continuum that includes the bumps (LS) and another that doesn't (GS)
            LS_brightness = brightness_anchors
            LS_wavelengths = new_wavelength_anchors
            no_bump_inds = [i for i, x in enumerate(self.bumps) if x == "False"]
            GS_brightness = [LS_brightness[ind] for ind in no_bump_inds]
            GS_wavelengths = [LS_wavelengths[ind] for ind in no_bump_inds]
        if len(self.Pl_inds) > 0:
        # We need to add a plateau (PL) continuum
            indices = []
            for anchor in plateau_waves:
                indices.append(np.abs(np.asarray(new_wavelength_anchors) - anchor).argmin())
            PL_wavelengths = [new_wavelength_anchors[ind] for ind in indices]
            PL_brightness = [brightness_anchors[ind] for ind in indices]
        if len(self.Pl_inds) > 0 and "True" in self.bumps:
            return(LS_brightness, LS_wavelengths, GS_brightness, GS_wavelengths, PL_brightness,
                   PL_wavelengths)
        elif len(self.Pl_inds) > 0 and "True" not in self.bumps:
            return(GS_brightness, GS_wavelengths, PL_brightness, PL_wavelengths)
        elif len(self.Pl_inds) == 0 and "True" in self.bumps:
            return(LS_brightness, LS_wavelengths, GS_brightness, GS_wavelengths)
        else:
            return(brightness_anchors, new_wavelength_anchors)



    def lower_brightness(self, anchor_waves, anchor_brightness, Cont, pixel):
        """
        Checks if the continuum is higher than the brightness for self.length consecutive points and
        lowers the continuum if it is
        This function likely needs some refinement

        Parameters
        ----------
        anchor_waves [lst]: a list of the anchor's wavelengths

        anchor_brightnesses [lst]: a list of the anchor's brightnesses

        Cont [UnivariateSpline object]: a continuum

        pixel [lst]: a pixel within the data cube (e.g. [10, 15])

        """
        pixel_brightnesses = self.spectral_cube[:, pixel[1], pixel[0]]
        no_cont_brightnesses = pixel_brightnesses - Cont
        below_zero_inds = np.where(no_cont_brightnesses < 0)[0] # Where data is less than the continuum
        # A few points bellow zero in no_cont_brightnesses could be due to noise, but a large range of
        # consecutive values below zero is a problem with the continuum
        consecutive_diff = np.diff(below_zero_inds)
        consecutive_diff_not_1 = np.where(consecutive_diff != 1)[0]
        split_below_zero_inds = np.split(below_zero_inds, consecutive_diff_not_1 + 1)
        # split_below_zero_inds is a list of arrays. The arrays are split based on where
        # the indicies of the points below zero aren't consecutive
        for array in split_below_zero_inds:
        # May be redundant if two arrays with consecutive < 0 values are between the same 2 anchor points
            if len(array) > self.length:
                subtracted_brightness_lower = np.median(no_cont_brightnesses[array])
                # Find the nearest anchor ind to the start of the problem area
                anchor_ind = np.abs(anchor_waves - self.wavelengths[array[0]]).argmin()
                anchor_brightness[anchor_ind] = anchor_brightness[anchor_ind] + subtracted_brightness_lower
                # Recall that subtracted_brightness_min is less than 0
                # Adding it will lower brightness values
        new_Cont = UnivariateSpline(anchor_waves, anchor_brightness, k = 3, s = 0)(self.wavelengths)
        # s = 0 so anchor points can't move
        # k = 3 for cubic
        return(new_Cont, anchor_brightness)



    def get_splines_with_anchors(self, pixel):
        """
        Creates cubic splines (LS, GS, and plateau).  Note that the plateau spline is also
        made using lines on both ends (with the cubic spline in the middle)

        Parameters
        ---------
        pixel [lst]: a pixel within the data cube (e.g. [10, 15])

        Returns
        -------
        A dictionary of continua, as well as the values of the wavelengths and brightnesses used for the
        anchor points of each continuum

        """
        all_wavelengths = self.wavelengths
        anchor_data = self.get_anchors_for_all_splines(pixel)
        spline_and_anchor_dict = {}
        if "True" in self.bumps:
            spline_and_anchor_dict["LS_wave_anchors"] = anchor_data[1]
            spline_and_anchor_dict["LS_brightness_anchors"] = anchor_data[0]
            spline_and_anchor_dict["GS_wave_anchors"] = anchor_data[3]
            spline_and_anchor_dict["GS_brightness_anchors"] = anchor_data[2]
            ContLS = UnivariateSpline(spline_and_anchor_dict["LS_wave_anchors"],
                                      spline_and_anchor_dict["LS_brightness_anchors"],
                                      k = 3, s = 0)(all_wavelengths)
            if self.length != None:
                new_LS = self.lower_brightness(spline_and_anchor_dict["LS_wave_anchors"],
                                               spline_and_anchor_dict["LS_brightness_anchors"],
                                               ContLS, pixel)
                spline_and_anchor_dict["LS_brightness_anchors"] = new_LS[1]
                spline_and_anchor_dict["ContLS"] = new_LS[0]
            else:
                spline_and_anchor_dict["ContLS"] = ContLS
            if len(self.Pl_inds) > 0:
                spline_and_anchor_dict["PL_wave_anchors"] = anchor_data[5]
                spline_and_anchor_dict["PL_brightness_anchors"] = anchor_data[4]
        else:
            spline_and_anchor_dict["GS_wave_anchors"] = anchor_data[1]
            spline_and_anchor_dict["GS_brightness_anchors"] = anchor_data[0]
            if len(self.Pl_inds) > 0:
                spline_and_anchor_dict["PL_wave_anchors"] = anchor_data[3]
                spline_and_anchor_dict["PL_brightness_anchors"] = anchor_data[2]
        ContGS = UnivariateSpline(spline_and_anchor_dict["GS_wave_anchors"],
                                  spline_and_anchor_dict["GS_brightness_anchors"],
                                  k = 3, s = 0)(all_wavelengths)
        if self.length != None:
            new_GS = self.lower_brightness(spline_and_anchor_dict["GS_wave_anchors"],
                                           spline_and_anchor_dict["GS_brightness_anchors"],
                                           ContGS, pixel)
            spline_and_anchor_dict["GS_brightness_anchors"] = new_GS[1]
            spline_and_anchor_dict["ContGS"] = new_GS[0]
        else:
            spline_and_anchor_dict["ContGS"] = ContGS
        if len(self.Pl_inds) > 0: # Easier to do this here since two cases above have PL conts
            ContPL = copy.deepcopy(spline_and_anchor_dict["ContGS"])
            for plateau in range(int(len(self.Pl_inds)/2)): # Each Pl defined by 2 points
                line = UnivariateSpline([spline_and_anchor_dict["PL_wave_anchors"][0 + 2*plateau],
                                         spline_and_anchor_dict["PL_wave_anchors"][1 + 2*plateau]],
                                        [spline_and_anchor_dict["PL_brightness_anchors"][0 + 2*plateau],
                                         spline_and_anchor_dict["PL_brightness_anchors"][1 + 2*plateau]],
                                        k = 1, s = 0)(all_wavelengths) # k = 1 for a line
                indices_contPL = np.where(np.logical_and(all_wavelengths > spline_and_anchor_dict["PL_wave_anchors"][0 + 2*plateau],
                                          all_wavelengths < spline_and_anchor_dict["PL_wave_anchors"][1 + 2*plateau]))[0]
                ContPL[indices_contPL] = line[indices_contPL]
            spline_and_anchor_dict["ContPL"] = ContPL
        return(spline_and_anchor_dict)



    def fake_get_splines_with_anchors(self):
        """
        A function that returns the same output as get_splines_with_anchors, but everything is
        0. Intended for flagged pixels.

        Returns
        -------
        See get_splines_with_anchors

        """
        spline_and_anchor_dict = {}
        splines = np.zeros(len(self.wavelengths))
        bump_inds = [i for i, x in enumerate(self.bumps) if x == "True"]
        Pl_inds = self.Pl_inds
        spline_and_anchor_dict["ContGS"] = splines
        if "True" in self.bumps:
            spline_and_anchor_dict["LS_wave_anchors"] = np.zeros(len(self.bumps))
            spline_and_anchor_dict["LS_brightness_anchors"] = np.zeros(len(self.bumps))
            spline_and_anchor_dict["ContLS"] = splines
            spline_and_anchor_dict["GS_wave_anchors"] = np.zeros(len(self.bumps) - len(bump_inds))
            spline_and_anchor_dict["GS_brightness_anchors"] = np.zeros(len(self.bumps) - len(bump_inds))
        else:
            spline_and_anchor_dict["GS_wave_anchors"] = np.zeros(len(self.bumps))
            spline_and_anchor_dict["GS_brightness_anchors"] = np.zeros(len(self.bumps))
        if len(Pl_inds) > 0:
            spline_and_anchor_dict["PL_wave_anchors"] = np.zeros(len(Pl_inds))
            spline_and_anchor_dict["PL_brightness_anchors"] = np.zeros(len(Pl_inds))
            spline_and_anchor_dict["ContPL"] = splines
        return(spline_and_anchor_dict)



class Continua():
    """
    A class for plotting and making continua files
    """

    def __init__(self, spectral_cube, directory_ipac, wavelengths, spectral_cube_unc=None, length = None):
        """
        The constructor for Continua

        Parameters
        ----------
        spectral_cube [array-like]: the array with the spectral cube

        spectral_cube_unc [array-like or NoneType]: the array with the uncertainty cube (or None if no
                                              uncertainties exist)

        directory_ipac [str]: the directory of the ipac file with the anchor info

        wavelengths [array-like]: the array with the wavelengths

        length [int or NoneType]: How many consecutive continuum points need to be above the data
                                  to be an issue. If None, no continua overshoot correcting occurs.

        """

        self.wavelengths = wavelengths
        self.spectral_cube = spectral_cube
        self.spectral_cube_unc = spectral_cube_unc
        
        self.anchors_and_splines = Anchor_points_and_splines(spectral_cube, directory_ipac, wavelengths, length = length)

        self.moments = self.anchors_and_splines.moments
        self.starting_waves = self.anchors_and_splines.starting_waves
        self.Pl_inds = self.anchors_and_splines.Pl_inds
        self.bumps = self.anchors_and_splines.bumps
        
        

    def make_fits_file(self, array, save_loc):
        """
        Takes a 3 dimensional np array and turns it into a .fits file.

        Parameters
        ----------
        array [np.array]: a 3D np.array

        save_loc [str]: the directory where the saved data is stored (e.g. r"path/file_name.fits")

        Returns
        -------
        A fits file of a cube

        """
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(data = array))
        # hdul.writeto(save_loc, overwrite=True)

    def make_continua(self, per_pix = None):
        """
        Creats a PDF of plots of the continua and/or the continua fits files

        Parameters
        ----------
        x_min_pix, x_max_pix, y_min_pix, y_max_pix [int or Nonetype]: the range of the pixels to include
                                                                      if None, the shape of the cube
                                                                      will be used (i.e. mins = 0,
                                                                      y_max_pix = cube.shape[1],
                                                                      x_max_pix = cube.shape[2])

        fits [bool]: True or False - whether or not fits files should be created

        plot [bool]: True or False - whether or not a pdf of continua plots should be created

        per_pix [int or NoneType]: the nth pixels to plot (e.g. per_pix = 6 means that every 6th pixel
                                                           is graphed, provided that the pixel's brightnesses
                                                           aren't all 0. Pixels with all 0 values are skipped)
                                   if None and plot = True, all continua will be graphed

        max_y_plot, min_y_plot, max_x_plot, min_x_plot [int, float, str, NoneType]: the desired plt.lim values for
                                                                                    the plots
                                                                                    max_y = 'median'is the only
                                                                                    excepted str
                                                                                    max_y = 'median'is recommended
                                                                                    when high spectral resolution
                                                                                    results in many features
                                                                                    atop PAH features (like with
                                                                                    JWST data)

        Returns
        -------
        continuum cube

        """
        data = self.spectral_cube
        x_min_pix = 0
        y_min_pix = 0
        x_max_pix = data.shape[2]
        y_max_pix = data.shape[1]

        ContGS_cube = np.zeros(data.shape)
        if "True" in self.bumps:
            ContLS_cube = np.zeros(data.shape)
        if len(self.Pl_inds) > 0:
            ContPL_cube = np.zeros(data.shape)
            
        pix_dict = {}
        for x in range(x_min_pix, x_max_pix):
            for y in range(y_min_pix, y_max_pix):
                pixel = [x, y]
                if not np.all(data[:, pixel[1], pixel[0]] == 0):
                    splines_and_anchors = self.anchors_and_splines.get_splines_with_anchors(pixel)
                else:
                    splines_and_anchors = self.anchors_and_splines.fake_get_splines_with_anchors()
                pix_dict[str(pixel)] = splines_and_anchors

        num_of_pixels = (x_max_pix - x_min_pix)*(y_max_pix - y_min_pix)
        pix_count = 0
        for x in range(x_min_pix, x_max_pix):
            for y in range(y_min_pix, y_max_pix):
                pix_count = pix_count + 1
                pixel = [x, y]
                ContGS_cube[:, y, x] = pix_dict[str(pixel)]["ContGS"]
                if "True" in self.bumps:
                     ContLS_cube[:, y, x] = pix_dict[str(pixel)]["ContLS"]
                if len(self.Pl_inds) > 0:
                     ContPL_cube[:, y, x] = pix_dict[str(pixel)]["ContPL"]
                if pix_count % 100 == 0:
                    print("Continua " + str(pix_count*100/num_of_pixels) + "% completed")

            pix_count = 0
        
        return ContGS_cube
'''    
cont = PAH.Continua(directory_cube=RebinnedCube.data, 
                directory_cube_unc=None, 
                directory_ipac = 'anchors_dust_all.ipac',
                array_waves = RebinnedCube.wavelengths)
RebinnedCube.cont_dust = cont.make_continua()
'''