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
from os import listdir
import os
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

# rename the 'flux aligner' functions 



####################################################################



'''
FUNCTIONS
'''



def planck_wl(
    x: Quantity, T: Quantity, output_unit: UnitBase | str = Unit("erg / (s cm^2 um sr)")
) -> Quantity:
    # note to self: arg : annotation is useful for type hints, can be used by external programs to infer things
    # but python leaves them alone. in the layout func() -> annotation is specifying the return type hints.
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



def mean_from_wave(wavelengths, data, shape, wave_list):
    """
    Calculates a mean of the wavelength and flux over a specified wavelength array, intended as a convinience
    function for the mbb fitting algorithm.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        wavelength array.
    data : numpy.ndarray
        data cube of fluxes.
    shape : tuple of ints
        y and x dimension lengths.
    wave_list : numpy.ndarray
        beginning and ending wavelengths for the mean to be evaluated over.
    """     
    new = np.newaxis
    N = len(wave_list[:,0,0])
    M = len(wave_list[0,:,0])
    
    # array indices (case, index, upper or lower)
    temp_index = np.argmin(abs(wavelengths[new, :, new, new] - wave_list[:, new, :, :]), axis=1)

    means = np.ones((N, M, shape[0], shape[1]))
    for i, j in ((i, j) for i in range(N) for j in range(M)):
        # short form wavelength indices
        w1 = temp_index[i, j, 0]
        w2 = temp_index[i, j, 1]
        means[i, j] = np.nanmean(data[w1 : w2, :, :], axis=0)
        
    return means
    
    
    
def remove_overlap(datasquare, index):
    """
    Removes the overlap of a spectra that has already been stitched. The second dimensiin in the square
    corresponds to the datatype, this function only permits a single spectrum.

    Parameters
    ----------
    datasquare : numpy.ndarray
        spectra that have already been stitched.
    index : int
        index where the stitching occured.
        
    Returns
    -------
    datasquare_new : numpy.ndarray
        stitched spectra with overlap removed.
    lower_index : int
        index where the overlapping pixels were removed.
    """    
    
    wavelengths = datasquare[0]

    wave_a = wavelengths[:index]
    wave_b = wavelengths[index:]
    
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
    """
    defines a time and returns it, to easily keep track of the length of segments.

    Parameters
    ----------
    msg : string
        message to be returned in the print statement.
    start_time : Nonetype
        if specified, used in returning the time. otherwise, it returns just the time without printing.
        
    Returns
    -------
    current_time : float
        output of the time function.
    """    
    
    current_time = round(time(), 2)
    
    if start_time is not None:
        print(f'{msg} {current_time - start_time} s')
        
    return current_time


  
def png_combine(directory_loc, pdf_name, reso=100, delete=True):
    """
    Regrids data cube. Can specify starting indices to ensure for example that a 2x2 central source
    becomes 1 pixel and not a quarter of 4 pixels.

    Parameters
    ----------
    directory_loc : string
        directory containing pngs to combine.
    pdf_name : string
        name and file loc of combined pdf.
    reso : int
        resolution in dpi of the pdf.
    delete : bool
        whether or not to empty directory_loc after running
    """   
    
    directory = listdir(directory_loc)
    pngs = []
    for file in directory:
        if '.png' in file:
            pngs.append(directory_loc + file)
    
    images = [Image.open(file) for file in pngs]
    alpha_removed = []

    for i in ((i) for i in range(len(images))):
        images[i].load()
        background = Image.new("RGB", images[i].size, (255, 255, 255))
        background.paste(images[i], mask=images[i].split()[3]) # 3 is the alpha channel
        alpha_removed.append(background)
    
    alpha_removed[0].save(
        pdf_name, "PDF" ,resolution=reso, save_all=True, append_images=alpha_removed[1:]
    )
    del(images) # do not want this chilling in memory as i suspect it is big
    
    if delete==True:
        for file in pngs:
            os.remove(file)
            
            

def line_slope(wavelengths, data, wave_bounds, tight=False):
    """
    Determines the slope of a line fit to data between specified wavelengths.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        wavelength array.
    data : numpy.ndarray
        data cube of fluxes.
    wave_bounds : numpy.ndarray
        beginning and ending wavelengths corresponding to the line slope.
    tight : Bool
        if True, uses the specified flux value of the wavelength instead of a median. Intended for 
        tricky regions with dense lines.
    """
    
    # dimensions of data cube
    array_y, array_x = data[0].shape
    t1, t2 = type(wave_bounds[0]), type(wave_bounds[1])
    # indices corresponding to the bounds
    if (t1 == int or t1 == np.int64) and (t2 == int or t2 == np.int64):
        w1, w2 = wave_bounds
    else:
        w1 = np.argmin(abs(wavelengths - wave_bounds[0]))
        w2 = np.argmin(abs(wavelengths - wave_bounds[1]))
    
    # calculating y vals, slope
    y_vals_1 = np.ones((array_y, array_x))
    y_vals_2 = np.ones((array_y, array_x))
    slope = np.ones((array_y, array_x))
    
    # y value logic (median or not)
    if tight==True:
        y_vals_1 = np.copy(data[w1])
        y_vals_2 = np.copy(data[w2])
    # edge cases
    elif w1 < 5:
        y_vals_2 = np.median(data[w2 : 5 + w2], axis=0)
        y_vals_1 = np.copy(y_vals_2)
        
    elif len(wavelengths) - w2 < 5:
        y_vals_1 = np.median(data[w1 - 5 : w1], axis=0)  
        y_vals_2 = np.copy(y_vals_1)
        
    else:
        y_vals_1 = np.median(data[w1 - 5 : w1], axis=0)  
        y_vals_2 = np.median(data[w2 : 5 + w2], axis=0)
    
    # need wavelengths to have 2d shape
    wave1 = wavelengths[w1]*np.ones((array_y, array_x))
    wave2 = wavelengths[w2]*np.ones((array_y, array_x))
    
    slope = (y_vals_2 - y_vals_1)/(wave2 - wave1)
    
    return slope, y_vals_1



def line_replacement(wavelengths, data, wave_list, tight=False, nan_for_data=False):
    """
    Replaces segments of a data cube with linear functions.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        wavelength array.
    data : numpy.ndarray
        data cube of fluxes.
    wave_list : numpy.ndarray
        Nx2 shape, each N is beginning and ending wavelengths corresponding to a line slope.
    tight : Bool
        if True, uses the specified flux value of the wavelength instead of a median. Intended for 
        tricky regions with dense lines, or when noise is negligable.
    nan_for_data : Bool
        if True, uses NaN for data elements that have not been replaced by a line
    """
    
    # dimensions of data cube
    array_y, array_x = data[0].shape

    # determining number of anchor points from wave_list
    N = len(wave_list[:,0])
    
    # turning wavelengths into corresponding indices
    index = np.ones(wave_list.shape).astype(np.int64)     
    for i, j in ((i, j) for i in range(N) for j in range(2)):
        index[i, j] = np.argmin(abs(wavelengths - wave_list[i,j]))
        
    # slope, y_vals_1  arrays
    slope = np.ones((N, array_y, array_x))
    y_vals_1 = np.ones((N, array_y, array_x))
    
    #need wavelengths[i] to have 2d shape
    wavelengths_cube = np.ones((len(wavelengths), array_y, array_x))
    for i, wave in enumerate(wavelengths):
        wavelengths_cube[i] = wave
        
    # determining slopes and y_vals
    for i in range(N):
        slope[i], y_vals_1[i] = line_slope(wavelengths, data, index[i], tight=tight)
    
    # building data array with lines replaced by slopes  
    if nan_for_data == True:
        new_data = np.nan*np.ones(data.shape)
    else:
        new_data = np.copy(data)

    for i in range(N):
        w1, w2 = index[i]
        for j in range(w2 - w1 + 1):
            k = w1 + j
            new_data[k] = slope[i]*(wavelengths_cube[k] - wavelengths_cube[w1]) + y_vals_1[i]
    
    return new_data



def linear_continuum(wavelengths, data, wave_list, tight=None):           
    """
    calculates a series of linear functions spanning the entire spectra, which serve as a continuum.
    
    Parameters
    ----------
    wavelengths : numpy.ndarray
        wavelength array.
    data : numpy.ndarray
        spectra array. only intended to be used with 1d spectra but may work with 3d (untested).
    wave_list : numpy.ndarray
        array of anchor points, assumed to be in the same units as wavelength array.
    tight : Nonetype
        if not nonetype, does not calculate median surrounding anchor points for y values.
        
    Returns
    -------
    continuum : numpy.ndarray
        continuum array. Same shape as data if 3d is used (untested).
    """    
    
    # determining number of anchor points from wave_list, and number of wavelengths
    N = len(wave_list)
    M = len(wavelengths)
    
    # adding 0 and M-1 to wave_list
    wave_list = np.insert(wave_list, N, M-1)
    wave_list = np.insert(wave_list, 0, 0)
    
    # need to turn Nx1 array into Nx2 array with sets of anchor points
    index = np.zeros((N+2, 2)).astype(np.int64)
    for i, wave in enumerate(wave_list):
        index[i] = wave, wave_list[i+1]
    
    linear_cont = line_replacement(wavelengths, data, index, tight=tight)    
    
    return linear_cont



def cube_max(data):
    """
    Also determines the max of each pixel in a data cube.
    
    Parameters
    ----------
    data : numpy.ndarray
       spectra array. only intended to be used with 1d spectra but may work with 3d (untested).
       
    Returns
    -------
    max_val : np.ndarray
        maximum of each value in the data cube
    """     

    # shape of data cube
    M, array_length_y, array_length_x = data.shape
    
    # array to store max vals
    max_val = np.zeros((array_length_y, array_length_x))
    where_are_NaNs = np.isnan(data) 
    data[where_are_NaNs] = 0
        
    # finding max index
    for i, j in ((i, j) for i in range(array_length_y) for j in range(array_length_x)):
        if np.max(data[:,i,j]) == 0:
            max_val[i,j] = np.nan
        else:
            max_index = np.nanargmax(data[:, i, j])
            min_range = max([max_index - 5, 0])
            max_range = min([max_index + 5, M])
            max_val[i,j] = np.nanmedian(data[min_range : max_range, i, j])
            
    return max_val



def pah_charge_fit(x, ev):
    # fit parameters depend on average absorbed photon energy
    ev_options = np.array([6, 8, 10, 12])
    ev_index = np.argmin(abs(ev - ev_options))
    x = np.copy(x[np.newaxis, :, :]*np.ones((5, x.shape[0], x.shape[1])))
    M = x.shape
    
    # defining charge log-base10 parabola ev-dep parameters
    # first index is for ev with 0 for 6ev, 2nd for charge frac with 0 for neutral
    A_ev = np.array([
        [1.87, 0.70, 0.64, 0.16, 0.02], # 6eV
        [2.16, 1.18, 0.78, 0.20, 0.02], # 8eV
        [1.93, 0.83, 0.76, 0.22, 0.03],  # 10eV
        [1.77, 0.74, 0.73, 0.24, 0.04]  # 12eV
        ])
    x0_ev =  np.array([
        [0.99, 0.58, 1.74, 0.48, 0.38], # 6eV
        [1.24, 1.52, 2.42, 0.52, 0.36], # 8eV
        [0.78, 0.53, 2.04, 0.49, 0.40], # 10eV
        [0.54, 0.39, 1.97, 0.49, 0.37]  # 12eV
        ])
    alpha_ev =  np.array([
        [-0.33, -0.56, -0.40, -0.80, -1.61], # 6eV
        [-0.27, -0.30, -0.23, -0.77, -1.59], # 8eV
        [-0.31, -0.49, -0.15, -0.73, -1.65], # 10eV
        [-0.35, -0.60, -0.04, -0.70, -1.67]  # 12eV
        ])
    beta_ev =  np.array([
        [0.05, 0.09, 0.10, 0.13, 0.22], # 6eV
        [0.09, 0.11, 0.14, 0.16, 0.26], # 8eV
        [0.10, 0.13, 0.17, 0.18, 0.34], # 10eV
        [0.12, 0.17, 0.19, 0.23, 0.40]  # 12eV
        ])
    
    # specific parameters for a given ev
    # shape needs to be (5, x[0], x[1]) for everything
    A = A_ev[ev_index][:, np.newaxis, np.newaxis]*np.ones(M)
    x0 = x0_ev[ev_index][:, np.newaxis, np.newaxis]*np.ones(M)
    alpha = alpha_ev[ev_index][:, np.newaxis, np.newaxis]*np.ones(M)
    beta = beta_ev[ev_index][:, np.newaxis, np.newaxis]*np.ones(M)
    
    # calculating set of curves that vary by charge fraction for specified ev
    exponent = -1*alpha -1*beta*np.log10(x/x0)
    return A*(x/x0)**exponent
    
    

def pah_charge(ratio_size, ratio_charge, ev):
    # note: this is a 1d function for simplicity
    
    # log parabolas to compare 11.2/7.7 ratio to
    fits = pah_charge_fit(ratio_size, ev) # (5, y, x)
    ratio_charge = np.copy(ratio_charge[np.newaxis, :, :]*np.ones(fits.shape))
    
    # index corresponds to charge_ratio
    # make 0.00 for neutral, 1.00 for fully cationic
    ionization_frac_options = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    charge_ratio_index = np.argmin(abs(fits - ratio_charge), axis=0)
    return ionization_frac_options[charge_ratio_index]
    


def pah_size(ratio_size, ionization_frac):
    # fit parameters depend on ionization fraction
    ionization_frac_index = (4*ionization_frac).astype(np.int64)
    
    # defining power law charge-dep parameters
    # index charge frac with 0 for neutral
    c0_if = np.array([-2.550, -2.490, -2.430, -2.310, -2.010])
    alpha_if = np.array([0.251, 0.248, 0.246, 0.242, 0.234])
    
    # specific parameters for a given ionization fraction
    c0 = c0_if[ionization_frac_index]
    alpha = alpha_if[ionization_frac_index]
    
    # IR is ionization ratio, of 11.2 and 3.3
    IR = np.log10(ratio_size)
    # now determining size from power law (note IR is a log of ionization ratio)
    # IR = c0 + NC**alpha   original, unaranged for NC form
    # log(NC) = log(IR - c0)/alpha
    exponent = (np.log10(IR - c0))/alpha
    return np.round(10**exponent, 0).astype(np.int64)



def pah_properties(integral_33, integral_77, integral_112, ev):
    # calculating ratios, in form of 11.2/3.3, 7.7
    # applying correction to 3.3 paper based on lemmens2023
    ratio_size = integral_112/(integral_33*1.34)
    ratio_charge = integral_112/integral_77

    ionization_frac = pah_charge(ratio_size, ratio_charge, ev)
    NC = pah_size(ratio_size, ionization_frac)
    
    return ionization_frac, NC
  


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
                # flat line over range of points, puts anchor at this value
                wave = wavelength_anchors[ind]
                wave_min = self.x0_mins[ind]
                wave_max = self.x0_maxes[ind]
                
                lower_ind = np.argmin(abs(self.wavelengths - wave_min))
                upper_ind = np.argmin(abs(self.wavelengths - wave_max))
                list_for_mean = cube[lower_ind : upper_ind, pixel[1], pixel[0]]
                brightness_anchors[ind] = np.nanmean(list_for_mean)
                
                # old stuff from when it found a min over the range
                # brightness_anchor, wavelength_to_change = self.find_min_brightness(wave, wave_min,
                #                                                                    wave_max, pixel)
                # brightness_anchors[ind] = brightness_anchor
                # wavelength_anchors[ind] = wavelength_to_change
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