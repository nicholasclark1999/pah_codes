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
from scipy.integrate import simpson
from scipy.signal import savgol_filter

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
from astropy.units import Unit

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

import pah_codes.PAHFunctions as pahf

new = np.newaxis

'''
TO DO
'''

# TODO

# move png_combine to regular functiosn 

# rename static methods to initiate like in psf matching code

# make error array a dictionary like integrals

# rename the 'flux aligner' functions 

# use pickle for saving classes



####################################################################



'''
FUNCTIONS
'''



class DataCube:
    """
    Dedicated class for PAH analysis

    Parameters
    ----------
    fits_file : string
        The local file location of the fits file the data cube.
    wavelengths : numpy.ndarray
        wavelength array ascociated with spectrum, implicit units of microns.
    original_data : numpy.ndarray
        The values of the spectrum, in JWST index format (wavelength first), not modified.
    data : numpy.ndarray
        The values of the spectrum, in JWST index format (wavelength first), can have emission lines removed.
    header : dict
        ext 1 JWST header.
    instrument_header : dict
        ext 0 JWST header.
    shape : tuple of ints
        the y and x length corresponding to data, in the JWST format (w,y,x).
    overlap : list of ints
        overlap array populated whenever data stitching occurs.
    fov_mask : numpy.ndarray
        2d array specifying what indices to consider for.
    """
    
    def __init__(self,
                 fits_file,
                 wavelengths,
                 original_data,
                 data,
                 header,
                 instrument_header,
                 shape,
                 overlap,
                 fov_mask):
        
        self.fits_file = fits_file
        self.wavelengths = wavelengths
        self.original_data = original_data
        self.data = data
        self.header = header
        self.instrument_header = instrument_header
        self.shape = shape
        self.overlap = overlap
        self.fov_mask = fov_mask
        
        
        
    @staticmethod
    def loading_function(fits_file): #XXX rename to initiate?
        """
        Static method used to generate a DataCube object from a fits file.

        Parameters
        ----------
        fits_file : string
            The local file location of the fits file of the data cube.

        Returns
        -------
        DataCube : DataCube
            An instance of the DataCube object.
        """
        
        #load in the data
        with fits.open(fits_file) as hdul:
            header = hdul[1].header
            instrument_header = hdul[0].header

            data = hdul[1].data

        original_data = np.copy(data)
        
        
        #wavelength data from header
        number_wavelengths = header["NAXIS3"]
        wavelength_increment = header["CDELT3"]
        wavelength_start = header["CRVAL3"]
        ref_pix = header['CRPIX3'] # reference pixel value, usually 1
        
        wavelengths = (np.arange(number_wavelengths) + ref_pix - 1) * wavelength_increment + wavelength_start
            
        # determining shape from number of dimensions  
        shape = data.shape
        dim = data.ndim
        if dim == 3:
            array_y = shape[1]
            array_x = shape[2]
        else:
            array_y = 1
            array_x = 1
            
        #standardized shape
        shape = np.array([array_y, array_x])
        
        # defining overlap list for if stitching occurs
        overlap = []
        
        # determining where fov is based on NaNs
        fov_mask = np.ones((array_y, array_x))
        where_are_NaNs = np.isnan(data[0]) 
        fov_mask[where_are_NaNs] = 0
        
        return DataCube(fits_file, wavelengths, original_data, data, header, instrument_header, shape, overlap, fov_mask)
    
    
    
    @staticmethod
    def loading_function_txt(fits_file): 
        """
        Static method used to generate a DataCube object from a txt file, intended for greg's 1d spectra 
        that contain wavelengths, flux, errors and segment number.

        Parameters
        ----------
        fits_file : string
            The local file location of the txt file of the data spectra.

        Returns
        -------
        DataCube : DataCube
            An instance of the DataCube object.
        """

        #load in the data
        image_file = np.loadtxt(fits_file, unpack=True)
        
        #header data, unused here
        header = 'unused'
        
        # instrument header, unused here
        instrument_header = 'unused'
        
        #extracting image data
        data = image_file[1]
        data = data[:, np.newaxis, np.newaxis]
        #error_data = fits.getdata(image_file, ext=2)
        original_data = np.copy(data)

        #making wavelength array, in micrometers
        wavelengths = image_file[0]
       
        # determining shape from number of dimensions  
        shape = data.shape
        dim = data.ndim
        if dim == 3:
            array_y = shape[1]
            array_x = shape[2]
        else:
            array_y = 1
            array_x = 1
            
        #standardized shape
        shape = np.array([array_y, array_x])
        
        # defining overlap list for if stitching occurs, unused here
        overlap = 'unused'
        
        # determining where fov is based on NaNs, unused here
        fov_mask = 'unused'
        
        return DataCube(fits_file, wavelengths, original_data, data, header, instrument_header, shape, overlap, fov_mask)
        
        
        
    def nan_replacer(self, nirspec=False):
        """
        replaces the nans of a datacube, while leaving the edge nans intact. Nans are replaced with the 
        data that comes immediately before, assuming that data with a large gap of nans in the data has 
        other issues as well.

        Parameters
        ----------
        nirspec : bool
            Nirspec spectra have a gap, so a range of indices are ignored where the gap occurs.
        """
        
        data = np.copy(self.data)
        
        N = len(data[:,0,0])
        where_are_nans = np.isnan(data) 
        
        if nirspec == True:
            # set a gap in the middle of the band to 0 instead of previous
            for i in range(1690, 2010):
                where_are_nans_nirspec = np.isnan(data[i]) 
                data[i][where_are_nans_nirspec] = 0
                
            for y, x in ((y, x) for y in range(self.shape[0]) for x in range(self.shape[1])):
                if any(where_are_nans[:,y,x]) == True and where_are_nans[0,y,x] == False:
                    for i in range(1, N):
                        if where_are_nans[i,y,x] == True:
                            data[i,y,x] = data[i-1,y,x]

        else:
            for y, x in ((y, x) for y in range(self.shape[0]) for x in range(self.shape[1])):
                if any(where_are_nans[:,y,x]) == True and where_are_nans[0,y,x] == False:
                    for i in range(1, N):
                        if where_are_nans[i,y,x] == True:
                            data[i,y,x] = data[i-1,y,x]
                        
        
        self.data = data


    
    def emission_line_remover(self, wave_list, special=None, cont_mode=False):
        """
        removes a single emission line occupying a specified wavelength range, by 
        replacing the line with a linear function. The slope is calculated by 
        using points below the blue end and above the red end. Creates local_cont argument.

        Parameters
        ----------
        wave_list : numpy.ndarray
            Nx2 array, specifying the beginning and end of each emission line.
        special : NoneType
            if not none, uses the specified flux value of the wavelength instead of a median. Intended for 
            tricky regions with dense lines.
        cont_mode : bool
            reusing the emission line code for building a local continuum instead of an emission line, subtracts
            spline continuum from the data at the same time.
        """
        
        wavelengths = self.wavelengths
        if cont_mode == False:
            data = self.data
        else:
            data = self.data - self.spline_cont
        array_y, array_x = self.shape[0], self.shape[1]

        # determining number of anchor points from wave_list
        N = len(wave_list[:,0])
        
        temp_index_1 = np.ones(N).astype(np.int64)   
        temp_index_2 = np.ones(N).astype(np.int64)   
        
        for i in range(N):
            temp_index_1[i] = np.argmin(abs(wavelengths - wave_list[i,0]))
            temp_index_2[i] = np.argmin(abs(wavelengths - wave_list[i,1]))
        
        # calculating slope y vals, slope
        pah_slope_y_vals_1 = np.ones((N, array_y, array_x))
        pah_slope_y_vals_2 = np.ones((N, array_y, array_x))
        pah_slope = np.ones((N, array_y, array_x))
        
        #need wavelengths[i] to have 2d shape
        wavelengths_cube = np.ones((len(wavelengths), array_y, array_x))
        for i in range(len(wavelengths)):
            wavelengths_cube[i] = wavelengths[i]
        
        for i in range(N):
            # short form wavelength indices    
            w1 = temp_index_1[i]
            w2 = temp_index_2[i]
            
            # special case when medians are dangerous 
            if special != None:
                pah_slope_y_vals_1[i] = data[w1]
                pah_slope_y_vals_2[i] = data[w2]
                
            # edge cases
            elif w1 < 5:
                pah_slope_y_vals_2[i] = np.median(data[w2 : 5 + w2], axis=0)
                pah_slope_y_vals_1[i] = pah_slope_y_vals_2[i]
                
            elif len(wavelengths) - w2 < 5:
                pah_slope_y_vals_1[i] = np.median(data[w1 - 5 : w1], axis=0)  
                pah_slope_y_vals_2[i] = pah_slope_y_vals_1[i]
                
            else:
                pah_slope_y_vals_1[i] = np.median(data[w1 - 5 : w1], axis=0)  
                pah_slope_y_vals_2[i] = np.median(data[w2 : 5 + w2], axis=0)
                
            pah_slope[i] = (pah_slope_y_vals_2[i] - pah_slope_y_vals_1[i])/\
                (wavelengths_cube[w2] - wavelengths_cube[w1])
        
        #putting it all together     
        new_data = np.copy(data)

        for i in range(N):
            # short form wavelength indices    
            w1 = temp_index_1[i]
            w2 = temp_index_2[i]
            
            M = w2 - w1
            for j in range(M+1):
                k = w1 + j
                new_data[k] = pah_slope[i]*(wavelengths_cube[k] - wavelengths_cube[w1]) + pah_slope_y_vals_1[i]
        
        if cont_mode == False:
            self.data = new_data
        else:
            self.local_cont = new_data
        
        
        
    def mbb_continuum(self, bb, bb_check, temp, amp, gamma):
        """
        generates a modified blackbody and performs a grid search to determing the best fit via least squares.
        Considers temperature and amplitude as seperate parameters to be specified, and allows for 
        check arrays where the mbb must not be greater than the data.

        Parameters
        ----------
        bb : numpy.ndarray
            array specifying the beginning and ending wavelengths of the flux that is averaged over for fitting.
        bb_check : numpy.ndarray
            beginning and ending wavelengths of the flux that is averaged over for verifying the fit stays less than the flux.
        temp : numpy.ndarray
            array of temperatures in kelvin to be considered.
        amp : numpy.ndarray
            array of amplitudes to consider, carry the units of the output data.
        gamma : float
            exponent in mbb controlling dust particle size distribution.
        """
        
        wavelengths = self.wavelengths
        data = self.data
        shape = self.shape
        
        size_case = len(bb[:, 0, 0])
        size_wave = len(bb[0, :, 0])
        
        size_wave_check = len(bb_check[0, :, 0])
        
        size_T = len(temp)
        size_A = len(amp)

        # will have a 4d object, index order case, wavelength, temp, amp
        size = (size_case, size_wave, size_T, size_A)
        size_check = (1, size_wave_check, size_T, size_A)
        
        # making input array and test arrays
        waves = (bb[: ,: ,0] + bb[: ,: ,1])/2
        waves_check =  (bb_check[:, :, 0] + bb_check[:, :, 1])/2

        wavecube = waves[:, :, new, new]*np.ones(size) *Unit('um')
        wavecube_check = waves_check[:, :, new, new]*np.ones(size_check) *Unit('um')

        tempcube = temp[new, new, :, new]*np.ones(size)*Unit('K')
        tempcube_check =  temp[new, new, :, new]*np.ones(size_check)*Unit('K')

        ampcube = amp[new, new, new, :]*np.ones(size)
        ampcube_check = amp[new, new, new, :]*np.ones(size_check)

        # calculating corresponding mbb for arrays
        mbb = pahf.modified_planck_wl(wavecube, tempcube, ampcube, gamma)
        mbb_check = pahf.modified_planck_wl(wavecube_check, tempcube_check, ampcube_check, gamma)

        # least squares test, and staying positive check
        meancube = pahf.mean_from_wave(data, wavelengths, shape, bb)
        meancube_check = pahf.mean_from_wave(data, wavelengths, shape, bb_check)

        # need to combine objects of  (case, wave, T, A) and (case, wave, y, x)
        least_squares = np.nansum(((mbb[:, :, new, new, :, :] - meancube[:, :, :, :, new, new])**2), axis=1)

        # need case shape to match least_squares shape
        check = meancube_check[0, :, :, :, new, new] - mbb_check[0, :, new, new, :, :]
        check = np.ones((size_case, size_wave_check, shape[0], shape[1], size_T, size_A))*check[new, :, :, :, :, :]

        # applying check, removing fits that go negative when they should be positive
        for i in range(size_wave_check):
            least_squares[check[:, i, :, :, :, :] < 0] = 1e10

        # finding best fit T and amp (dims 3 and 4)
        minT_ls = np.min(least_squares, axis=3)
        minA = np.argmin(minT_ls, axis=3)

        minA_ls = np.min(least_squares, axis=4)
        minT = np.argmin(minA_ls, axis=3)

        min_least_squares_all_case = np.ones((size_case, shape[0], shape[1]))
        optimal_vals_all_case = np.ones((size_case, shape[0], shape[1], 2))
        for i,j,k in ((i,j,k) for i in range(size_case) for j in range(shape[0]) for k in range(shape[1])): # this is a loop through a generator expression
            a = minT[i, j, k]
            b = minA[i, j, k]
            min_least_squares_all_case[i, j, k] = least_squares[i, j, k, a, b] 
            optimal_vals_all_case[i, j, k, 0] = temp[a]
            optimal_vals_all_case[i, j, k, 1] = amp[b]

        case_to_use = np.argmin(min_least_squares_all_case, axis=0)

        best_mbb = np.ones((len(wavelengths), shape[0], shape[1]))
        best_wavecube = np.ones((size_wave, shape[0], shape[1]))
        best_meancube = np.ones((size_wave, shape[0], shape[1]))

        optimal_vals = np.ones((shape[0], shape[1], 2))
        for i, j in ((i, j) for i in range(shape[0]) for j in range(shape[1])):
            best_wavecube[:, i, j] = wavecube[case_to_use[i, j], :, i, j]
            best_meancube[:, i, j] = meancube[case_to_use[i, j], :, i, j]
            
            optimal_vals[i, j] = T, A = optimal_vals_all_case[case_to_use[i, j], i, j]
            best_mbb[:, i, j] = pahf.modified_planck_wl(wavelengths*Unit('um'), T*Unit('K'), A, gamma)
        
        self.case_to_use = case_to_use
        self.best_wavecube = best_wavecube
        self.best_meancube = best_meancube
        
        self.dust_vals = optimal_vals
        self.best_mbb = best_mbb
        
        
        
    def spline_continuum(self, anchor_point_ipac, all_cont=False):
        """
        wrapper function for the spline continuum code.

        Parameters
        ----------
        anchor_point_ipac : string
            file location of the ipac table containing the anchor points.
       all_cont : boolean 
           for if mbb, spline, and linear continuum are being ran in succession.
        """
        
        wavelengths = self.wavelengths
        data = np.copy(self.data)
        
        if all_cont == True:
            self.spline_cont = pahf.Continua(data - self.best_mbb, anchor_point_ipac, wavelengths).make_continua()
        else:
            self.spline_cont = pahf.Continua(data, anchor_point_ipac, wavelengths).make_continua()
            
        self.anchor_points = anchor_point_ipac
        

    
    def linear_continuum(self, wave_list, tight=None, all_cont=False):
        """
        calculates a series of connected linear functions meant to serve as local continua under well-behaved fatures.

        Parameters
        ----------
        wave_list : numpy.ndarray
            array specifying the wavelengths of the fluxes to be used for constructing the linear functions.
        tight : NoneType
            uses less values in median calculation for when there are small amounts of continuum available.
       all_cont : boolean 
           for if mbb, spline, and linear continuum are being ran in succession.
        """
        wavelengths = self.wavelengths

        if all_cont == True:
            data = np.copy(self.data) - self.spline_cont - self.best_mbb
        else:
            data = np.copy(self.data)
        
        array_y, array_x = self.shape[0], self.shape[1]

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
                
        self.linear_cont = continuum
        
        
        
    def all_continuum(self, 
                      bb, bb_check, temp, amp, gamma,
                      anchor_point_ipac, 
                      wave_list, tight=None):
        """
       wrapper function that runs mbb, spline and linear cont in succession using their inputs.
        """

        self.mbb_continuum(bb, bb_check, temp, amp, gamma)
        self.spline_continuum(anchor_point_ipac, all_cont=True)
        self.linear_continuum(wave_list, tight, all_cont=True)
        self.continuum = self.linear_cont + self.spline_cont + self.best_mbb
    
    
    
    def pah_feature_integrator(
            self, 
            feature_extent, 
            unit='MJy', feature_name=None,
            no_neg=False,
            calc_max=False
            ):          
        """
        converts units and then integrates a feature over the specified wavelength range.

        Parameters
        ----------
        feature_extent : tuple of floats
            beginning and ending wavelengths to be integrated over.
        unit : string
            unit of data that needs to be converted (not sr.) assumed to be MJy or Jy as options.
       feature_name : NoneType
           name of the feature being integrated, for if multiple are considered in succession.
         no neg : bool
             negative intensities are assumed to be zero and set accordingly.
         calc_max : bool
             calculate and store the max value of the feature using the integration bounds
           
        Returns
        -------
        integral : dict
            A dictionary containing an ndarray as the val, and the feature name as the key.
        """                          
        
        # retrieving relevant wavelength range
        feature_start =  np.argmin(abs(self.wavelengths - feature_extent[0]))
        feature_end = np.argmin(abs(self.wavelengths - feature_extent[1]))
    
        wavelengths = self.wavelengths[feature_start : feature_end]
        data = (self.data - self.continuum)[feature_start : feature_end]
        
        # calculating max if enabled
        if calc_max == True:
            array_length_y, array_length_x = self.shape
            max_val = np.zeros((array_length_y, array_length_x))
            where_are_NaNs = np.isnan(data) 
            data[where_are_NaNs] = 0
            
            # finding max index
            for i, j in ((i, j) for i in range(array_length_y) for j in range(array_length_x)):
                if np.max(data[:,i,j]) == 0:
                    max_val[i,j] = np.nan
                else:
                    max_index = np.nanargmax(data[:, i, j])
                    min_range = max([max_index - 10, 0])
                    max_range = min([max_index + 10, len(data[:,i,j])])
                    max_val[i,j] = np.nanmedian(data[min_range : max_range, i, j])
    
        # changing units 
        si_cube = np.zeros(data.shape)*(u.W/((u.m**2)*u.micron*u.sr))
        if unit == 'MJy':
            jy_cube = (data*10**6)*(u.Jy/u.sr)
        elif unit == 'Jy':
            jy_cube = (data)*(u.Jy/u.sr)

        si_cube = jy_cube.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = 
                             u.spectral_density(wavelengths[:, new, new]*u.micron))
        
        # performing numerical integration
        integrand_temp = si_cube.value
        integral = simpson(integrand_temp, x=wavelengths, axis=0)
        
        # interpretation of negative integral
        if no_neg == True and integral < 0:
            integral[integral<0] = np.nan
        
        if feature_name is not None: # feature_name will be a string
            # check if feature_integrals dict exists, create it if it doesnt
            try: 
                test = self.feature_integrals  
            except:
                self.feature_integrals = {feature_name : integral}
            else:
                self.feature_integrals[feature_name] = integral
                
        if calc_max == True:
            try: 
                test = self.feature_max
            except:
                self.feature_max = {feature_name : max_val}
            else:
                self.feature_max[feature_name] = max_val
         
        return integral


    
    def CalculateR(self, wavelength):
        """
        calculates the resolution of MIRI MRS based on the specified wavelength

        Parameters
        ----------
        wavelength : float
            wavelength R is calculated for.
            
        Returns
        -------
        R : float
            the calculated resolution corresponding to the wavelength.
        """     
        
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
            
        return R

    
    
    def error_finder(self, feature_wavelength, feature_extent, error_wave):
        """
        calculates the error of assosiated integrals, based on their RMS.

        Parameters
        ----------
        feature_wavelength : float
            main wavelength assosiated with the integrated feature, used for the resolution calculation.
        feature_extent : tuple of floats
            beginning and ending wavelengths to be integrated over.
       error_wave : float
           wavelength to be used for the RMS calculation, assumed to correspond to continuum.
           
        Returns
        -------
        error : numpy.ndarray
            error array corresponding to the intensity with matching units.
        """     
        error_index = np.argmin(abs(self.wavelengths - error_wave))
        
        wavelengths = self.wavelengths[error_index - 25 : error_index + 25]
        data = (self.data - self.continuum)[error_index - 25 : error_index + 25]
        array_y, array_x = self.shape[0], self.shape[1]
        
        # changing units 
        si_cube = np.zeros(data.shape)*(u.W/((u.m**2)*u.micron*u.sr))
        jy_cube = (data*10**6)*(u.Jy/u.sr)

        # need to convert units one spectra at a time
        for i, j in ((i, j) for i in range(array_y) for j in range(array_x)):
            si_cube[:,i,j] = jy_cube[:,i,j].to(u.W/((u.m**2)*u.micron*u.sr), equivalencies =\
                                                u.spectral_density(wavelengths*u.micron))

        rms_data = si_cube.value
    
        # calculating RMS, turning into error
        rms = (np.var(rms_data, axis=0))**0.5
        resolution = self.CalculateR(feature_wavelength)
        delta_wave = feature_wavelength/resolution
        num_points = (feature_extent[1] - feature_extent[0])/delta_wave
        error = rms*delta_wave*(num_points)**0.5
        
        return error
    

    
    def spectra_stitcher(self, DataCube2, offset=None, no_offset=False, nirspec_to_miri=False):
        """
        This function takes in 2 adjacent wavelength and image data arrays, presumably 
        from the same part of the image fov (field of view), so they correspond to 
        the same location in the sky. It then finds which indices in the lower wavelength 
        data overlap with the beginning of the higher wavelength data, and combines 
        the 2 data arrays in the middle of this region.
        
        It needs to work with arrays that may have different intervals, so it is split into 2
        to take a longer running but more careful approach if needed.

        Parameters
        ----------
        DataCube2 : DataCube
            instance to be stitched into self DataCube.
        offset : NoneType
            uses specified offset instead of calculating one.
        no_offset : bool
            no offsets applied when True.
        nirspec_to_miri : bool
            additional considerations when stitching nirspec and miri spectra together.
        """    
        
        wave_a = self.wavelengths
        wave_b = DataCube2.wavelengths
        
        data_a = self.data
        data_b = DataCube2.data
        
        #check if wavelength interval is the same or different
        check_a = np.round(wave_a[-1] - wave_a[-2], 4)
        check_b = np.round(wave_b[1] - wave_b[0], 4)
        
        def apply_offset(data1, data2, index1, index2, val):
            offset1 = np.nanmean(data1[index1 - val : index1 + val], axis=0)
            offset2 = np.nanmean(data2[index2 - val : index2 + val], axis=0)
            
            offset = offset1 - offset2
            
            return offset

        if check_a == check_b:
        
            #check where the overlap is
            overlap = np.argmin(abs(wave_a - wave_b[0]))
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

        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        if no_offset == True:
            offset = 0
        elif nirspec_to_miri == True:
            offset = apply_offset(data_a, data_b, lower_index, upper_index, 50)
        else:
            offset = apply_offset(data_a, data_b, lower_index, upper_index, 10)
                
        temp = data_b + offset
        data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        
        # also stitch original_data using the same parameters including offset
        original_temp = DataCube2.original_data + offset
        original_data = np.concatenate((self.original_data[:lower_index], original_temp[upper_index:]), axis=0)
        
        if wavelengths[lower_index-1] >= wavelengths[lower_index]:
            wavelengths = np.delete(wavelengths, lower_index, axis=0)
            data = np.delete(data, lower_index, axis=0)
            original_data = np.delete(original_data, lower_index, axis=0)
        
        overlap = lower_index #(lower_index, upper_index)
        
        # add info to DataCube
        self.wavelengths = wavelengths
        self.original_data = original_data
        self.data = data
        self.overlap.append(overlap)

        

    def rand_index_gen(self, N, mask=None):
        """
        generates N random indices to use for plotting, without repeats.

        Parameters
        ----------
        N : int
            How many random indices to calculate. Since repeats are avoided, N should be larger than
            the number of possible indices to prevent infinite loops.
        mask : NoneType
            when an array is specified as a mask, random indices will need to be in the mask region to be counted.
        """    
        
        if mask is None:
            mask = self.fov_mask
            
        random.seed(69420)
        
        rand_y_list = []
        rand_x_list = []
        
        rand_duplicates = {'hello'}
        
        i = 0
        while i < N:
            rand_y = random.randint(0, self.shape[0] - 1)
            rand_x = random.randint(0, self.shape[1] - 1)
            if mask[rand_y, rand_x] != 0:
                temp = f'{rand_y}, {rand_x}'
                if temp not in rand_duplicates:
                    rand_y_list.append(rand_y)
                    rand_x_list.append(rand_x)
                    rand_duplicates.add(temp)
                    i += 1
            
        self.rand_index = [np.array(rand_y_list), np.array(rand_x_list)]
            

    
    def region_to_mask(self, region_file):
        """
        converts region files generated in ds9 to numpy arrays.

        Parameters
        ----------
        region_file : string
            file location of the region file.
           
        Returns
        -------
        regmask : numpy.ndarray
            array corresponding to region. 1 inside, and 0 outside region.
        """     

        reg = regions.Regions.read(region_file, format='ds9')
        fits_cube = fits.open(self.fits_file)
        w = wcs.WCS(fits_cube[1].header).dropaxis(2)

        regmask = reg[0].to_pixel(w).to_mask(mode='subpixels', subpixels=1).to_image(shape=self.shape)
    
        return regmask
    
    
    
    def regrid(self, N, x_start=0, y_start=0):
        """
        Regrids data cube. Can specify starting indices to ensure for example that a 2x2 central source
        becomes 1 pixel and not a quarter of 4 pixels.

        Parameters
        ----------
        N : int
            size of square to regrid to, ex 2x2.
        x_start : int
            first x index to use in regridding.
        y_start : int
            first y index to use in regridding.
        """     
        data = self.data
        original_data = self.original_data
        
        #defining current size with trimmed beginning if specified
        size_y = len(data[0,:,0]) 
        size_x = len(data[0,0,:]) 
        
        reduced_size_y = size_y - y_start
        reduced_size_x = size_x - x_start
        
        #Figure out if any indices need to be discarded, so that the current size will
        #be divisible by N
        remainder_y = (reduced_size_y) % N
        remainder_x = (reduced_size_x) % N
        
        if remainder_y != 0:
            size_y = size_y - remainder_y
            
        if remainder_x != 0:
            size_x = size_x - remainder_x
    
        #building new arrays
        size_wavelength = int(len(data[:,0,0]))
        rebinned_size_y = int(reduced_size_y/N)
        rebinned_size_x =  int(reduced_size_x/N)
        rebinned_data = np.zeros((size_wavelength, rebinned_size_y, rebinned_size_x))
        rebinned_original_data = np.zeros((size_wavelength, rebinned_size_y, rebinned_size_x))
        
        for y in range(y_start, size_y, N):
            for x in range(x_start, size_x, N):
                #note that y:y+N will have y+1,...,y+N, with length N, so want to subtract 1 from these to include y
                
                #taking mean over the pixels to be put in 1 bin
                temp_data = np.mean(data[:, y : y + N, x : x + N], axis=(1,2))
                temp_original_data = np.mean(original_data[:, y : y + N, x : x + N], axis=(1,2))
                
                #adding new pixel to array. y/N and x/N should always be integers, because the remainder was removed above.
                rebinned_data[:, int((y - y_start)/N), int((x - x_start)/N)] = temp_data
                rebinned_original_data[:, int((y - y_start)/N), int((x - x_start)/N)] = temp_original_data
        
        # replacing data and sizes
        self.data = rebinned_data
        self.original_data = rebinned_original_data
        self.shape = ([rebinned_size_y, rebinned_size_x])
        
        # replace rand_index if it exists
        try:
            rand_y, rand_x = self.rand_index
            rebinned_rand_y = (rand_y - y_start)//N
            rebinned_rand_x = (rand_x - x_start)//N
            self.rand_index = [rebinned_rand_y, rebinned_rand_x]
            
        except:
           print('oops') # pass
       
        
       
    def make_template(self, template_loc):
        
        # intended to be used immediately after stitching, and no later. 
        
        # assume for now that template_loc is a collection of indices
        
        # making a new DataCube instance to store the template data
        TemplateCube = copy.deepcopy(self)
        
        # updating shape
        TemplateCube.shape = np.array([1, 1])
        
        # defining for convinience
        data = TemplateCube.data
        original_data = TemplateCube.original_data
        y = template_loc[:, 0]
        x = template_loc[:, 1]
        
        # calculating mean over indices
        N = len(y)
        
        template_data = data[:, y[0], x[0]]
        template_original_data = original_data[:, y[0], x[0]]
        i = 1
        while i < N:
            template_data += data[:, y[i], x[i]]
            template_original_data += original_data[:, y[i], x[i]]
            i += 1
            
        template_data = template_data/N
        template_original_data = template_original_data/N
    
        # making data 3 dimensional for consistency
        template_data = template_data[:, np.newaxis, np.newaxis]
        template_original_data = template_original_data[:, np.newaxis, np.newaxis]
        
        # updating arguments
        TemplateCube.data = template_data
        TemplateCube.original_data = template_original_data
        
        return TemplateCube
    
    
    
    def save(self, file_loc):
        with open(file_loc, 'wb') as file:
            pickle.dump(self.__dict__, file)
            
            
            
    @staticmethod
    def load(file_loc):
        with open(file_loc, 'rb') as file:
            data = pickle.load(file)
            
        class_inst = DataCube('_', '_', '_', '_', '_', '_', '_', '_', '_')
        class_inst.__dict__ = data
        
        return class_inst
    
