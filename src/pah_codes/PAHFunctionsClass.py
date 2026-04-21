#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:40:43 2024
Modified exrensively on Sat Sep 27 10:27:00 2025 over spain

@author: nclark

"""

'''
CHANGES TO APPLY
'''

# updated spectra_stitcher to deal with pupd at boundary
# simplified spectra_stitcher, will need to test on ppahs
# removed old spline code
# removed calculate R, moved to functions
# simplified regions import

# renaming methods: 
# nan_replacer -> replace_nan
# emission_line_remover -> mask_line
# loading_function -> initiate_fits
# loading_function_txt -> initiate_txt
# fov_trim -> trim_fov
# mbb_continuum -> calc_cont_mbb
# spline_continuum -> calc_cont_spline
# linear_continuum -> calc_cont_linear
# pah_feature_integrator -> integrate
# pah_properties -> calc_pah_properties
# error_finder -> calc_error
# spectra_stitcher -> stitch_data
# rand_index_gen -> generate_random_index

# renaming variables:


'''
TO DO
'''

# make error array a dictionary like integrals
# investigate importing Unit and Unit as u

'''
IMPORTING MODULES
'''



# standard stuff
import numpy as np
import random
from scipy.integrate import simpson
import copy

# used for fits file handling
from astropy.io import fits

# needed for fringe remover
import pickle 
from astropy.units import Unit

# needed for unit_changer
import astropy.units as u

# needed for region function
import regions
from astropy.wcs import wcs

# warning supression
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# importing pah_codes
import PAHFunctions as pahf

new = np.newaxis



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
    def initiate_fits(fits_file):
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
    def initiate_txt(fits_file): 
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
        
        # defining overlap list for if stitching occurs
        overlap = []
        
        # determining where fov is based on NaNs, unused here
        fov_mask = 'unused'
        
        return DataCube(fits_file, wavelengths, original_data, data, header, instrument_header, shape, overlap, fov_mask)
        
        
        
    def replace_nan(self, nirspec=False):
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
        
        
        
    def trim_fov(self, region_file):
        """
        replaces noisy edge pixels with nans, using a provided region file generated in ds9.
        does not trim original_data. 
        
        Parameters
        ----------
        region_file : string
            file location of the region file.
        """     
        
        data = self.data
        
        # turning region file into numpy array of same spatial shape as data
        region = regions.Regions.read(region_file, format='ds9')
        fits_cube = fits.open(self.fits_file)
        w = wcs.WCS(fits_cube[1].header).dropaxis(2)
        regmask = region[0].to_pixel(w).to_mask(mode='subpixels', subpixels=1).to_image(shape=self.shape)
        
        # nan instead of 0 outside region
        regmask[regmask == 0] = np.nan
    
        # updating data 
        data = regmask[np.newaxis, :, :]*data
        
        self.data = data


    
    def mask_line(self, wave_list, tight=False):
        """
        Wrapper for line_replacement, intended to remove isolated emission and
        absorption lines. Replaces data argument.

        Parameters
        ----------
        wave_list : numpy.ndarray
            Nx2 array, specifying the beginning and end of each emission line.
        tight : Bool
            if True, uses the specified flux value of the wavelength instead of a median. Intended for 
            tricky regions with dense lines.
        """
        
        wavelengths = self.wavelengths
        data = self.data
        
        new_data = pahf.line_replacement(wavelengths, data, wave_list, tight=tight)
        self.data = new_data

        
        
    def calc_cont_mbb(self, bb, bb_check, temp, amp, gamma):
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
        meancube = pahf.mean_from_wave(wavelengths, data, shape, bb)
        meancube_check = pahf.mean_from_wave(wavelengths, data, shape, bb_check)

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

        mbb_cont = np.ones((len(wavelengths), shape[0], shape[1]))
        best_wavecube = np.ones((size_wave, shape[0], shape[1]))
        best_meancube = np.ones((size_wave, shape[0], shape[1]))

        optimal_vals = np.ones((shape[0], shape[1], 2))
        for i, j in ((i, j) for i in range(shape[0]) for j in range(shape[1])):
            best_wavecube[:, i, j] = wavecube[case_to_use[i, j], :, i, j]
            best_meancube[:, i, j] = meancube[case_to_use[i, j], :, i, j]
            
            optimal_vals[i, j] = T, A = optimal_vals_all_case[case_to_use[i, j], i, j]
            mbb_cont[:, i, j] = pahf.modified_planck_wl(wavelengths*Unit('um'), T*Unit('K'), A, gamma)
        
        self.case_to_use = case_to_use
        self.best_wavecube = best_wavecube
        self.best_meancube = best_meancube
        self.dust_vals = optimal_vals
        self.mbb_cont = mbb_cont
        
        
        
    def calc_cont_spline(self, ap_file_loc, mbb_cont_sub=False):
        """
        wrapper function for spline continuum code.

        Parameters
        ----------
        ap_file_loc : string
            file location of the anchor point info file
       mbb_cont_sub : boolean 
           subtract mbb_cont from data before calculating spline_cont
        """
        
        wavelengths = self.wavelengths
        data = np.copy(self.data)
        
        # subtract mbb_cont if enabled
        if mbb_cont_sub == True:
            data -= self.mbb_cont
            
        # getting ap info from txt file
        ap_file = np.loadtxt(
            ap_file_loc, 
            skiprows=1, 
            unpack=True)
        ap_x, ap_method, ext, ap_lb, ap_ub = ap_file
        ext = ext.astype(np.int64)
        
        spline_cont = pahf.spline_from_anchor_points(
            wavelengths, 
            data, 
            ap_x, 
            ap_method=ap_method, 
            ext=ext, 
            ap_lb=ap_lb, 
            ap_ub=ap_ub)
        
        self.spline_cont = spline_cont
        self.anchor_points = ap_x # note in old version this is a file loc
        

    
    def calc_cont_linear(self, wave_list, tight=False, cont_sub=None):
        """
        Fits linear functions to data to serve as a continuum. Depending on dimensions
        of wave_list, either isolates well-behaved features from the complexes
        they are a part of using linear functions, with areas outside of specified 
        wavelength ranges are NaN. 
        Or, fits a series of linear functions over the entire wavelength range.

        Parameters
        ----------
        wave_list : numpy.ndarray
            Nx2 array, specifying the beginning and end of each linear function.
            Or, Nx1 array, specifying a series of linear continuum anchor points.
        tight : bool
            If True, uses a single y value for continuum instead of a median of surrounding values.
       cont_sub : NoneType
           If mbb, spline are in a string, they will be subtracted from the data.
        """
        
        wavelengths = self.wavelengths
        data = np.copy(self.data)
        
        # determining data to fit linear continuum to
        if cont_sub is not None:
            if 'mbb' in cont_sub:
                data -= self.mbb_cont
            if 'spline' in cont_sub:
                data -= self.spline_cont
        
        # calculating linear cont
        if wave_list.ndim == 2:
            linear_cont = pahf.line_replacement(wavelengths, data, wave_list, tight=tight, nan_for_data=True)
        
        else:
            # note linear_continuum rearranges wave_list and then inserts it into line_replacement
            linear_cont = pahf.linear_continuum(wavelengths, data, wave_list, tight=tight)
        
        self.local_cont = linear_cont
    
    
    
    def integrate( # XXX split up integration and max calc
            self, 
            feature_bounds, 
            feature_name=None,
            max_bounds=None, 
            cont_sub=None, 
            units='MJy', 
            no_neg=False):          
        """
        Converts units and then integrates a feature over the specified wavelength range.
        Also determines the max of this feature.

        Parameters
        ----------
        feature_bounds : tuple of floats
            beginning and ending wavelengths to be integrated over.
       feature_name : NoneType
           name of the feature being integrated, for if multiple are considered in succession.
        max_bounds : NoneType
            beginning and ending wavelengths to consider for max, if not none.
       cont_sub : NoneType
           If mbb, spline, or local are in a string, they will be subtracted from the data specifically.
        units : string
            unit of data that needs to be converted (not sr.) assumed to be MJy or Jy as options.
         no neg : bool
             negative intensities are assumed to be zero and set accordingly.
           
        Returns
        -------
        integral : dict
            A dictionary containing an ndarray as the val, and the feature name as the key.
        """                          
        
        # retrieving relevant wavelength range
        feature_start =  np.argmin(abs(self.wavelengths - feature_bounds[0]))
        feature_end = np.argmin(abs(self.wavelengths - feature_bounds[1]))
    
        wavelengths = self.wavelengths[feature_start : feature_end]
        data = np.copy(self.data[feature_start : feature_end])
        
        # determining specific cont to subtract from data
        # if not none and not a specific string, nothing is subtracted
        if cont_sub is None: # XXX make it do no sub
            data -= self.continuum[feature_start : feature_end]
        else:
            # remove potential \n from end because they are annoying if cont_sub is printed
            if '\n' in cont_sub:
                cont_sub = cont_sub[:-1] # \n counts as 1 character
            # subtract conts from data
            if 'mbb' in cont_sub:
                data -= self.mbb_cont[feature_start : feature_end]
            if 'spline' in cont_sub:
                data -= self.spline_cont[feature_start : feature_end]
            if 'local' in cont_sub:
                # note local_cont may have nans on either side of the feature
                data -= self.local_cont[feature_start : feature_end]
                data[np.isnan(data)] = 0
        
        # calculating max
        max_data = np.copy(data)
        if max_bounds is not None:
            if max_bounds[0] > wavelengths[0] and max_bounds[1] < wavelengths[-1]:
                max_start =  np.argmin(abs(wavelengths - max_bounds[0]))
                max_end = np.argmin(abs(wavelengths - max_bounds[1]))
                max_data = data[max_start : max_end]
            
        max_val = pahf.cube_max(max_data)
    
        # changing units 
        si_cube = np.zeros(data.shape)*(u.W/((u.m**2)*u.micron*u.sr))
        if units == 'MJy':
            jy_cube = (data*10**6)*(u.Jy/u.sr)
        elif units == 'Jy':
            jy_cube = (data)*(u.Jy/u.sr)

        si_cube = jy_cube.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = 
                             u.spectral_density(wavelengths[:, new, new]*u.micron))
        
        # performing numerical integration
        integrand_temp = si_cube.value
        integral = simpson(integrand_temp, x=wavelengths, axis=0)
        
        # interpretation of negative integral
        if no_neg == True and np.any(integral) < 0:
            integral[integral<0] = np.nan
        
        if feature_name is not None: # feature_name will be a string
            # check if feature_integrals dict exists, create it if it doesnt
            if hasattr(self, 'feature_integrals') == True:
                self.feature_integrals[feature_name] = integral
                self.feature_max[feature_name] = max_val
                self.feature_bounds[feature_name] = feature_bounds
                self.feature_cont_type[feature_name] = cont_sub
            else:
                self.feature_integrals = {feature_name : integral}
                self.feature_max = {feature_name : max_val}
                self.feature_bounds = {feature_name : feature_bounds}
                self.feature_cont_type = {feature_name : cont_sub}
                
        else:
            return integral
        
        
        
    def calc_pah_properties(self, ev, include_3p4=False):
        integral_33 = self.feature_integrals['3p3']
        if include_3p4 == True:
            integral_33 += self.feature_integrals['3p4']
            
        integral_77 = self.feature_integrals['7p7']
        integral_112 = self.feature_integrals['11p2']
        self.charge, self.size = pahf.calc_pah_properties(integral_33, integral_77, integral_112, ev)


    
    def calc_error(self, feature_wavelength, feature_extent, error_wave):
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
        array_y, array_x = self.shape
        
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
        resolution = pahf.calculate_miri_R(feature_wavelength)
        delta_wave = feature_wavelength/resolution
        num_points = (feature_extent[1] - feature_extent[0])/delta_wave
        error = rms*delta_wave*(num_points)**0.5
        
        return error
    

    
    def stitch_data(self, DataCube2, offset=None, no_offset=False, boundary_pupd=False, nirspec_to_miri=False):
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
        boundary_pupd : bool
            some lines like at ch3b-3c are right at boundary, and if pupd present need to consider less area for overlap.
            use furthest from overlap indices instead of nearest in this case. 
        nirspec_to_miri : bool
            additional considerations when stitching nirspec and miri spectra together.
        """    
        
        wave_a = self.wavelengths
        wave_b = DataCube2.wavelengths
        
        data_a = self.data
        data_b = DataCube2.data
        
        # find where wavelength overlap begins and ends
        overlap_begin = np.argmin(abs(wave_a - wave_b[0]))
        overlap_end = np.argmin(abs(wave_b - wave_a[-1]))
        
        split_wave = (wave_b[overlap_end] + wave_a[overlap_begin])/2
        lower_ind = np.argmin(abs(wave_a - split_wave))
        upper_ind = np.argmin(abs(wave_b - split_wave))
        
        # verify wavelengths only increase
        while wave_b[upper_ind] <= wave_a[lower_ind]:
            upper_ind += 1
        
        # stitching at lower_ind, upper_ind
        wavelengths = np.hstack((wave_a[:lower_ind], wave_b[upper_ind:]))
        
        # calculating offsets
        if no_offset == True:
            offset = 0
        else:
            if boundary_pupd == True:
                # boundary vals only, meant for starting of wave_b for overlap
                offset_a = np.nanmean(data_a[overlap_begin : overlap_begin + 10], axis=0)
                offset_b = np.nanmean(data_b[:10], axis=0)
            else:
                offset_a = np.nanmean(data_a[overlap_begin:], axis=0)
                offset_b = np.nanmean(data_b[:overlap_end], axis=0)
            
            offset = offset_a - offset_b
    
        # applying offset, stitching at lower_ind, upper_ind
        temp = data_b + offset
        data = np.concatenate((data_a[:lower_ind], temp[upper_ind:]), axis=0)
        
        # also stitch original_data using the same parameters including offset
        original_temp = DataCube2.original_data + offset
        original_data = np.concatenate((self.original_data[:lower_ind], original_temp[upper_ind:]), axis=0)
        
        overlap = lower_ind #(lower_index, upper_index)
        
        # add info to DataCube
        self.wavelengths = wavelengths
        self.original_data = original_data
        self.data = data
        self.overlap.append(overlap)
        
        

    def generate_random_index(self, N, mask=None):
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

        region = regions.Regions.read(region_file, format='ds9')
        fits_cube = fits.open(self.fits_file)
        w = wcs.WCS(fits_cube[1].header).dropaxis(2)

        regmask = region[0].to_pixel(w).to_mask(mode='subpixels', subpixels=1).to_image(shape=self.shape)
    
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
    
