#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:40:43 2024
Modified extensively on Sat Sep 27 10:27:00 2025 over spain

@author: nclark
"""

'''
CHANGES TO APPLY
'''

# added calculate_miri_R, renamed from CalculateR
# removed bethany's continuum code

'''
TO DO
'''

# investigate astropy unit and constant imports and classify better

'''
IMPORTING MODULES
'''



# standard stuff
import numpy as np
import os
from time import time

# spline code
from scipy.interpolate import make_splrep

# saving imagaes as PDFs
# install by > python3 -m pip install --upgrade Pillow  
# ref. https://pillow.readthedocs.io/en/latest/installation.html#basic-installation
from PIL import Image  

#needed for mbb function
from astropy.units import UnitBase, Unit, Quantity
from astropy.constants import h, c, k_B

# warning supression
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)



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
    
    directory = os.listdir(directory_loc)
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
    
    # adding first and last wavelengths to wave_list
    wave_list = np.insert(wave_list, N, wavelengths[M-1])
    wave_list = np.insert(wave_list, 0, wavelengths[0])
    
    # need to turn (N+2)x1 array into (N+1)x2 array with sets of anchor points
    wave_pairs = np.zeros((N+1, 2))
    for i in range(N+1):
        wave_pairs[i] = wave_list[i], wave_list[i+1]
    
    linear_cont = line_replacement(wavelengths, data, wave_pairs, tight=tight)    
    
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
    
    # 7.7 goes from 7.0 to 8.2
    # 11.2 and 11.0 go from 10.5 to 11.6
    
    # log parabolas to compare 11.2/7.7 ratio to
    fits = pah_charge_fit(ratio_size, ev) # (5, y, x)
    ratio_charge = np.copy(ratio_charge[np.newaxis, :, :]*np.ones(fits.shape))
    
    # index corresponds to charge_ratio
    # make 0.00 for neutral, 1.00 for fully cationic
    ionization_frac_options = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    charge_ratio_index = np.argmin(abs(fits - ratio_charge), axis=0)
    return ionization_frac_options[charge_ratio_index]
    


def pah_size(ratio_size, ionization_frac):
    
    # 3.3 goes from 3.1 to 3.5, however no aliphatics included in the calculation!
    # 11.2 and 11.0 go from 10.5 to 11.6
    
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



def calc_pah_properties(integral_33, integral_77, integral_112, ev):
    # calculating ratios, in form of 11.2/3.3, 7.7
    # applying correction to 3.3 paper based on lemmens2023
    ratio_size = integral_112/(integral_33*1.34)
    ratio_charge = integral_112/integral_77

    ionization_frac = pah_charge(ratio_size, ratio_charge, ev)
    NC = pah_size(ratio_size, ionization_frac)
    
    return ionization_frac, NC



def calculate_miri_R(wavelength):
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



'''
SPLINE CONTINUUM CODE
'''



# short function to convert wavelengths to indices
def wave_to_ind(wavelengths, x):
    ind = np.zeros(x.shape).astype(np.int64)
    for i, wave in enumerate(x):
        ind[i] = np.argmin(abs(wavelengths - wave))
    return ind



def anchor_point(
        data, 
        ap_ind, 
        ap_method=None, 
        ext=None, 
        ap_lb_ind=None, 
        ap_ub_ind=None
        ):
    """
    Calculates the y value of a single anchor point.

    Parameters
    ----------
    data : numpy.ndarray
        spectra data cube. assumes that the spectral axis is 0.
    ap_ind : int
        anchor point x index corresponding to the y val to be calculated.
    ap_method : int, optional
        approach to be used for determining the y val.
    ext : int, optional
        extent of bounds to be used in median calculations. if ext=2, median has 5 entires.
    ap_lb_ind : int, optional
        index of lower bound used for some methods.
    ap_ub_ind : int, optional
        index of upper bound used for some methods.
        
    Returns
    -------
    ap_y : int
        y val corresponding to ap_x, together making an anchor point used for spline calculation.
    """    
    # convinience variable
    N = data.shape[0]

    # determining logic of all extra conditions now to avoid excessive nesting
    # trading efficiency for readability of code
    
    # ap_ind cannot be too close to edges
    ec_2 = ap_ind > 9 and ap_ind < N-10
    
    if ext is not None:
        # ap_ind cannot be too close to edges
        ec_ext = ap_ind > ext-1 and ap_ind < N-ext
    else:
        ec_ext = False
        
    if ap_lb_ind is not None and ap_ub_ind is not None:
        # bounds need to be in correct order and not too close to edges
        ec_b = ap_ub_ind > ap_lb_ind and ap_lb_ind > ext-1 and ap_ub_ind < N-ext
    else:
        ec_b = False

    # calculating ap_y
    if ap_method == 1 and ec_ext == True:
        # median using ext
        ap_y = np.nanmedian(data[ap_ind - ext : ap_ind + ext + 1], axis=0)

    elif ap_method == 2 and ec_2 == True:
        # median with 10 on either side of ap_ind, 21 total
        ap_y = np.nanmedian(data[ap_ind - 10 : ap_ind + 11], axis=0)
    
    elif ap_method == 3 and ec_b == True:
        # use bounds to calculate linear function, anchor point on this function.
        d1 = np.nanmedian(data[ap_lb_ind - ext : ap_lb_ind + ext + 1], axis=0)
        d2 = np.nanmedian(data[ap_ub_ind - ext : ap_ub_ind + ext + 1], axis=0)
        # define line in terms of indices
        m = (d2 - d1)/(ap_ub_ind - ap_lb_ind)
        ap_y = m*(ap_ind - ap_lb_ind) + d1
        
    elif ap_method == 4 and ec_b == True: 
        # use bounds to calculate a flat line, anchor point on this line.
        d1 = np.nanmedian(data[ap_lb_ind - ext : ap_lb_ind + ext + 1], axis=0)
        d2 = np.nanmedian(data[ap_ub_ind - ext : ap_ub_ind + ext + 1], axis=0)
        ap_y = (d1 + d2)/2
        
    else:
        # use data[ind] only, unless the conditions are met for something more complex.
        # this is the expected output if ap_method=0, or if the intended args are incorrect.
        ap_y = data[ap_ind]
            
    return ap_y



def spline_from_anchor_points(
        wavelengths, 
        data, 
        ap_x, 
        ap_method=None, 
        ext=None, 
        ap_lb=None, 
        ap_ub=None
        ):
    """
    Turns a list of wavelengths into anchor points, and fits a spline to them.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        wavelengths array. Units should match ap_x, ap_lb, ap_ub.
    data : numpy.ndarray
        spectra data cube. assumes that the spectral axis is 0.
    ap_x : numpy.ndarray
        anchor point wavelengths. All ap are assumed to have the same length.
    ap_method : list of int, optional
        approach to be used for determining the anchor point y val.
    ext : list of int, optional
        extent of bounds to be used in median calculations. if ext=2, median has 5 entires.
    ap_lb : list of int, optional
        wavelengths of lower bounds used for some methods.
    ap_ub : list of int, optional
        wavelengths of upper bounds used for some methods.
        
    Returns
    -------
    spl : numpy.ndarray
        spline fits applied to wavelengths, same shape as data.
    """   
    # shape of data
    shape_y, shape_x = data[0].shape
    M = ap_x.shape[0]
    
    # data needs to have no nans, leave data intact 
    data_nonan = np.copy(data)
    data_nonan[np.isnan(data_nonan)] = 0
    
    # convert ap_x to ap_ind
    ap_ind = wave_to_ind(wavelengths, ap_x)
    
    # need both ap_lb and ap_ub to not be None for their routines to function
    if ap_lb is not None and ap_ub is not None:
        ap_lb_ind = wave_to_ind(wavelengths, ap_lb)
        ap_ub_ind = wave_to_ind(wavelengths, ap_ub)
    else:
        ap_lb_ind = [None]*M
        ap_ub_ind = [None]*M
    
    # make ap_method iterable if it is None
    if ap_method is None:
        ap_method = [None]*M
    
    # calculating the anchor point y vals
    ap_y = np.zeros((M, shape_y, shape_x))
    # each input is a list/ndarray of the same length
    for w, ind in enumerate(ap_ind):
        ap_y[w] = anchor_point(
            data_nonan, 
            ind, 
            ap_method=ap_method[w], 
            ext=ext[w], 
            ap_lb_ind=ap_lb_ind[w], 
            ap_ub_ind=ap_ub_ind[w]
            )
    
    # calculate BSpline instance
    spl_func = make_splrep(ap_x, ap_y)
    spl = spl_func(wavelengths)
    return spl
