#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:40:43 2024
Modified extensively on Sat Sep 27 10:27:00 2025 over spain
Generalized and simplified on Thurs Jan 29 6:03:00 during SOGS meeting

@author: nclark
"""

'''
CHANGES TO APPLY
'''



# apply to ppahs: 
# made cont_type behave like cont_sub in the integration function, allowing for multiple to be specified.


'''
IMPORTING MODULES
'''



# standard stuff
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import  AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
import copy
import pyregion
from scipy.signal import savgol_filter

# warning supression
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# importing functions
import PAHFunctionsClass as PAH
import PAHFunctions as pahf



####################################################################



'''
PLT.PLOT()
'''



# XXX plots data and original data
def line_removal_plotter(
        DataCubeInst,
        indices=None, 
        title_obj='',
        title_extras='', 
        y_units='MJy/sr',
        resolution=100, 
        save_loc='PDFtime/temp/',
        save=True, 
        close=True):
    
    # index logic
    if indices == None:
        y, x = 0, 0
        i = ''
    elif len(indices) == 2:
        y, x = indices
        i = f'{y}, {x}'
    elif len(indices) == 3:
        i, y, x = indices
        i = f'{i}: {y}, {x}'

    else:
        i = 'bad indices'
        y = 0
        x = 0
    
    # defining variables
    wavelengths = DataCubeInst.wavelengths
    data = DataCubeInst.data[:, y, x]
    original_data = DataCubeInst.original_data[:, y, x]
        
    # y lim logic
    lower_bound = 0.9*np.nanmin(data)
    upper_bound = 1.1*np.nanmax(data)
    
    # title logic for nice punctuation
    if title_obj != '':
        title = title_obj + '; ' + i
    else:
        title = i
        
    if i != '':
        title += '; ' + title_extras
    else:
        title += title_extras
    
    # showing plot logic
    if close == True:
        plt.ioff()

    ax = plt.figure(figsize=(18,10)).add_subplot(111)
    plt.title(title)
    
    plt.plot(wavelengths, original_data, linestyle='dotted', color='black')
    plt.plot(wavelengths, data, color='black')
    
    ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
    ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
    ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
    ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel('Wavelength (micron)', fontsize=16)
    plt.ylabel(f'Flux {y_units}', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.xlim(wavelengths[0], wavelengths[-1])
    plt.ylim(lower_bound, upper_bound)
    
    # saving logic
    if save == True:
        save_title = title_obj + i
        plt.savefig(save_loc + save_title + '.png', dpi=resolution, bbox_inches='tight')     

    # showing plot logic
    if close == True:
        plt.close()
        plt.ion()
        
        

# XXX plots original data, data, spline fit. capable of plotting cont sub
def cont_plotter(
        DataCubeInst,
        cont_type,
        bounds,
        indices=None, 
        anchor_points=None, 
        mbb_points=False,
        cont_sub=False, 
        cont_replace=None,
        extra_cont='',
        title_obj='',
        title_extras='', 
        y_units='MJy/sr',
        resolution=100, 
        save_loc='PDFtime/temp/',
        save=True, 
        close=True):
    
    # initial x bound logic
    lower, upper = bounds
    lower_i =  np.argmin(abs(DataCubeInst.wavelengths - lower))
    upper_i = np.argmin(abs(DataCubeInst.wavelengths - upper))
    
    # index logic
    if indices == None:
        y, x = 0, 0
        i = ''
    elif len(indices) == 2:
        y, x = indices
        i = f'{y}, {x}'
    elif len(indices) == 3:
        i, y, x = indices
        i = f'{i}: {y}, {x}'

    else:
        i = 'bad indices'
        y = 0
        x = 0
    
    # defining variables
    wavelengths = DataCubeInst.wavelengths[lower_i : upper_i]
    data = DataCubeInst.data[lower_i : upper_i, y, x]
    
    # cont type
    continuum = 0*wavelengths
    if 'spline' in cont_type:
        continuum += DataCubeInst.spline_cont[lower_i : upper_i, y, x]
    if 'mbb' in cont_type:
        continuum += DataCubeInst.mbb_cont[lower_i : upper_i, y, x]
    if 'local' in cont_type:
        continuum += DataCubeInst.local_cont[lower_i : upper_i, y, x]
    if 'cont' in cont_type:
        continuum += DataCubeInst.continuum[lower_i : upper_i, y, x]
        
    # cont subtraction
    if cont_sub == True:
        data = np.copy(data - continuum)
        continuum = 0*wavelengths
        
    # cont replacement
    if cont_replace == 'local':
        local = DataCubeInst.local_cont[lower_i : upper_i, y, x]
        where_are_nans = np.isnan(local) 
        # replace data only where values arent nan
        for j in range(len(data)):
            if where_are_nans[j] == False:
                data[j] = local[j]
        
    # extra cont
    if extra_cont != '':
        continuum = 0*wavelengths
        if 'spline' in extra_cont:
            continuum += DataCubeInst.spline_cont[lower_i : upper_i, y, x]
        if 'mbb' in extra_cont:
            continuum += DataCubeInst.mbb_cont[lower_i : upper_i, y, x]
        if 'local' in extra_cont:
            continuum += DataCubeInst.local_cont[lower_i : upper_i, y, x]
        if 'cont' in extra_cont:
            continuum += DataCubeInst.continuum[lower_i : upper_i, y, x]
        
    # y lim logic
    if np.nanmax(continuum) == 0:
        lower_bound = 0.9*np.nanmin(data)
        upper_bound = 1.1*np.nanmax(data)
    else:
        lower_bound = 0.9*np.nanmin(continuum)
        upper_bound = 1.1*np.nanmax(continuum)
    
    # title logic for nice punctuation
    if title_obj != '':
        title = title_obj + '; ' + i
    else:
        title = i
        
    if i != '':
        title += '; ' + title_extras
    else:
        title += title_extras
        
    # showing plot logic
    if close == True:
        plt.ioff()

    # making the plot
    ax = plt.figure(figsize=(18,10)).add_subplot(111)
    plt.title(title)
    
    plt.plot(wavelengths, data)
    plt.plot(wavelengths, continuum)
    
    # plotting anchor points if enabled
    if anchor_points is not None:
        for j, line in enumerate(anchor_points):
            plt.plot([line, line], [lower_bound, upper_bound], color='green', linestyle='dashed', alpha=0.5)
    
    # plotting mbb points if enabled
    if mbb_points == True:
        best_wavecube = DataCubeInst.best_wavecube
        best_meancube = DataCubeInst.best_meancube
        plt.scatter(best_wavecube[:, y, x], best_meancube[:, y, x], color='black', zorder=10)
    
    plt.plot(wavelengths, 0*wavelengths, color='black')
    
    ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
    ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
    ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
    ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel('Wavelength (micron)', fontsize=16)
    plt.ylabel(f'Flux {y_units}', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.xlim(wavelengths[0], wavelengths[-1])
    plt.ylim(lower_bound, upper_bound)
    
    # saving logic
    if save == True:
        save_title = title_obj + i + title_extras
        plt.savefig(save_loc + save_title + '.png', dpi=resolution, bbox_inches='tight')     

    # showing plot logic
    if close == True:
        plt.close()
        plt.ion()



# XXX plots multiple scaled original data, data, spline fit. capable of plotting cont sub
def multi_cont_plotter(
        list_DataCubeInst,
        cont_type_list,
        bounds,
        colours,
        skip=None, 
        indices=None, 
        smooth=None, 
        normalize=None,
        norm_to=None,
        anchor_points=None, 
        mbb_points=False,
        cont_sub=False, 
        cont_replace=None, 
        extra_cont='',
        title_obj='',
        title_extras='', 
        y_units='MJy/sr',
        resolution=100, 
        save_loc='PDFtime/temp/',
        save=True, 
        close=True):
    
    # skipping logic
    if skip is None:
        N = len(list_DataCubeInst)
        skip = np.zeros(N)
    
    # index logic
    if indices == None:
        y, x = 0, 0
        i = ''
    elif len(indices) == 2:
        y, x = indices
        i = f'{y}, {x}'
    elif len(indices) == 3:
        i, y, x = indices
        i = f'{i}: {y}, {x}'

    else:
        i = 'bad indices'
        y = 0
        x = 0
    
    # y, x lim logic
    lower_y_list, upper_y_list = [], []
    lower_x_list, upper_x_list = [], []
        
    # title logic for nice punctuation
    if title_obj != '':
        title = title_obj + '; ' + i
    else:
        title = i
        
    if i != '':
        title += '; ' + title_extras
    else:
        title += title_extras
        
    # showing plot logic
    if close == True:
        plt.ioff()

    # making the plot
    ax = plt.figure(figsize=(18,10)).add_subplot(111)
    plt.title(title)
    
    for j, DataCubeInst in enumerate(list_DataCubeInst):
        if skip[j] == 0:
            # initial x bound logic
            lower, upper = bounds[j]
            lower_i =  np.argmin(abs(DataCubeInst.wavelengths - lower))
            upper_i = np.argmin(abs(DataCubeInst.wavelengths - upper))
            
            # defining variables
            wavelengths = DataCubeInst.wavelengths[lower_i : upper_i]
            data = DataCubeInst.data[lower_i : upper_i, y, x]
            
            # cont type
            cont_type = cont_type_list[j]
            continuum = 0*wavelengths
            if 'spline' in cont_type:
                continuum += DataCubeInst.spline_cont[lower_i : upper_i, y, x]
            if 'mbb' in cont_type:
                continuum += DataCubeInst.mbb_cont[lower_i : upper_i, y, x]
            if 'local' in cont_type:
                continuum += DataCubeInst.local_cont[lower_i : upper_i, y, x]
            if 'cont' in cont_type:
                continuum += DataCubeInst.continuum[lower_i : upper_i, y, x]
            
            # cont subtraction
            if cont_sub == True:
                data = np.copy(data - continuum)
                continuum = 0*wavelengths
                
            # cont replacement
            if cont_replace == 'local':
                local = DataCubeInst.local_cont[lower_i : upper_i, y, x]
                where_are_nans = np.isnan(local) 
                # replace data only where values arent nan
                for k in range(len(data)):
                    if where_are_nans[k] == False:
                        data[k] = local[k]
            
            # extra cont
            if extra_cont != '':
                continuum = 0*wavelengths
                if 'spline' in extra_cont:
                    continuum += DataCubeInst.spline_cont[lower_i : upper_i, y, x]
                if 'mbb' in extra_cont:
                    continuum = DataCubeInst.mbb_cont[lower_i : upper_i, y, x]
                if 'local' in extra_cont:
                    continuum += DataCubeInst.local_cont[lower_i : upper_i, y, x]
                if 'cont' in extra_cont:
                    continuum += DataCubeInst.continuum[lower_i : upper_i, y, x]
            
            # smoothing logic
            if smooth is not None:
                if len(smooth) == 2:
                    data = savgol_filter(data, smooth[0], smooth[1])
                else:
                    data = savgol_filter(data, 101, 3) # window size 51, polynomial order 3
            
            # normalization logic
            norm_den, norm_num = 1.0, 1.0
            if normalize is not None:
                if type(normalize) == str:
                    # normalize to specified feature peak
                    norm_den = DataCubeInst.feature_max[normalize][y, x]
                elif type(normalize) == float or type(normalize) == int:
                    # normalize to specified wavelength
                    norm_i = np.argmin(abs(wavelengths - normalize))
                    norm_den = np.nanmedian(data[norm_i - 3 : norm_i + 3])
                elif len(normalize) == 2:
                    norm_lower = np.argmin(abs(wavelengths - normalize[0]))
                    norm_upper = np.argmin(abs(wavelengths - normalize[1]))
                    norm_i = np.argmax(data[norm_lower : norm_upper]) + norm_lower
                    norm_den = np.nanmedian(data[norm_i - 3 : norm_i + 3])
                    
            if norm_to is not None:
                if type(norm_to) == str:
                    # normalize to specified feature peak
                    norm_num = DataCubeInst.feature_max[norm_to][y, x]
                elif type(norm_to) == float or type(norm_to) == int:
                    # normalize to specified wavelength
                    norm_i = np.argmin(abs(wavelengths - norm_to))
                    norm_num = np.nanmedian(data[norm_i - 3 : norm_i + 3])
                elif len(norm_to) == 2:
                    norm_lower = np.argmin(abs(wavelengths - norm_to[0]))
                    norm_upper = np.argmin(abs(wavelengths - norm_to[1]))
                    norm_i = np.argmax(data[norm_lower : norm_upper]) + norm_lower
                    norm_num = np.nanmedian(data[norm_i - 3 : norm_i + 3])
            
            scale = norm_num/norm_den
    
            # y, x lim logic
            if np.argmax(continuum) == 0:
                lower_y_list.append(0.9*np.nanmin(scale*data))
                upper_y_list.append(1.1*np.nanmax(scale*data))
            else:
                lower_y_list.append(0.9*np.nanmin(scale*continuum))
                upper_y_list.append(1.1*np.nanmax(scale*continuum))
            lower_x_list.append(lower)
            upper_x_list.append(upper)
    
            plt.plot(wavelengths, scale*data, color=colours[j])
            plt.plot(wavelengths, scale*continuum, color=colours[j])
            
    # y, x lim logic
    lower_y, upper_y = min(lower_y_list), max(upper_y_list)
    lower_y = max([lower_y, 0])
    lower_x, upper_x = min(lower_x_list), max(upper_x_list)
    
    # plotting anchor points if enabled
    if anchor_points is not None:
        for j, line in enumerate(anchor_points):
            plt.plot([line, line], [lower_y, upper_y], color='green', linestyle='dashed', alpha=0.5)
    
    # plotting mbb points if enabled
    if mbb_points == True:
        best_wavecube = DataCubeInst.best_wavecube
        best_meancube = DataCubeInst.best_meancube
        plt.scatter(best_wavecube[:, y, x], best_meancube[:, y, x], color='black', zorder=10)
    
    plt.plot(wavelengths, 0*wavelengths, color='black')
    
    ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
    ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
    ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
    ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel('Wavelength (micron)', fontsize=16)
    plt.ylabel(f'Flux {y_units}', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.xlim(lower_x, upper_x)
    plt.ylim(lower_y, upper_y)
    
    # saving logic
    if save == True:
        save_title = title_obj + i + title_extras
        plt.savefig(save_loc + save_title + '.png', dpi=resolution, bbox_inches='tight')     

    # showing plot logic
    if close == True:
        plt.close()
        plt.ion()



'''
PLT.IMSHOW()
'''



# XXX plots a single image with options for scatterplots, contours, and regions
def image_plotter(
        image, 
        title, 
        bounds=None, 
        simple_cbar=False,
        scatter=None, 
        contour_list=None,
        region_list=None, 
        xlim=(1, 32),
        ylim=(2, 30),
        resolution=100, 
        save_loc='PDFtime/', 
        save=True, 
        close=True):
    
    # bounds logic
    if bounds is not None:
        lower, upper = bounds
    else:
        lower, upper = np.nanmin(image), np.nanmax(image)
    
    # showing plot logic
    if close == True:
        plt.ioff()

    # making the plot
    ax = plt.figure(figsize=(10,10)).add_subplot(111)
    plt.title(title, fontsize=20)
    
    plt.imshow(image, vmin=lower, vmax=upper, cmap='gist_heat')
    
    # cbar logic
    if simple_cbar==False:
        plt.colorbar(location = "right", fraction=0.042, pad=0.01)
    else:
        cbar = plt.colorbar(location = "right", ticks=[], fraction=0.042, pad=0.01, format='%.2f')
        cbar.ax.yaxis.set_offset_position('left')

    # plotting scatter plot (black) if enabled
    if scatter is not None:
        plt.scatter(scatter[0], scatter[1], color='black', s=plt.rcParams['lines.markersize'] ** 3.2)
    
    # plotting contours if enabled
    if contour_list is not None:
        for contour in contour_list:
            plt.contour(contour['im'], levels=contour['levels'], colors=contour['colour'])
    
    # plotting regions if enabled
    if region_list is not None:
        for q in region_list:
            p = copy.deepcopy(q)
            ax.add_patch(p)

    ax.invert_yaxis()
    ax.tick_params(axis = "y", color = "k", left = False, right = False, direction = "out")
    ax.tick_params(axis = "x", color = "k", bottom = False, top = False,  direction = "out")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    # saving logic
    if save == True:
        plt.savefig(save_loc + title + '.png', dpi=resolution, bbox_inches='tight')     

    # showing plot logic
    if close == True:
        plt.close()
        plt.ion()
        
        

# XXX least squares plotting for mbb fitting case comparison
def ls_plotter(i, y, x, least_squares, least_squares2, minA, minT, minA2, minT2, resolution):
    ax = plt.figure(f'{i}: {y}, {x}', figsize=(18,10)).add_subplot(121)
    plt.title(f'{i}: {y}, {x}, case 1 anchors')
    plt.imshow(np.log10(least_squares), vmin=0.1, vmax=10)
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
    plt.imshow(np.log10(least_squares2), vmin=0.1, vmax=10)
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

    plt.savefig(f'PDFtime/temp/{i}__y{y}_x{x}.png', dpi=resolution)
    plt.show()
    plt.close()