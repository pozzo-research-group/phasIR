import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edge_detection as ed
import pixel_analysis as pa
import bulk as bk

def test_image_crop ():
    """
    Test:
    Function is used within the auto_crop function to  crop using the inputs
    given by the user.

    Parameters:
    -----------
    tocrop : array
        The raw tiff file that is stored in a dictionary and pulled from each
        key using a wrapper. Acts as the base image for the auto_crop

    left : int
        Number of pixels taken off of the left side of the image

    right : int
        Number of pixels taken off of the right side of the image

    top : int
        Number of pixels taken off of the top side of the image

    bottom : int
        Number of pixels taken off of the bottom side of the image

    Returns
    --------
    crop : array
        The array of the tiff file with the requested columns/rows removed

    """
    #set Inputs
    tocrop = ed.input_file('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    left = 3
    right = 5
    top = 3
    bottom = 10

    #running
    #crop = bk.image_crop(tocrop, left, right, top, bottom)

    #asserts
    #assert isinstance(crop, np.ndarray), "output in not an array"

    return

def test_plot_image ():
    """
    Test: Plots the given cropped image - used as an internal function

    Parameters:
    -----------
    crop: array
        The array of the tiff file with the requested columns/rows removed

    plotname : string
        Name pulled from the orginal file name - is the chart title

    Returns:
    --------
    No returns : will print the plot

    """
    tocrop = ed.input_file('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    left = 3
    right = 5
    top = 3
    bottom = 10
    plotname = 'DA_ST_Mid_50_1'

    crop = bk.image_crop(tocrop, left, right, top, bottom)
    bk.plot_image(crop, plotname)

    return


def test_choose_crop ():
    """
    Will ask user to choose if the image will be cropped or not. Will skip the
        specific image

    Allowed inputs are y or n. Any other inputs will result in a re-request

    Parameters:
    -----------
    tocrop : array
        The raw tiff file that is stored in a dictionary and pulled from each
        key using a wrapper. Acts as the base image for the auto_crop

    plotname : string
        Name pulled from the orginal file name - is the chart title

    Returns:
    --------
    crop: array
        The array of the tiff file with the requested columns/rows removed. Needs
        to be returned twice to save to the dictionary and then be able to be
        out of the function for use in next functions.

    """
    #Inputs
    tocrop = ed.input_file('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    plotname = 'DA_ST_Mid_50_1'

    #running
    #crop, crop = bk.choose_crop(tocrop, plotname)

    #asserts

    return

def test_auto_crop ():
    """
    Will request an imput from the user to determine how much of the image to
        crop off the sides, top, and bottom. Will produce a cropped image

    Inputs MUST be numerical. the program will fail if not numerical

    Parameters:
    -----------
    tocrop : array
        The raw tiff file that is stored in a dictionary and pulled from each
        key using a wrapper. Acts as the base image for the auto_crop

    plotname : string
        Name pulled from the orginal file name - is the chart title

    Returns:
    --------
    crop: array
        The array of the tiff file with the requested columns/rows removed
    """
    #inputs
    tocrop = ed.input_file('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    plotname = 'DA_ST_Mid_50_1'

    #Running
    #crop = bk.auto_crop(tocrop, plotname)

    #asserts

    return


def test_inflection_points ():
    """
    This is a rewrap of the inflection point analysis function using the additive
        rows and columns to find the centriods. All function are the same, but
        the variable names have been changed to match the rest of the bulk
        wrapping functions

    IMPORTANT: This function assumes that the sample is being run on a 96 well
        plate. If this is not correct the number of detected wells will be off

    Parameters:
    -----------
    crop: array
        The array of the tiff file with the requested columns/rows removed

    Returns:
    --------
    inf_temp : list
        the inflection points of the wells in the video

    """
    #Inputs
    tocrop = ed.input_file('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    left = 3
    right = 5
    top = 3
    bottom = 10
    plotname = 'DA_ST_Mid_50_1'

    #crop = bk.image_crop(tocrop, left, right, top, bottom)

    #running
    #inf_temp = bk.inflection_points(crop)

    #asserts

    return

# def bulk_crop (cv_file_names):
#     """
#     Wrapper for all of the bulk cropping functions. Wraps through all of the
#         files in the inputed folder, asks for input if the user would like to
#         crop the specific function, then asks for inputs for cropping then
#         crops the specifed folder in the way requested. Then continues to loop
#         through all of the files
#
#     Parameters:
#     -----------
#     cv_file_names : list
#         list of all of the file names in a specified folder, needs
#         to be created before running the bulk wrapper
#
#     Returns:
#     --------
#     d_crop : dictionary
#         A dictionary of all of the information from the raw tiff files for all of
#         the files in the specifed folder
#
#     d_names : dictionary
#         A dictionary of all of the file names from all of the files in the specified
#         folder. Will correlate with the keys in the d_crop dictionary
#
#     """
#
#
#     for i,file in enumerate(cv_file_names):
#         #file input
#         d_files['%s' % i] = ed.input_file('../../MR_Validation/CameraHeight/'+str(file))
#         tocrop = d_files['%s' %i]
#
#         # create names
#         hold_name = cv_file_names[i]
#         d_names['%s' % i] = hold_name[:-5]
#         plotname = d_names[str(i)]
#         keyname = str(i)
#
#         #auto crop
#         d_crop['%s' % i], crop = choose_crop(tocrop, plotname)
#
#     return d_crop, d_names
#
# def bulk_analyze (cv_file_names, d_crop, d_names):
#     """
#     Wrapper for all of the bulk analysis functions. Wraps through all of the
#         files in the inputed folder. Runs analysis functions and then continues to loop
#         through all of the files
#
#     Parameters:
#     -----------
#     cv_file_names : list
#         list of all of the file names in a specified folder, needs
#         to be created before running the bulk wrapper
#
#     d_crop : dictionary
#         A dictionary of all of the information from the raw tiff files for all of
#         the files in the specifed folder
#
#     d_names : dictionary
#         A dictionary of all of the file names from all of the files in the specified
#         folder. Will correlate with the keys in the d_crop dictionary
#
#     Returns:
#     --------
#     d_inftemp : dictionary
#         A dictionary of all the inflection temperatures for each file in the
#         specifed folder
#
#     all_inf : dataframe
#         a dataframe with all of the sample wells and all of the frames. The
#         columns will have the file name and the rows will have the well index
#
#     """
#
#     for i, file in enumerate (cv_file_names):
#         plotname = d_names[str(i)]
#         keyname = str(i)
#
#         crop = d_crop[keyname]
#         #save inftemps
#         d_inftemp['%s' % i], inf_temp = inflection_points(crop)
#         #create df output
#         all_inf[plotname] = inf_temp
#
#     return d_inftemp, all_inf
#
# def bulk_process (cv_file_names):
#     """
#     Wrapper for all of the bulk functions. Runs the bulk cropper followed by the
#         bulk analyzer.
#
#     Parameters:
#     -----------
#     cv_file_names : list
#         list of all of the file names in a specified folder, needs
#         to be created before running the bulk wrapper
#
#     Returns:
#     --------
#     d_crop : dictionary
#         A dictionary of all of the information from the raw tiff files for all of
#         the files in the specifed folder
#
#     d_inftemp : dictionary
#         A dictionary of all the inflection temperatures for each file in the
#         specifed folder
#
#     all_inf : dataframe
#         a dataframe with all of the sample wells and all of the frames. The
#         columns will have the file name and the rows will have the well index
#
#     """
#     d_crop, d_names = bulk_crop(cv_file_names)
#
#     d_inftemp, all_inf = bulk_analyze(cv_file_names, d_crop, d_names)
#
#     return d_crop, d_inftemp, all_inf
