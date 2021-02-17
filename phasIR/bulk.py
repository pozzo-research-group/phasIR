import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import musicalrobot
import os

from musicalrobot import edge_detection_ver2 as ed
from musicalrobot import pixel_analysis as pa


def dict_pack(d_files, d_names, d_crop, d_inftemp, d_temp, d_plate):
    """
    Function is used within the auto_crop function to  crop using the inputs
    given by the user.

    Parameters:
    -----------
    d_files : dictonary
        contains the raw file in the requested folder

    d_names : dictonary
        Contains all the names of the files in the requested folder

    d_crop : dictonary
        Contains the cropped versions of the files in the folder

    d_inftemp : dictonary
        Contins all of the inflection points of the files in the folder

    Returns
    --------
    d_all : dictonary
        Nested dictionary of all of the needed dictonary

    """
    d_all = {'d_files': d_files,
             'd_names': d_names,
             'd_crop': d_crop,
             'd_inftemp': d_inftemp,
             'd_temp': d_temp,
             'd_plate': d_plate}

    return d_all


def dict_unpack(d_all):
    """
    Function is used within the auto_crop function to  crop using the inputs
    given by the user.

    Parameters:
    -----------
    d_all : dictonary
        Nested dictionary of all of the needed dictonary

    Returns
    --------
    d_files : dictonary
        contains the raw file in the requested folder

    d_names : dictonary
        Contains all the names of the files in the requested folder

    d_crop : dictonary
        Contains the cropped versions of the files in the folder

    d_inftemp : dictonary
        Contins all of the inflection points of the files in the folder

    """
    d_files = d_all['d_files']
    d_names = d_all['d_names']
    d_crop = d_all['d_crop']
    d_inftemp = d_all['d_inftemp']
    d_temp = d_all['d_temp']
    d_plate = d_all['d_plate']

    return d_files, d_names, d_crop, d_inftemp, d_temp, d_plate


def image_crop(tocrop, top, bottom, left, right):
    """
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
    crop = []

    # checking tocrop is an array and converting if its not
    if isinstance(tocrop, np.ndarray) is True:
        pass
    else:
        tocrop = np.asarray(tocrop)

    frames, height, width = tocrop.shape
    for frame in tocrop:
        crop.append(frame[0 + top: height - bottom, 0 + left: width - right])

    return crop


def plot_image(crop, plotname):
    """
    Plots the given cropped image - used as an internal function

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

    plt.imshow(crop[50])
    plt.colorbar()
    plt.title(plotname)
    plt.show()

    return


def plot_profiles(temp, plate_temp, save_location):
    """

    """

    for i in range(len(temp[:96])):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        # Plotting frame number vs sample temp
        frame_number = np.linspace(1, len(temp[i]), len(temp[i]))
        # plot number 1
        ax[0].scatter(frame_number, temp[i], s=0.5)
        ax[0].set_title('Frame number vs Sample temp:Sample '+str(i+1))
        ax[0].set_xlabel('Frame_number')
        ax[0].set_ylabel('Sample temperature($^{o}$C)')

        # Plotting plate temp vs sample temp
        ax[1].scatter(plate_temp[i], temp[i], s=0.5)
        ax[1].set_title('Plate temp vs Sample temp:Sample '+str(i+1))
        ax[1].set_xlabel('Plate temperature($^{o}$C)')
        ax[1].set_ylabel('Sample temperature($^{o}$C)')

        plt.savefig(save_location + "/Sample " + str(i+1) + ".jpg")

    return


def choose_crop(tocrop, plotname):
    """
    Will ask user to choose if the image will be cropped or not. Will skip the
        specific image

    Allowed inputs are y or n. Any other inputs will result in a re-request

    Parameters:
    -----------
    crop: array
        The array of the tiff file with the requested columns/rows removed

    plotname : string
        Name pulled from the orginal file name - is the chart title

    Returns:
    --------
    crop: array
        The array of the tiff file with the requested columns/rows removed.
        Needs to be returned twice to save to the dictionary and then be
        able to be out of the function for use in next functions.

    """
    crop = []
    out = 0
    plot_image(tocrop, plotname)

    while out == 0:
        crop = tocrop
        decide = input(
            "Do you want to run the crop for this video? Options are y/n: ")
        if decide == 'y':
            crop = auto_crop(tocrop, plotname)
            out = 1
            break
        elif decide == 'n':
            out = 1
        else:
            print("please type either y or n")
            out = 0

    return crop, crop


def auto_crop(tocrop, plotname):
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
    # intro constants
    TotalChange = 1
    left = 0
    right = 0
    top = 0
    bottom = 0

    # User inputs - plot will show between each iteration
    # and will show updates with inputs
    while TotalChange != 0:
        crop = image_crop(tocrop, top, bottom, left, right)
        plot_image(crop, plotname)

        TotalChange = 0
        change = int(input("Enter the change you want for LEFT "))
        left = left + int(change)
        TotalChange = TotalChange + abs(change)

        change = int(input("Enter the change you want for RIGHT "))
        right = right + int(change)
        TotalChange = TotalChange + abs(change)

        change = int(input("Enter the change you want for TOP "))
        top = top + int(change)
        TotalChange = TotalChange + abs(change)

        change = int(input("Enter the change you want for BOTTOM "))
        bottom = bottom + int(change)
        TotalChange = TotalChange + abs(change)

    return crop


def inflection_points(crop, plotname, save_location):
    """
    This is a rewrap of the inflection point analysis function using the
    additive rows and columns to find the centriods. All function are the same,
    but the variable names have been changed to match the rest of the bulk
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

    img_eq = pa.image_eq(crop)
    column_sum, row_sum = pa.pixel_sum(img_eq)

    r_peaks, c_peaks = pa.peak_values(column_sum, row_sum, 12, 8, img_eq)
    sample_location = pa.locations(r_peaks, c_peaks, img_eq)

    temp, plate_temp = pa.pixel_intensity(sample_location, crop,
                                          'Row', 'Column', 'plate_location')

    s_peaks, s_infl = ed.peak_detection(temp, plate_temp, 'Sample')
    p_peaks, p_infl = ed.peak_detection(temp, plate_temp, 'Plate')
    inf_temp = ed.inflection_point(temp, plate_temp, s_peaks, p_peaks)

    plot_profiles(temp, plate_temp, save_location)

    return inf_temp, inf_temp, temp, plate_temp


def bulk_crop(cv_file_names, location, d_all):
    """
    Wrapper for all of the bulk cropping functions. Wraps through all of the
        files in the inputed folder, asks for input if the user would like to
        crop the specific function, then asks for inputs for cropping then
        crops the specifed folder in the way requested. Then continues to loop
        through all of the files

    Parameters:
    -----------
    cv_file_names : list
        list of all of the file names in a specified folder, needs to be
        created before running the bulk wrapper

    location : string
        string containing the file location of the desired folder from the
        current location of the workbook

    d_all : dictonary
        Nested dictionary of all of the needed dictonary

    Returns:
    --------
    d_all : dictonary
        Nested dictionary of all of the needed dictonary, Should contain all
        the new additions from the functions
    """
    d_files, d_names, d_crop, d_inftemp, d_temp, d_plate = dict_unpack(d_all)

    for i, file in enumerate(cv_file_names):
        # file input
        d_files['%s' % i] = ed.input_file(location + str(file))

        # create names
        hold_name = cv_file_names[i]
        d_names['%s' % i] = hold_name[:-5]
        plotname = d_names[str(i)]
        keyname = str(i)

        if len(d_crop) == 0:
            tocrop = d_files['%s' % i]
        else:
            if keyname in d_crop:
                tocrop = d_crop['%s' % i]
            else:
                tocrop = d_files['%s' % i]

        # auto crops
        d_crop['%s' % i], crop = choose_crop(tocrop, plotname)

    d_all = dict_pack(d_files, d_names, d_crop, d_inftemp, d_temp, d_plate)

    return d_all


def bulk_analyze(cv_file_names, d_all):
    """
    Wrapper for all of the bulk analysis functions. Wraps through all of the
        files in the inputed folder. Runs analysis functions and then continues
        to loop through all of the files

    Parameters:
    -----------
    cv_file_names : list
        list of all of the file names in a specified folder, needs
        to be created before running the bulk wrapper

    d_all : dictonary
        Nested dictionary of all of the needed dictonary

    Returns:
    --------
    d_all : dictonary
        Nested dictionary of all of the needed dictonary,
        Should contain all the new additions from the functions

    all_inf : dataframe
        a dataframe with all of the sample wells and all of the frames. The
        columns will have the file name and the rows will have the well index

    """
    d_files, d_names, d_crop, d_inftemp, d_temp, d_plate = dict_unpack(d_all)

    all_inf = pd.DataFrame()

    for i, file in enumerate(cv_file_names):
        plotname = d_names[str(i)]
        keyname = str(i)

        save_location = 'temp_profiles/' + plotname
        if not os.path.exists(save_location):
            os.makedirs(save_location)

        crop = d_crop[keyname]
        # save inftemps
        d_inftemp['%s' % i], inf_temp, d_temp, d_plate = inflection_points(
            crop, plotname, save_location)
        # create df output
        all_inf[plotname] = inf_temp

    d_all = dict_pack(d_files, d_names, d_crop, d_inftemp, d_temp, d_plate)

    return d_all, all_inf


def bulk_process(cv_file_names, location, d_all):
    """
    Wrapper for all of the bulk functions. Runs the bulk cropper followed by
    the bulk analyzer.

    Parameters:
    -----------
    cv_file_names : list
        list of all of the file names in a specified folder, needs
        to be created before running the bulk wrapper

    location : string
        string containing the file location of the desired folder from the
        current location of the workbook

    d_all : dictonary
        Nested dictionary of all of the needed dictonary, should either be
        empty or from a previous run when an update is needed

    Returns:
    --------
    d_all : dictonary
        Nested dictionary of all of the needed dictonaries
        Should contain all the new additions from the functions

    all_inf : dataframe
        a dataframe with all of the sample wells and all of the frames. The
        columns will have the file name and the rows will have the well index

    """
    d_crop, d_names, d_all = bulk_crop(cv_file_names, location, d_all)

    d_inftemp, all_inf, d_all = bulk_analyze(cv_file_names, d_all)

    return d_all, all_inf
