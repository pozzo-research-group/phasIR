import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import skimage
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import edge_detection_ver2 as ed

from skimage import io
from skimage.draw import circle
from scipy.signal import find_peaks
from skimage.restoration import denoise_tv_chambolle
from irtemp import centikelvin_to_celsius
from data_encoding import final_result

##############################################################################
##############################################################################
# #####################Sample detection using peaks######################### #
##############################################################################
##############################################################################


# Image equalization
def image_eq(frames):
    '''
    Function to obtained an equalized image using all the frames
    in the video.

    Parameters
    -----------
    frames : List
        List of arrays of frames in the video.

    Returns
    --------
    img_eq: Array
        Equalized image
    '''
    n_frames = len(frames)
    for II in range(n_frames):
        frame = frames[II]
        img_eq = (frame - np.amin(frame))/(np.amax(frame)-np.amin(frame))
        if II == 0:
            img_ave = img_eq
        else:
            img_ave = img_ave + img_eq
    img_average = img_ave/n_frames
    img_eq = (img_ave - np.amin(img_ave))/(np.amax(img_ave)-np.amin(img_ave))
    return img_eq


# Function to obtain sum of pixels over all the rows and columns
def pixel_sum(img_eq):
    '''
    Funtion to determine sum of pixels over all the rows and columns
    to obtain plots with peaks at the sample position in the array.

    Parameters
    -----------
        img_eq : Array
            Equalized image

    Returns
    --------
        column_sum: List
            Sum of pixels over all the columns
        row_sum: List
            Sum of pixels over all the rows
            Also returns plots of column sum and row sum.
    '''

    # Denoising the image
    frame = denoise_tv_chambolle(img_eq)
    rows = frame.shape[0]
    columns = frame.shape[1]
    # Adding all the pixels in each column of the equalized image
    column_sum = []
    for i in range(0, columns):
        column_sum.append(sum(frame[:, i]))
    # Adding all the rows in each column of the equalized image
    row_sum = []
    for j in range(0, rows):
        row_sum.append(sum(frame[j, :]))
    # To convert the troughs to peaks
    column_sum = [x * -1 for x in column_sum]
    # To convert the troughs to peaks
    row_sum = [x * -1 for x in row_sum]
    # Plotting the sum of pixel values in each column
    plt.plot(range(len(column_sum)), column_sum)
    plt.xlabel('Column index')
    plt.ylabel('Sum of pixel values over columns')
    plt.title('Sum of pixel values over columns against column index')
    plt.show()
    # Plotting the sum of pixel values in each row
    plt.plot(range(len(row_sum)), row_sum)
    plt.xlabel('Row index')
    plt.ylabel('Sum of pixel values over rows')
    plt.title('Sum of pixel values over rows against row index')
    plt.show()
    return column_sum, row_sum


# To determine the peak values in the row and column sum and thus sample
# location.
def peak_values(column_sum, row_sum, n_columns, n_rows, freeze_heat):
    '''
    Function to find peaks from the column_sum and row_sum arrays
    and return a dataframe with sample locations.


    Parameters
    -----------
        column_sum: List
            Sum of pixel values over all the columns in the
            image array.
        row_sum: List
            Sum of pixel values over all the rows in the
            image array.
        n_columns: Int
            Number of columns of samples in the image
        n_rows: Int
            Number of rows of samples in the image.
        freeze_heat:
            True or False
                It is true when the wells are at a lower temperature
                when compared to the plate in 'img_eq'

    Returns
    --------
        sample_location: Dataframe
            A dataframe containing sample and plate locations and a plot
            with locations superimposed on the image to be processed.
    '''
    if freeze_heat is True:
        # Converting the troughs to peaks
        column_sum = [x*-1 for x in column_sum]
        row_sum = [x*-1 for x in row_sum]
    # Finding peaks in the column sum array
    all_column_peaks = find_peaks(column_sum, height=(None, 1000), distance=7)
    # Getting column peaks as a list
    column_indices = list(all_column_peaks[0])
    # Getting column peak_heights
    column_heights = list(all_column_peaks[1]['peak_heights'])
    # Picking the highest peak values equal to the number of sample columns
    column_peak_indices = []
    for i in range(n_columns):
        column_peaks = column_heights.index(max(column_heights))
        column_peak_indices.append(column_indices[column_peaks])
        pop_index = column_heights.index(max(column_heights))
        column_heights.pop(pop_index)
        column_indices.pop(pop_index)
    # Sorting column indices in ascending order
    column_peak_indices.sort()
    # Finding peaks in the row sum array
    all_row_peaks = find_peaks(row_sum, height=(None, 1000), distance=7)
    # Getting row peaks as a list
    row_indices = list(all_row_peaks[0])
    # Getting peak heights
    row_heights = list(all_row_peaks[1]['peak_heights'])
    # Picking the highest peak values equal to the number of sample rows
    row_peak_indices = []
    for i in range(n_rows):
        row_peaks = row_heights.index(max(row_heights))
        row_peak_indices.append(row_indices[row_peaks])
        pop_index = row_heights.index(max(row_heights))
        row_heights.pop(pop_index)
        row_indices.pop(pop_index)
    # Sorting row indices in ascending order
    row_peak_indices.sort()
    return row_peak_indices, column_peak_indices


def locations(row_peak_indices, column_peak_indices, image):
    '''
    Function to get location of all the samples(row and column) and their
    respective plate locations. (Same column but different rows)

    Parameters
    -----------
    row_peak_indices : List
        List containing the location of all the sample rows
    column_peak_indices : List
        List containing the location of all the sample columns.

    Returns
    --------
    sample_location : Dataframe
        Dataframe containing sample location and plate location.
    '''
    # Appending row and column peak values to arrays to get plate location.
    row = []
    column = []
    plate_location = []
    n_columns = len(column_peak_indices)
    n_rows = len(row_peak_indices)
    for i in range(0, n_columns):
        for j in range(0, n_rows):
            row.append(row_peak_indices[j])
            column.append(column_peak_indices[i])
            if j == 0:
                # Calculating plate location for each well
                #  based on the well location.
                plate_location.append(int((row[j]-0)/2))
            else:
                plate_location.append(int((row[j] + row[j-1])/2))
    # Dataframe containing the well location(row and column)
    sample_location = pd.DataFrame(
        list(zip(row, column, plate_location)),
        columns=['Row', 'Column', 'plate_location'])
    plt.imshow(image)
    plt.scatter(sample_location['Column'],
                sample_location['Row'], s=4, color='Red')
    plt.scatter(sample_location['Column'],
                sample_location['plate_location'], s=4, color='Purple')
    plt.title('Sample and plate location at which the temperature' +
              ' profile is monitored')
    plt.show()
    return sample_location


# To determine the sample and plate temperature using peak locations.
def pixel_intensity(sample_location, frames, x_name, y_name, plate_name):
    '''
    Function to find pixel intensity at all sample locations
    and plate locations in each frame.

    Parameters
    -----------
    sample_location : Dataframe
        A dataframe containing sample and plate locations.
    frames : Array
        An array of arrays containing all the frames of a video.
    x_name : String
        Name of the column in sample_location containing the row values
        of the samples.
    y_name : String
        Name of the column in sample_location containing the column values
        of the samples.
    plate_name : String
        Name of the column in sample_location containing the row values
        of the plate location.

    Returns
    --------
    temp : List
        Temperature of all the samples in every frame of the video.
    plate_temp : List
        Temperature of the plate next to every sample in every
        frame of the video.
    '''
    temp = []
    plate_temp = []
    # Row values of the peaks
    row = sample_location[x_name]
    # Column values of the peaks
    col = sample_location[y_name]
    # Row value of the plate location
    p_row = sample_location[plate_name]
    for i in range(len(sample_location)):
        temp_well = []
        plate_well_temp = []
        for frame in frames:
            rr, cc = circle(x[i], y[i], radius=1)
            sample_intensity = np.mean(frame[rr, cc])
            temp_well.append(centikelvin_to_celsius(frame[row[i]][col[i]]))
            plate_well_temp.append(centikelvin_to_celsius(
                frame[p_row[i]][col[i]]))
        temp.append(temp_well)
        plate_temp.append(plate_well_temp)
    return temp, plate_temp


# #### Wrapping Function ##### #
def pixel_temp(frames, n_columns, n_rows, path, freeze_heat):
    '''
    Function to determine the temperature of the samples and plate locations
    by analysing pixel values and finding peaks.

    Parameters
    -----------
    frames: Array
        The frames of a video to be analysed.
    n_columns: Int
        Number of columns of samples in the image
    n_rows: Int
        Number of rows of samples in the image.

    Returns
    --------
    m_df : Dataframe
        A dataframe containing row and column coordinates of each sample
        and its respective inflection point obtained.
    '''
    # flip_frames = ed.flip_frame(frames)
    # Function to obtained an equalized image using all the frames
    # in the video.
    img_eq = image_eq(frames)
    # Funtion to determine sum of pixels over all the rows and columns
    # to obtain plots with peaks at the sample position in the array.
    column_sum, row_sum = pixel_sum(img_eq)
    # Function to find peaks from the column_sum and row_sum arrays
    r_peaks, c_peaks = peak_values(column_sum, row_sum, n_columns,
                                   n_rows, freeze_heat=False)
    # Function to return a dataframe with sample locations and
    # respective plate locations.
    sample_location = locations(r_peaks, c_peaks, img_eq)
    # Function to find pixel intensity at all sample locations
    # and plate locations in each frame.
    temp, plate_temp = pixel_intensity(
        sample_location, frames, x_name='Row', y_name='Column',
        plate_name='plate_location')
    # Function to obtain the peaks in sample temperature profile
    s_peaks, s_infl = ed.peak_detection(temp, plate_temp, 'Sample')
    # Neural network classification
    result_df = final_result(temp, plate_temp, path)
    # # Function to obtain the peaks in plate location temperature profile
    # p_peaks, p_infl = ed.peak_detection(temp, plate_temp,'Plate')
    # # Function to obtain the inflection point(melting point)
    # # from the temperature profile.
    # inf_temp = inflection_point(temp, plate_temp, s_peaks, p_peaks)
    # Dataframe with sample location (row and column coordinates) and
    # respective inflection point.
    m_df = pd.DataFrame({'Row': sample_location.Row,
                         'Column': sample_location.Column,
                         'Melting point': np.asarray(s_infl)[:, 0]})
    return result_df
