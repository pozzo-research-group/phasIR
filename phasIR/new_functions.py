# import necessary modules
# These functions will need to be split into different modules

from scipy.interpolate import LSQUnivariateSpline
import os
import sys
import numpy as np
import pandas as pd
import random

from skimage import io
from skimage import feature
from skimage.draw import circle
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from skimage import filters
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.interpolate import BSpline
from irtemp import centikelvin_to_celsius
from scipy import signal

import matplotlib.pyplot as plt
from scipy import ndimage


###############################################################################
# Image analysis

def pixel_intensity(sample_location, plate_location, frames, r=2):
    '''
    Function to find pixel intensity at all sample locations
    and plate locations in each frame.

    Parameters
    -----------
    sample_location : Dataframe
        A dataframe containing the coordinates of all samples.
    plate_location : Dataframe
        A dataframe containing the coordinates of the plate to use for each
        sample.
    frames : Array
        An array of arrays containing the frames of the IR video.
    r: Int
       Integer value to use for the radius of the circle applied to each
       sample location. The final sample temperature is obtained as an average
       value.

    Returns
    --------
    sample_temp : List
        A list containig the temperature of all the samples over the
        entire IR video.
    plate_temp : List
        Temperature of the plate next to every sample in every
        frame of the video.
    '''
    sample_temp = []
    plate_temp = []
    # Row values of the peaks
    row = sample_location['Row']
    # Column values of the peaks
    col = sample_location['Column']
    # Row value of the plate location
    p_row = plate_location['Plate_row']
    # Row value of the plate location
    p_col = plate_location['Plate_col']
    for i in range(len(sample_location)):
        temp_well = []
        plate_well_temp = []
        for frame in frames:
            rr, cc = circle(row[i], col[i], radius=r)
            sample_intensity = np.mean(frame[rr, cc])
            plate_intensity = np.mean(frame[p_row[i*4:(i*4)+4],
                                      p_col[i*4:(i*4)+4]])
            temp_well.append(centikelvin_to_celsius(sample_intensity))
            plate_well_temp.append(centikelvin_to_celsius(plate_intensity))
        sample_temp.append(temp_well)
        plate_temp.append(plate_well_temp)
    return sample_temp, plate_temp


def edge_detection(frame, n_samples, method='canny', sigma=1):
    """
    Function to detect the edges of the wells, fill and label them to
    determine their centroids.

    Parameters
    -----------
    frames : Array
        The frames to be processed and determine the
        sample temperature from.
    n_samples : Int
        The number of samples in the input video.
    method : String
        Edge detection algorithm to be used
    sigma : float
        Standard deviation of the Gaussian filter applied during the
        skimage.feature.canny method to detect the wells on the plates.

    Returns
    --------
    samples : Array
        Array containing all samples in the frame stored as skimage objects.
    """

    # Define placeholder variables
    broken = False

    # use canny edge detection method
    if method is 'canny':
        for size in range(15, 9, -1):
            image = frame - frame.min()
            image = image/image.max()

            edges = feature.canny(image, sigma=sigma)

            filled_samples = binary_fill_holes(edges)
            cleaned_samples = \
                remove_small_objects(filled_samples, min_size=size)
            labeled_samples = label(cleaned_samples)
            samples = \
                regionprops(labeled_samples, intensity_image=frame)

            # Check that the number of objects found in the image is not
            # higher than the number of wells.
            # If higher, increase the threshold [thres] even further.
            if len(samples) > n_samples:
                broken = True

            if len(samples) == n_samples:
                break

    elif method is 'sobel':
        for size in range(15, 9, -1):
            # use sobel
            edges = filters.sobel(frame)
            edges = edges > edges.mean() * 2  # booleanize data
            filled_samples = binary_fill_holes(edges)
            cleaned_samples = \
                remove_small_objects(filled_samples, min_size=size)
            labeled_samples = label(cleaned_samples)
            samples = \
                regionprops(labeled_samples, intensity_image=frame)

            # Check that the number of objects found in the image is not
            # higher than the number of wells.
            # If higher, increase the threshold [thres] even further.
            if len(samples) > n_samples:
                broken = True

            if len(samples) == n_samples:
                break

    if broken:
        print('The number of objects found on the image is higher than' +
              ' the sample number provided ', n_samples)

    return samples


def sample_locations(samples, n_samples, dtype='float'):
    """
    Function used to extract the centroid coordinates of each well.

    Paramters
    ---------
    samples: skimage.measure._regionprops.RegionProperties
        skimage object containing the properties of each well found
        using the edge detection function. The centroid of the well
        is an attribute of this object.
    n_samples: int
        Integer number representing the number of samples in a plate.
        This number shuold match the number fo wells

    Returns
    -------
    sample_location : pd.DataFrame
        DataFrame containing the sample centroid's coordinates
        ordered column wise.
    """
    # Extract centroid of the samples.
    sample_col = []
    sample_row = []
    if dtype == 'int':
        for i in range(len(samples)):
            sample_col.append(int(round(samples[i].centroid[1])))
            sample_row.append(int(round(samples[i].centroid[0])))
    elif dtype == 'float':
        for i in range(len(samples)):
            sample_col.append(samples[i].centroid[1])
            sample_row.append(samples[i].centroid[0])

    # Save the sample coordinates and order them
    # Create a unique index to help with the sorting of the samples
    unique_index = random.sample(range(100), n_samples)
    location_dict = {'Column': sample_col, 'Row': sample_row,
                     'unique_index': unique_index}
    # Convert dictionary in a DataFrame and sort it
    sample_location = pd.DataFrame(location_dict)
    sample_location.sort_values(['Column', 'Row'],
                                inplace=True, ignore_index=True)

    return sample_location


def sort_samples(sample_location, n_columns, n_rows):
    '''
    Function to sort the samples location to match the order in which
    the samples are pipetted.

    Parameters
    ------------
    regprops : Dict
        A dictionary of dataframes containing information about the sample.
    n_columns : Int
        Number of columns of samples
    n_rows : Int
        Number of rows of samples

    Returns
    --------
    sorted_location : Dict
        A dataframe with information about samples. The order of the samples
        is sorted from top to bottom and from left to right, following the
        pipetting order.
    '''
    sorted_rows = []
    # Sorting the dataframe according to the row coordinate in each column.
    # The samples are pipetted out top to bottom from left to right.
    # The order of the samples in the dataframe
    # should match the order of pipetting.
    for j in range(0, n_columns):
        df = sample_location[j*n_rows:(j+1)*n_rows].sort_values(['Row'])
        sorted_rows.append(df)
    new_location = pd.concat(sorted_rows)
    # Creating an index to be used for reordering all the dataframes.
    # The unique index is the sum of row and column coordinates.
    reorder_index = new_location.unique_index
    sample_location.set_index('unique_index', inplace=True)
    sorted_location = sample_location.reindex(reorder_index)

    sorted_location = sorted_location.reset_index(drop=True)
    return sorted_location


def diagonal_points(x_coord, y_coord, radius):
    """
    Function to generate four diagonla points given a center and a radius

    Paramters
    ---------
    x_coord: float or int
        X-coordinates of center point
    y_coord: float or int
        Y-coordinates of center point
    radius: int or float
        Distance to add and subtract to the both coordinates of the center
        point to obtain the four diagonal points

    Returns
    -------
    x_coords: list
        List containing the X-coordinates of the diagonal points
    y_coords: list
        List containing the Y-coordinates of the diagonal points
    """

    x_coords = [np.round(x_coord+radius), np.round(x_coord-radius),
                np.round(x_coord+radius), np.round(x_coord-radius)]
    y_coords = [np.round(y_coord+radius), np.round(y_coord+radius),
                np.round(y_coord-radius), np.round(y_coord-radius)]

    return x_coords, y_coords


def plate_location(sample_location, n_columns, n_rows):
    """
    Function to obtain the coordinate of the plate points to use for each
    sample. The plate points are the four diagonal coordinates for each
    well.

    Parameters
    ----------
    sample_location: pd.DataFrame
        Dataframe containing the coordinates of the each sample on the plate
    n_columns: int
        Number representing the columns of wells composng the wellplate
    n_row: int
        Number representing the rows of wells composng the wellplate

    Returns
    -------
    plate_coordinates: pd.DataFrame
        Dataframe containig the coordinate of the plate to use for the
        evaluation of the melting point of each sample.
    """
    plate_x_coords = []
    plate_y_coords = []

    rows = np.array(sample_location['Row'])
    columns = np.array(sample_location['Column'])

    for i in range(len(sample_location)):
        if i >= len(sample_location)-(n_rows+1):
            radius = (columns[i]-columns[i-n_rows])/2
        else:
            radius = (columns[i+n_rows]-columns[i])/2
        diagonal_x, diagonal_y = diagonal_points(columns[i], rows[i], radius)
        plate_x_coords.append(diagonal_x)
        plate_y_coords.append(diagonal_y)

    plate_x = [float(x) for x in np.array(plate_x_coords).reshape(-1, 1)]
    plate_y = [float(y) for y in np.array(plate_y_coords).reshape(-1, 1)]

    plate_coordinates = pd.DataFrame({'Plate_col': plate_x,
                                      'Plate_row': plate_y}, dtype=int)
    return plate_coordinates

###############################################################################
# Find inflection point


def apply_spline(x_data, y_data, n_points=300, knots=8, smooth=3, plot=False):
    """
    Function to apply a spline function to the temperature data to smooth it
    and remove any artifact before finding the melting point of the sample.

    Parameters
    ----------
    x_data: list
        list containg the plate temperature for a single well
    y_data: list
        list containg the sample temperature for a single well
    n_points: int
        Number of points to use for the spline applied to the temperature data.
    knots: int
        Number representing the number of inner knots to use for the spline fit
    smooth: int
        Number representing the smoothness - the order- of the spline fit
    plot: Bool
        True will display the plot of containing the temperature data and the
        spine fit to it

    Returns
    -------
    spline_xdata: array
        Array containing the X-coordinate of the spine fit
    spline_ydata: array
        Array containing the Y-coordinate of the spine fit
    ax: matplotlib.subplots axes object
        matplotlib axes object containig the temperature data and the
        spline fit
    """
    # Create a DataFrame to easily sort the data (based on x_data)
    # and remove any duplicate. This is a requirement for
    # the spline function
    df = pd.DataFrame(data=[x_data, y_data]).T
    df.columns = ['plate_temp', 'sample_temp']
    df = df.sort_values(by='plate_temp', ignore_index=True)
    df2 = df[~df['plate_temp'].duplicated()]

    # Redefine the x and y arrays to use to generate the spline function
    new_x = np.asarray(df2['plate_temp'])
    new_y = np.asarray(df2['sample_temp'])

    # Define the number of interior knots
    t = np.linspace((new_x[1]), (new_x[-2]), knots)
    # Define the new x data to generate the new spline data with
    spline_xdata = np.linspace((new_x[1]), (new_x[-2]), n_points)
    # Calculate the spline using the scipy.interpolate function
    spline_func = LSQUnivariateSpline(new_x, new_y, k=smooth, t=t)
    # Generate the new y-data from the spline just
    spline_ydata = spline_func(spline_xdata)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x_data, y_data, c='blue', label='original data')
    ax.plot(new_x, new_y, c='red', label='ordered data')
    ax.plot(spline_xdata, spline_ydata, c='orange', label='spline data')
    ax.set_xlabel('Plate Temperature [$^{\circ}$C]')  # noqa: W605
    ax.set_ylabel('Sample Temperature [$^{\circ}$C]')  # noqa: W605
    ax.legend()
    if plot:
        return spline_xdata, spline_ydata, ax
    else:
        plt.close()
        return spline_xdata, spline_ydata, ax


def find_inflection(spline_xdata, spline_ydata):
    """
    Function used to fine the inflection point corresponding to the melting
    temperature of the sample.

    Parameters
    ----------
    spline_xdata: array
        Array containing the X-coordinate of the spine fit
    spline_ydata: array
        Array containing the Y-coordinate of the spine fit

    Returns
    -------
    inflection_pointL tuple
        Tuple containg the temperature of the plate [0] and sample [1] at the
        melting point
    """
    # Calculate the 1st derivative of the spline function
    spline_derivative = np.gradient(y_data)

    # First remove any negative value from the derivative list
    # The expected data has an upwards trend, so any negative value
    # Would be coming from a data articaft- most likely form the spline fnc.
    spline_derivative = [point for point in spline_derivative if point > 0]

    # The melting point can be extracted from the inflection point
    # of the spline. Since the data is mostly linera, the second derivative
    # will be zero in those regions as well. For this reason, the second
    # derivative will not be used to identify the inflection point.

    inverse_derivative = 1/np.array(spline_derivative)

    # The inflection point, if present, should be a maximum point of the
    # inverse_derivative list. Use peak finding method to extract the
    # Inflection point

    peaks, properties = find_peaks(inv_der, height=0)
    # Peak heights
    peak_heights = properties['peak_heights']

    if len(peaks) > 1:

        # If there is more than one peak identified, check its index
        # If there are peaks that fall within 10% of the extrema
        # remove them
        n = len(spline_derivative)
        tresh = int(0.1*n)
        new_peaks = [(peaks[i], peak_heights[i]) for i in range(len(peaks))
                     if peaks[i] > tresh and peaks[i] < n-tresh]
        peaks = pd.DataFrame(new_peaks, columns=['indices', 'heights'])

        # add raise error message if new list of peaks is empty!!
    else:
        peaks = pd.DataFrame({'indices': peaks, 'heights': peak_heights})

    inflection_index = peaks['indices'].loc[int(peaks[['heights']].idxmax())]

    inflection_point = (spline_xdata[inflection_index],
                        spline_ydata[inflection_index])

    return inflection_point


def inflection_temperature(x_data, y_data, n_points=300, knots=8,
                           smooth=4, plot=False):
    """
    Function wrapper to find the melting point of the sample data provided.

    Parameters
    ----------
    x_data: list
        list containg the plate temperature for a single well
    y_data: list
        list containg the sample temperature for a single well
    n_points: int
        Number of points to use for the spline applied to the temperature data.
    knots: int
        Number representing the number of inner knots to use for the spline fit
    smooth: int
        Number representing the smoothness - the order- of the spline fit
    plot: Bool
        True will display the plot of containing the temperature data and the
        spine fit to it

    Returns
    -------
    inflection[1]: float
        The sample temperature representing its melting point.

    """

    x_spline, y_spline, ax = apply_spline(x_data, y_data, n_points=n_points,
                                          knots=knots, smooth=smooth)
    inflection = find_inflection(x_spline, y_spline)
    if plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(x_data, y_data, c='darkblue', label='original data')
        ax.plot(x_spline, y_spline, c='orangered', label='spline data')
        ax.scatter(inflection[0], inflection[1], c='k', s=25, marker='X',
                   label='Inflection Point', zorder=10)
        ax.set_xlabel('Plate Temperature [$^{\circ}$C]')  # noqa: W605
        ax.set_ylabel('Sample Temperature [$^{\circ}$C]')  # noqa: W605
        ax.legend()

    return np.round(inflection[1], 1)
