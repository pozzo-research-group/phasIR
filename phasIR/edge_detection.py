import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import random
import h5py

from skimage import io
from skimage import feature
from skimage.draw import circle
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.signal import filtfilt
from scipy.interpolate import BSpline
from irtemp import centikelvin_to_celsius
import data_encoding as de


# Function to load the input file
def input_file(file_name):
    '''
    To load the imput file as an array.

    Parameters
    -----------
    file_name : String
        Name of the Tiff or HDF5 file to be loaded
        as it is saved on the disk.
        Provide file path if it is not in the same directory as
        the jupyter notebook.

    Returns
    --------
    frames : Array
        In case of a video, returns an array for each frame
        in the video.
        In case of an image, return an array.
    '''
    file_type = file_name[-4:]
    if file_type == 'HDF5':
        file = h5py.File(file_name, 'r')
        frames = []
        for i in range(1, len(file.keys())+1):
            frames.append(file['image'+str(i)])
    elif file_type == 'tiff':
        frames = io.imread(file_name)
    return frames


# Function to flip the frames horizontally and vertically to correct
# for the mirroring during recording.
def flip_frame(frames):
    '''
    To flip all the loaded frames horizontally and vertically
    to correct for the mirroring during recording.

    Parameters
    -----------
    frames : Array
        An array containing an array for each frame
        in the video or just a single array in case of an image.

    Returns
    --------
    flip_frames : Array
        Flipped frames that can be processed to get temperature data.
    '''
    flip_frames = []
    for frame in frames:
        f_frame = np.fliplr(frame)
        flip_frames.append(np.flipud(f_frame))
    return flip_frames


# Function to detect edges, fill and label the samples.
def edge_detection(frames, n_samples):
    '''
    To detect the edges of the wells, fill and label them to
    determine their centroids.

    Parameters
    -----------
    frames : Array
        The frames to be processed and determine the
        sample temperature from.
    n_samples : Int
        The number of samples in the input video.

    Returns
    --------
    labeled_samples : Array
        All the samples in the frame are labeled
        so that they can be used as props to get pixel data.
    '''
    for size in range(15, 9, -1):
        for thres in range(1500, 900, -100):
            edges = feature.canny(frames[0]/thres)
            filled_samples = binary_fill_holes(edges)
            cl_samples = remove_small_objects(filled_samples, min_size=size)
            labeled_samples = label(cl_samples)
            props = regionprops(labeled_samples, intensity_image=frames[0])
            if len(props) == n_samples:
                break
#             if thres == 1000 and len(props) != n_samples:
#                 print('Not all the samples are being recognized with
#                 the set threshold range for size ',size)
        if len(props) == n_samples:
            break
    if size == 10 and thres == 1000 and len(props) != n_samples:
        print('Not all the samples are being recognized with the set \
            minimum size and threshold range')
    return labeled_samples


# Function to determine centroids of all the samples
def regprop(labeled_samples, frames, n_rows, n_columns):
    '''
    Determines the area and centroid of all samples.

    Parameters
    -----------
    labeled_samples: Array
        An array with labeled samples.
    frames : Array
        Original intensity image to determine
        the intensity at sample centroids.
    n_rows: Int
        Number of rows of sample
    n_columns: Int
        Number of columns of sample

    Returns
    --------
    regprops: Dict
        A dictionary of dataframes with information about samples in every
        frame of the video.
    '''
    regprops = {}
    n_samples = n_rows * n_columns
    unique_index = random.sample(range(100), n_samples)
    for i in range(len(frames)):
        props = regionprops(labeled_samples, intensity_image=frames[i])
        # Initializing arrays for all sample properties obtained from regprops.
        row = np.zeros(len(props)).astype(int)
        column = np.zeros(len(props)).astype(int)
        area = np.zeros(len(props))
        radius = np.zeros(len(props))
        perim = np.zeros(len(props))
        intensity = np.zeros(len(props), dtype=np.float64)
        plate = np.zeros(len(props), dtype=np.float64)
        plate_coord = np.zeros(len(props))

        c = 0
        for prop in props:
            row[c] = int(prop.centroid[0])
            column[c] = int(prop.centroid[1])
            # print(y[c])
            area[c] = prop.area
            perim[c] = prop.perimeter
            radius[c] = prop.equivalent_diameter/2
            rr, cc = circle(row[c], column[c], radius=radius[c]/3)
            intensity[c] = np.mean(frames[i][rr, cc])
            plate[c] = frames[i][row[c]][column[c]+int(radius[c])+3]
            plate_coord[c] = column[c]+radius[c]+3
            c = c + 1
        regprops[i] = pd.DataFrame({'Row': row, 'Column': column,
                                    'Plate_temp(cK)': plate,
                                    'Radius': radius,
                                    'Plate_coord': plate_coord,
                                    'Area': area, 'Perim': perim,
                                    'Sample_temp(cK)': intensity,
                                    'unique_index': unique_index},
                                   dtype=np.float64)
        if len(regprops[i]) != n_samples:
            print('Wrong number of samples are being detected in frame %d' % i)
        regprops[i].sort_values(['Column', 'Row'], inplace=True)
    return regprops


def sort_regprops(regprops, n_columns, n_rows):
    '''
    Function to sort the regprops to match the order in which the samples
    are pipetted.

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
    sorted_regprops : Dict
        A dictionary of dataframe with information about samples in every
        frame of the video. The order of the samples is sorted from
        top to bottom and from left to right.
    '''
    sorted_regprops = {}
    # n_samples = n_columns * n_rows
    # After sorting the dataframe according by columns in ascending order.
    sorted_rows = []
    # Sorting the dataframe according to the row coordinate in each column.
    # The samples are pipetted out top to bottom from left to right.
    # The order of the samples in the dataframe
    # should match the order of pipetting.
    for j in range(0, n_columns):
        df = regprops[0][j*n_rows:(j+1)*n_rows].sort_values(['Row'])
        sorted_rows.append(df)
    regprops[0] = pd.concat(sorted_rows)
    # Creating an index to be used for reordering all the dataframes.
    # The unique index is the sum of row and column coordinates.
    reorder_index = regprops[0].unique_index
    for k in range(0, len(regprops)):
        regprops[k].set_index('unique_index', inplace=True)
        sorted_regprops[k] = regprops[k].reindex(reorder_index)
    return sorted_regprops


# Function to obtain temperature of samples and plate temp
def sample_temp(sorted_regprops, frames):
    '''
    Function to concatenate all the obtained temperature data
    from the pixel values into lists.

    Parameters
    ----------
    sorted_regprops : Dict
        The dictionary of sorted dataframes containing temperature data.
    frames : Array
        The array of frames to be processed to obtain temperature data.

    Returns
    -------
    temp : List
        Temperature of all the samples in every frame of the video.
    plate_temp : List
        Temperature of the plate next to every sample in every
        frame of the video.
    '''
    temp = []
    plate_temp = []
    for j in range(len(sorted_regprops[1])):
        temp_well = []
        plate_well_temp = []
        for i in range(len(frames)):
            temp_well.append(centikelvin_to_celsius
                             (list(sorted_regprops[i]['Sample_temp(cK)'])[j]))
            plate_well_temp.append(centikelvin_to_celsius(list
                                   (sorted_regprops[i]['Plate_temp(cK)'])[j]))
        temp.append(temp_well)
        plate_temp.append(plate_well_temp)
    return temp, plate_temp


# # Function to obtain melting point by extracting the inflection point
# def peak_detection(sample_temp, plate_temp, material):
#     '''
#     Function to determine inflection point in the sample temperature
#     profile(melting point)

#     Parameters
#     -----------
#     sample_temp : List
#         Temperature of all the samples in every frame of the video.
#     plate_temp : List
#         Temperature profiles of all the plate locations
#     material : String
#         Can be 'Plate' or 'Sample'

#     Returns
#     --------
#     peaks : List
#         List of two highest peak(inflection points) indices in the
#         given temperature profiles.
#     infl : List
#         List of temperature at inflection points for
#         given temperature profiles.

#     '''
#     infl = []
#     peak_indices = []
#     for i in range(len(sample_temp)):
#         frames = np.linspace(1,len(sample_temp[i]),len(sample_temp[i]))
#         # Fitting a spline to the temperature profile of the samples.
#         if material == 'Plate':
#             bspl = BSpline(frames,plate_temp[i],k=3)
#             # Stacking x and y to calculate gradient.
#             gradient_array = np.column_stack((frames,bspl(frames)))
#         else:
#             f = interp1d(plate_temp[i], sample_temp[i],bounds_error=False)
#             gradient_array = np.column_stack(
#                 (plate_temp[i],f(plate_temp[i])))
#         # Calculating gradient
#         gradient = np.gradient(gradient_array,axis=0)
#         # Calculating derivative
#         derivative = gradient[:,1]/gradient[:,0]
#         # Finding peaks in the derivative plot.
#         peaks, properties = find_peaks(derivative, height=0)
#         # Peak heights
#         peak_heights = properties['peak_heights']
#         a = list(peak_heights)
#         max_height1 = np.max(a)
#         a.remove(max_height1)
#         max_height2 = np.max(a)
#         # Appending the index of the two highest peaks to lists.
#         inf_index1 = list(peak_heights).index(max_height1)
#         inf_index2 = list(peak_heights).index(max_height2)
#         # Appending the frame number in which these peaks occur to a list
#         peak_indices.append([peaks[inf_index1],peaks[inf_index2]])
#         # Appending the temperature at the peaks.
#         if material == 'Plate':
#             infl.append([plate_temp[i][peaks[inf_index1]],
#                         plate_temp[i][peaks[inf_index2]]])
#         else:
#             infl.append([sample_temp[i][peaks[inf_index1]],
#                         sample_temp[i][peaks[inf_index2]]])
#     return peak_indices, infl

# Function to obtain melting point by extracting the inflection point
def peak_detection(sample_temp, plate_temp, material):
    '''
    Function to determine inflection point in the sample temperature
    profile(melting point)

    Parameters
    -----------
    sample_temp : List
        Temperature of all the samples in every frame of the video.
    plate_temp : List
        Temperature profiles of all the plate locations
    material : String
        Can be 'Plate' or 'Sample'

    Returns
    --------
    peaks : List
        List of two highest peak(inflection points) indices in the
        given temperature profiles.
    infl : List
        List of temperature at inflection points for
        given temperature profiles.

    '''
    infl = []
    peak_indices = []
    for i in range(len(sample_temp)):
        # Fitting a spline to the temperature profile of the samples.
        # if material == 'Plate':
        #     bspl = BSpline(frames,plate_temp[i],k=3)
        #     # Stacking x and y to calculate gradient.
        #     gradient_array = np.column_stack((frames,bspl(frames)))
        # else:
        f = interp1d(plate_temp[i], sample_temp[i], bounds_error=False)
        x = np.linspace(min(plate_temp[i]), max(plate_temp[i]),
                        len(plate_temp[i]))
        y = f(x)
        n = 25  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        yy = filtfilt(b, a, y)
        gradient_array = np.column_stack((x, yy))
        # Calculating gradient
        first_gradient = np.gradient(gradient_array, axis=0)
        # Calculating derivative
        derivative = first_gradient[:, 1]/first_gradient[:, 0]
        # Finding peaks in the derivative plot.
        peaks, properties = find_peaks(derivative, height=0)
        # Peak heights
        peak_heights = properties['peak_heights']
        a = list(peak_heights)
        max_height1 = np.max(a)
        a.remove(max_height1)
        max_height2 = np.max(a)
        # Appending the index of the two highest peaks to lists.
        inf_index1 = list(peak_heights).index(max_height1)
        inf_index2 = list(peak_heights).index(max_height2)
        # Appending the frame number in which these peaks occur to a list
        peak_indices.append([peaks[inf_index1], peaks[inf_index2]])
        # Appending the temperature at the peaks.
        if material == 'Plate':
            infl.append([x[peak_indices[i][0]],
                        x[peak_indices[i][1]]])
        else:
            infl.append([yy[peak_indices[i][0]],
                        yy[peak_indices[i][1]]])
    return peak_indices, infl


# def inflection_point(s_temp, p_temp, s_peaks, p_peaks):
#     '''
#     Function to get the inflection point(melting point) for each sample.

#     Parameters
#     -----------
#     s_temp : List
#         Sample temperature profiles
#     p_temp : List
#         Plate location temperature profiles
#     s_peaks : List
#         List of two highest peak(inflection points) indices in the
#         temperature profile of the samples.
#     p_peaks : List
#         List of two highest peak(inflection points) indices in the
#         temperature profile of the plate.

#     Returns
#     --------
#     inf_temp : List
#         List of temperature at inflection points for each sample

#     '''
#     inf_peak = []
#     inf_temp = []
#     for i, peaks in enumerate(s_peaks):
#         for peak in peaks:
#             # Making sure the peak is present only in the sample temp profile
#             if abs(peak - p_peaks[i][0]) >= 3:
#                 inf_peak.append(peak)
#                 break
#             else:
#                 pass
#     # Appending the temperature of the sample at the inflection point
#     for i, temp in enumerate(s_temp):
#         inf_temp.append(temp[inf_peak[i]])
#     return inf_temp


# Wrapping functions
# Wrapping function to get the inflection point
def inflection_temp(frames, n_rows, n_columns, path):
    '''
    Function to obtain sample temperature and plate temperature
    in every frame of the video using edge detection.

    Parameters
    -----------
    frames : List
        An list containing an array for each frame
        in the cropped video or just a single array
        in case of an image.
    n_rows: List
        Number of rows of sample
    n_columns: List
        Number of columns of sample
    path : String
        Path to the location to temporarily store neural
        network input images.

    Returns
    --------
    regprops : Dict
        A dictionary of dataframes containing temperature data.
    s_temp : List
        A list containing a list a temperatures for each sample
        in every frame of the video.
    plate_temp : List
        A list containing a list a temperatures for each plate
        location in every frame of the video.
    s_infl : List
        A list containing the two possible melting points of all
        the samples obtained by the plot.
    m_df : Dataframe
        A dataframe containing row and column coordinates of each sample
        and its respective inflection point obtained.
    result_df : Dataframe
        Dataframe containing well number, predictions of noise net anf
        inflection net and melting point.

    '''

    # Determining the number of samples
    n_samples = n_columns * n_rows
    # Use the function 'flip_frame' to flip the frames horizontally
    # and vertically to correct for the mirroring during recording
    # flip_frames = flip_frame(frames)
    # Use the function 'edge_detection' to detect edges, fill and
    # label the samples.
    labeled_samples = edge_detection(frames, n_samples)
    # Use the function 'regprop' to determine centroids of all the samples
    regprops = regprop(labeled_samples, frames, n_rows, n_columns)
    # Use the function 'sort_regprops' to sort the dataframes in regprops
    sorted_regprops = sort_regprops(regprops, n_columns, n_rows)
    # Use the function 'sample_temp' to obtain temperature of samples
    # and plate temp
    s_temp, p_temp = sample_temp(sorted_regprops, frames)
    # Use the function 'sample_peaks' to determine the inflections points
    # and temperatures in sample temperature profiles
    s_peaks, s_infl = peak_detection(s_temp, p_temp, 'Sample')
    # # Use the function 'plate_peaks' to determine the inflections
    # # in plate temperature profiles
    # p_peaks, p_infl = peak_detection(s_temp, p_temp, 'Plate')
    # # Use the function 'infection_point' to obtain melting point of samples
    # inf_temp = inflection_point(s_temp, p_temp, s_peaks, p_peaks)
    result_df = de.final_result(s_temp, p_temp, path)
    # Creating a dataframe with row and column coordinates
    # of sample centroid and its melting temperature (Inflection point).
    m_df = pd.DataFrame({'Row': regprops[0].Row,
                         'Column': regprops[0].Column,
                         'Melting point': np.asarray(s_infl)[:, 0]})
    return sorted_regprops, s_temp, p_temp, s_infl, result_df
