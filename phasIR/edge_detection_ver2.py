"""
    Module to improve edge detection algorithm for well detection
"""

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
from pixel_analysis import image_eq


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Function to load the input file
def input_file(file_name):
    '''
    To load the imput file as an array.

    Parameters
    -----------
    file_name : String
        Name of the file to be loaded as it is saved on the disk.
        Provide file path if it is not in the same directory as
        the jupyter notebook.

    Returns
    --------
    frames : Array
        In case of a video, returns an array for each frame
        in the video. In case of an image, return an array.
    '''
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
def edge_detection(frames, n_samples, method='canny', track=False):
    """
    To detect the edges of the wells, fill and label them to
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
    track : Boolean
        to enable spatial tracking (to be implemented with real-time
        in the future)

    Returns
    --------
    labeled_samples : Array
        All the samples in the frame are labeled
        so that they can be used as props to get pixel data.
    """

    # when enable spatial tracking
    if track:
        # type cast to ndarray
        if not isinstance(frames, np.ndarray):
            frames_array = np.array(frames)
        else:
            frames_array = frames

        video_length = len(frames_array)
        video_with_label = np.empty(frames_array.shape, dtype=int)
        background = frames_array.mean(0)
        alpha = 2  # intensity threshold
        counter = 0
        missing = 0
        boolean_mask = None
        for time in range(video_length):
            # remove background proportional to time in video
            img_lin_bg = \
                frames_array[time] - background * time / (video_length - 1)
            # apply sobel filter
            edges_lin_bg = filters.sobel(img_lin_bg)
            #  booleanize with certain threshold alpha
            edges_lin_bg = edges_lin_bg > edges_lin_bg.mean() * alpha
            # erode edges, fill in holes
            edges_lin_bg = \
                ndimage.binary_erosion(edges_lin_bg, mask=boolean_mask)

            edges_lin_bg = binary_fill_holes(edges_lin_bg)

            # find progressive background
            if time is 0:
                progressive_background = 0
            else:
                progressive_background = frames_array[0:time].mean(0)
            # remove background
            img_prog_bg = frames_array[time] - progressive_background
            # apply sobel filter
            edges_prog_bg = filters.sobel(img_prog_bg)
            #  booleanize with certain threshold alpha
            edges_prog_bg = edges_prog_bg > edges_prog_bg.mean() * alpha
            # erode edges, fill in holes
            edges_prog_bg = \
                ndimage.binary_erosion(edges_prog_bg, mask=boolean_mask)

            edges_prog_bg = binary_fill_holes(edges_prog_bg)

            # combining
            combined_samples = edges_lin_bg + edges_prog_bg
            #  make the boolean mask for the for frame
            if time is 0:
                boolean_mask = ~ndimage.binary_erosion(combined_samples)
                # boolean_mask = ~combined_samples

            # labeled_samples =
            # ndimage.binary_erosion(labeled_samples, mask=boolean_mask)
            # labeled_samples =
            # binary_fill_holes(labeled_samples, structure=np.ones((2,2)))

            # remove stray pixels and label
            combined_samples = \
                remove_small_objects(combined_samples, min_size=2)
            labeled_samples = label(combined_samples)

            # confirm matching labels vs n_samples
            unique, counts = np.unique(labeled_samples, return_counts=True)
            label_dict = dict(zip(unique, counts))

            #  in case of missing label
            if len(label_dict) < n_samples+1:
                trial = 0
                # keep eroding to separate the samples
                while len(label_dict) < n_samples+1 and trial < 10:
                    labeled_samples = \
                        ndimage.binary_erosion(
                            labeled_samples, mask=boolean_mask)
                    labeled_samples = label(labeled_samples)
                    unique, counts = \
                        np.unique(labeled_samples, return_counts=True)
                    label_dict = dict(zip(unique, counts))
                    trial += 1
                # print('missing:', time)
                missing += 1

            # in case of extra label identify
            if len(label_dict) > n_samples + 1:
                trial = 0
                # keep removing smaller labels until matching with n_samples
                while len(label_dict) > n_samples + 1 and trial < 10:
                    temp = min(label_dict.values())
                    labeled_samples = \
                        remove_small_objects(
                            labeled_samples, min_size=temp + 1)
                    unique, counts = \
                        np.unique(labeled_samples, return_counts=True)
                    label_dict = dict(zip(unique, counts))
                    trial += 1

                # print('excess:', time, val)
                counter += 1

            video_with_label[time] = labeled_samples
        # print(counter)
        # print(missing)
        return video_with_label

    # when disable spatial tracking (default)
    else:
        labeled_samples = None
        size = None
        thres = None
        props = None

        # use canny edge detection method
        if method is 'canny':
            for size in range(15, 9, -1):
                for thres in range(1500, 900, -100):
                    edges = feature.canny(frames[0]/thres)

                    # fig = plt.figure(2)  # for debugging
                    # plt.imshow(edges)
                    # plt.show()

                    filled_samples = binary_fill_holes(edges)
                    cl_samples = \
                        remove_small_objects(filled_samples, min_size=size)
                    labeled_samples = label(cl_samples)
                    props = \
                        regionprops(labeled_samples, intensity_image=frames[0])

                    # fig = plt.figure(3)
                    # plt.imshow(filled_samples)  # for debugging

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
            # plt.show()  # for debugging
            return labeled_samples

        # use sobel edge detection method
        if method is 'sobel':
            for size in range(15, 9, -1):
                # use sobel
                edges = filters.sobel(frames[0])
                edges = edges > edges.mean() * 3  # booleanize data

                # fig = plt.figure(2)  # for debugging
                # plt.imshow(edges)
                # plt.colorbar()

                #  fill holes and remove noise
                filled_samples = binary_fill_holes(edges)
                cl_samples = \
                    remove_small_objects(filled_samples, min_size=size)
                labeled_samples = label(cl_samples)
                props = regionprops(labeled_samples, intensity_image=frames[0])

                # fig = plt.figure(3)
                # plt.imshow(filled_samples)  # for debugging

                if len(props) == n_samples:
                    break
            if size == 10 and len(props) != n_samples:
                print('Not all the samples are being recognized with the set \
                    minimum size and threshold range')
            # plt.show()  # for debugging
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

    missing = 0
    index = 0

    for i in range(len(frames)):
        if len(labeled_samples.shape) is 3:
            props = regionprops(labeled_samples[i], intensity_image=frames[i])
        elif len(labeled_samples.shape) is 2:
            props = regionprops(labeled_samples, intensity_image=frames[i])
        else:
            raise ValueError('Invalid labeled samples dimension')

        # Initializing arrays for all sample properties obtained from regprops.
        row = np.zeros(len(props)).astype(int)
        column = np.zeros(len(props)).astype(int)
        area = np.zeros(len(props))
        radius = np.zeros(len(props))
        perim = np.zeros(len(props))
        intensity = np.zeros(len(props), dtype=np.float64)
        plate = np.zeros(len(props), dtype=np.float64)
        plate_coord = np.zeros(len(props))

        unsorted_label = np.zeros((len(props), 5)).astype(int)
        sorted_label = np.zeros((len(props), 4)).astype(int)

        # collect data on centroid
        for item in range(len(props)):
            unsorted_label[item, 0] = int(props[item].centroid[0])
            unsorted_label[item, 1] = int(props[item].centroid[1])
            unsorted_label[item, 3] = item
            unsorted_label[item, 4] = np.unique(labeled_samples[i])[item+1]

        # sort label based on euclidean distance
        for item in range(len(props)):
            unsorted_label[item, 2] = np.power(
                unsorted_label[item, 0] + unsorted_label[:, 0].min(), 2) + \
                np.power(
                    unsorted_label[item, 1] - unsorted_label[:, 1].min(), 2)

            sorted_label = unsorted_label[unsorted_label[:, 2].argsort()]

        c = 0
        for item in range(len(props)):
            prop = props[sorted_label[item, 3]]

            row[c] = int(prop.centroid[0])
            column[c] = int(prop.centroid[1])
            area[c] = prop.area

            loc_index = \
                np.argwhere(labeled_samples[i] == sorted_label[item, 4])

            left_side_column = min(loc_index[:, 0]) - 1
            right_side_column = max(loc_index[:, 0]) + 1
            left_side_row = min(loc_index[:, 1]) - 1
            right_side_row = max(loc_index[:, 1]) + 1

            # This part is for gettng the total temp
            # and then get the average temp in each samples

            sample_temp = []
            for loc_index_len in range(len(loc_index)):
                x_coordinate = loc_index[loc_index_len].tolist()[0]
                y_coordinate = loc_index[loc_index_len].tolist()[1]

                result = frames[i][x_coordinate][y_coordinate]
                sample_temp.append(result)
            sum_temp_sample = np.sum(sample_temp)
            intensity[c] = sum_temp_sample / area[c]

            # This part is getting the environment temperature
            envir_area = (right_side_column - left_side_column + 1) * (
                right_side_row - left_side_row + 1) - area[c]

            # First, get the total temperature in the range crop rectangle
            total_rectangle_temp_list = []
            for j in range(right_side_column - left_side_column + 1):
                for k in range(right_side_row - left_side_row + 1):
                    crop_temp = \
                        frames[i][left_side_column + j][left_side_row + k]
                    total_rectangle_temp_list.append(crop_temp)

            # Next, use the result from the last step to minus the
            # sum_temp_sample, and you can get the sum_temp_envir
            total_rectangle_temp = np.sum(total_rectangle_temp_list)
            sum_temp_envir = total_rectangle_temp - sum_temp_sample
            plate[c] = sum_temp_envir / envir_area

            c = c + 1

        try:
            regprops[index] =\
                pd.DataFrame({'Row': row, 'Column': column,
                              'Plate_temp(cK)': plate,
                              'Radius': radius,
                              'Plate_coord': plate_coord,
                              'Area': area, 'Perim': perim,
                              'Sample_temp(cK)': intensity,
                              'unique_index': unique_index},
                             dtype=np.float64)
            regprops[index].sort_values(['Column', 'Row'], inplace=True)
            index += 1
        except ValueError:
            # print('Wrong number of samples detected in frame %d' % i)
            missing += 1
            continue

        if len(intensity) != n_samples:
            print('Wrong number of samples are being detected in frame %d' % i)

    if missing > 0:
        print(str(missing) + ' frames skipped due to missing samples')

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

    missing = 0
    for j in range(0, n_columns):
        df = regprops[0][j*n_rows:(j+1)*n_rows].sort_values(['Row'])
        sorted_rows.append(df)
    regprops[0] = pd.concat(sorted_rows)
    # Creating an index to be used for reordering all the dataframes.
    # The unique index is the sum of row and column coordinates.
    reorder_index = regprops[0].unique_index
    index = 0
    for k in range(0, len(regprops)):
        # print(k)
        try:
            regprops[k].set_index('unique_index', inplace=True)
            sorted_regprops[index] = regprops[k].reindex(reorder_index)
            index += 1
        except KeyError:
            # sorted_regprops[k] = regprops[k-1].reindex(reorder_index)
            # print('Skip frame ' + str(k) + ' due to missing sample')
            missing += 1
            continue

    if missing > 0:
        print(str(missing) + ' frames skipped due to missing samples')

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
            try:
                temp_well.append(
                    centikelvin_to_celsius(
                        list(sorted_regprops[i]['Sample_temp(cK)'])[j]))
                plate_well_temp.append(
                    centikelvin_to_celsius(
                         list(sorted_regprops[i]['Plate_temp(cK)'])[j]))
            except KeyError:
                continue
        temp.append(temp_well)
        plate_temp.append(plate_well_temp)
    return temp, plate_temp


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
        frames = np.linspace(1, len(sample_temp[i]), len(sample_temp[i]))
        # Fitting a spline to the temperature profile of the samples.
        if material == 'Plate':
            bspl = BSpline(frames, sample_temp[i], k=3)
            # Stacking x and y to calculate gradient.
            gradient_array = np.column_stack((frames, bspl(frames)))
        else:
            f = interp1d(plate_temp[2], sample_temp[2], bounds_error=False)
            gradient_array = np.column_stack((plate_temp[i], f(plate_temp[i])))
        # Calculating gradient
        gradient = np.gradient(gradient_array, axis=0)
        # Calculating derivative
        derivative = gradient[:, 1]/gradient[:, 0]
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
            infl.append([plate_temp[i][peaks[inf_index1]],
                        plate_temp[i][peaks[inf_index2]]])
        else:
            infl.append([sample_temp[i][peaks[inf_index1]],
                        sample_temp[i][peaks[inf_index2]]])
    return peak_indices, infl


def inflection_point(s_temp, p_temp, s_peaks, p_peaks):
    '''
    Function to get the inflection point(melting point) for each sample.

    Parameters
    -----------
    s_temp : List
        Sample temperature profiles
    p_temp : List
        Plate location temperature profiles
    s_peaks : List
        List of two highest peak(inflection points) indices in the
        temperature profile of the samples.
    p_peaks : List
        List of two highest peak(inflection points) indices in the
        temperature profile of the plate.

    Returns
    --------
    inf_temp : List
        List of temperature at inflection points for each sample

    '''
    inf_peak = []
    inf_temp = []
    for i, peaks in enumerate(s_peaks):
        for peak in peaks:
            # Making sure the peak is present only in the sample temp profile
            if abs(peak - p_peaks[i][0]) >= 3:
                inf_peak.append(peak)
                break
            else:
                pass
    # Appending the temperature of the sample at the inflection point
    for i, temp in enumerate(s_temp):
        inf_temp.append(temp[inf_peak[i]])
    return inf_temp


# Wrapping functions
# Wrapping function to get the inflection point
def inflection_temp(frames, n_rows, n_columns, method='canny', ver=2):
    """
    Function to obtain sample temperature and plate temperature
    in every frame of the video using edge detection.

    Parameters
    -----------
    frames : List
        An list containing an array for each frame
        in the video or just a single array in case of an image.
    n_rows: List
        Number of rows of sample
    n_columns: List
        Number of columns of sample
    method: str
        Name of method to use for edge detection for version 2
    ver: int
        Number of detection version to be used

    Returns
    --------
    frames : Array
        An array of images which are flipped to correct the
        rotation caused by the IR camera
    regprops : Dict
        A dictionary of dataframes containing temperature data.
    s_temp : List
        A list containing a list a temperatures for each sample
        in every frame of the video.
    plate_temp : List
        A list containing a list a temperatures for each plate
        location in every frame of the video.
    inf_temp : List
        A list containing melting point of all
        the samples obtained by the plot.
    m_df : Dataframe
        A dataframe containing row and column coordinates of each sample
        and its respective inflection point obtained.
    """

    # Determining the number of samples
    n_samples = n_columns * n_rows
    # Use the function 'flip_frame' to flip the frames horizontally
    # and vertically to correct for the mirroring during recording
    # flip_frames = flip_frame(frames)
    # Use the function 'edge_detection' to detect edges, fill and
    # label the samples.

    # run the edge detection using version 1 or 2 based on user input
    if ver is 1:
        labeled_samples = edge_detection(frames, n_samples, method=method)
        # print(len(labeled_samples), type(labeled_samples))
    elif ver == 2:
        track = True
        labeled_samples = edge_detection(frames, n_samples, method=method,
                                         track=track)
        # print(len(labeled_samples), type(labeled_samples))
    else:
        raise ValueError('Invalid version input')

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

    # Use the function 'plate_peaks' to determine the inflections
    # in plate temperature profiles
    p_peaks, p_infl = peak_detection(s_temp, p_temp, 'Plate')

    # Use the function 'infection_point' to obtain melting point of samples
    inf_temp = inflection_point(s_temp, p_temp, s_peaks, p_peaks)

    # Creating a dataframe with row and column coordinates
    # of sample centroid and its melting temperature (Inflection point).
    m_df = pd.DataFrame({'Row': regprops[0].Row,
                         'Column': regprops[0].Column,
                         'Melting point': inf_temp})
    return sorted_regprops, s_temp, p_temp, inf_temp, m_df


# Function to crop a square image
def square_crop(frame, coordinate, half_length):
    """
    Takes a given frame, the coordinate of the centroid and the half length
    of the square. The function will crop the frame into square with a given
    side length. If the crop part is out of bounds, it will fit to the bounds
    of the given image.

    Parameters
    -------------
    frame: array
       The array of the tiff file.
    coordinate: array
        the array of coordinates of the centroid. e.g. [x, y]
    half_length: int
       half side length of the square
    -------------

    Returns
    -------------
    The array of the square image
    -------------

    """

    x1 = int(coordinate[0] - half_length)
    x2 = int(coordinate[0] + half_length)
    y1 = int(coordinate[1] - half_length)
    y2 = int(coordinate[1] + half_length)
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > len(frame.T):
        x2 = len(frame.T)
    if y2 > len(frame):
        y2 = len(frame)
    crop_image = frame[y1:y2, x1:x2]
    return crop_image
