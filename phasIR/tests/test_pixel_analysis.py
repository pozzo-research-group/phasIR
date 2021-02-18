import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import irtemp
import edge_detection
import pixel_analysis


# def test_name():
#     '''Doc String'''
#     #inputs
#     #running function
#     #asserts
#     return



##################### Peak detection and pixel analysis function #######################################

def test_image_eq():
    ''' Test for fucntion which equalizes a low contrast image'''
    frames = edge_detection.input_file('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[35:85,40:120])
    pixel_frames = edge_detection.flip_frame(crop_frame)
    img_eq = pixel_analysis.image_eq(pixel_frames)
    assert isinstance(img_eq,np.ndarray),'Output is not an array'
    assert pixel_frames[0].shape == img_eq.shape, 'Output array shape is not same as the input array shape.'
    return

def test_pixel_sum():
    '''Test for function which obtains the sum of pixels over all rows and columns'''
    frames = edge_detection.input_file('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[35:85,40:120])
    pixel_frames = edge_detection.flip_frame(crop_frame)
    img_eq = pixel_analysis.image_eq(pixel_frames)
    column_sum, row_sum = pixel_analysis.pixel_sum(img_eq)
    assert isinstance(column_sum,list),'Column sum is not a list'
    assert isinstance(row_sum,list),'Row sum is not a list'
    assert len(row_sum) == img_eq.shape[0], 'The length of row_sum is not equal to number of rows in the input image'
    assert len(column_sum) == img_eq.shape[1], 'The length of column_sum is not equal to number of columns in the input image'
    return

def test_peak_values():
    '''Test for function which finds peaks from the column_sum and row_sum arrays
        and return a dataframe with sample locations and plate locations.'''
    frames = edge_detection.input_file('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[35:85,40:120])
    pixel_frames = edge_detection.flip_frame(crop_frame)
    img_eq = pixel_analysis.image_eq(pixel_frames)
    column_sum, row_sum = pixel_analysis.pixel_sum(img_eq)
    n_columns = 3
    n_rows = 3
    r_peaks, c_peaks = pixel_analysis.peak_values(column_sum,row_sum,n_columns,n_rows,freeze_heat=False)
    assert isinstance(r_peaks, list), 'Output is not a list'
    assert isinstance(c_peaks, list), 'Output is not a list'
    assert len(r_peaks) == n_rows, 'Wrong number of sample rows detected'
    assert len(c_peaks) == n_columns, 'Wrong number of sample columns detected'
    return

def test_locations():
    '''Test for functin which returns a dataframe containing row and column
    locations of samples and their respective plate location at which temperature
    profiles are monitored'''
    frames = edge_detection.input_file('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[35:85,40:120])
    pixel_frames = edge_detection.flip_frame(crop_frame)
    img_eq = pixel_analysis.image_eq(pixel_frames)
    column_sum, row_sum = pixel_analysis.pixel_sum(img_eq)
    n_columns = 3
    n_rows = 3
    r_peaks, c_peaks = pixel_analysis.peak_values(column_sum,row_sum,n_columns,n_rows,freeze_heat=False)
    sample_location = pixel_analysis.locations(r_peaks, c_peaks, img_eq)
    assert isinstance(sample_location,pd.DataFrame),'Output is not a dataframe'
    assert len(sample_location)==n_columns*n_rows, 'Wrong number of sample locations are present'
    return

def test_pixel_intensity():
    '''Test for function which determines sample temperature and plate temperature'''
    frames = edge_detection.input_file('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[35:85,40:120])
    pixel_frames = edge_detection.flip_frame(crop_frame)
    img_eq = pixel_analysis.image_eq(pixel_frames)
    column_sum, row_sum = pixel_analysis.pixel_sum(img_eq)
    n_columns = 3
    n_rows = 3
    r_peaks, c_peaks = pixel_analysis.peak_values(column_sum,row_sum,n_columns,n_rows,freeze_heat=False)
    sample_location = pixel_analysis.locations(r_peaks, c_peaks, img_eq)
    x_name = 'Row'
    y_name = 'Column'
    plate_name = 'plate_location'
    pixel_sample,pixel_plate = pixel_analysis.pixel_intensity(sample_location, pixel_frames, x_name,y_name,plate_name)
    assert isinstance(pixel_sample,list),'Output is not a list'
    assert isinstance(pixel_plate,list),'Output is not a list'
    assert len(pixel_sample)==n_columns*n_rows,'Temperature obtained for wrong number of samples'
    assert len(pixel_plate)==n_columns*n_rows,'Temperature obtained for wrong number of plate locations'
    return

def test_pixel_temp():
    '''Test for the wrapping function'''
    frames = edge_detection.input_file('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[35:85,40:120])
    pixel_frames = edge_detection.flip_frame(crop_frame)
    n_columns = 3
    n_rows = 3
    path = 'musicalrobot/data/'
    result_df = pixel_analysis.pixel_temp(pixel_frames,n_columns,n_rows, path, freeze_heat=False)
    assert isinstance(result_df,pd.DataFrame),'Output obtained is not a dataframe'
    assert len(result_df)==n_columns*n_rows,'Temperature obtained for wrong number of samples'
    return