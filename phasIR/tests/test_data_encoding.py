import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import irtemp
import edge_detection
import data_encoding


def test_derivative():
    '''Test for function which calculates the derivative
       of the temperature profile for neural network input'''
    file_name = ('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    frames = edge_detection.input_file(file_name)
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[35:85,40:120])
    n_samples = 9; n_rows = 3; n_columns = 3
    labeled_samples = edge_detection.edge_detection(crop_frame, n_samples)
    regprops = edge_detection.regprop(labeled_samples, crop_frame, n_rows, n_columns)
    sorted_regprops = edge_detection.sort_regprops(regprops, n_columns, n_rows)
    temp, plate_temp = edge_detection.sample_temp(sorted_regprops,crop_frame)
    derivative_list = data_encoding.derivative(temp, plate_temp)
    assert isinstance(derivative_list, list), 'Output is not a list'
    assert len(derivative_list) == len(temp), 'Incorrect number of derivatives'
    return

def test_plot_to_array():
    '''Test for function which generates a gray images
       of the temperature profile'''
    file_name = ('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    frames = edge_detection.input_file(file_name)
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[35:85,40:120])
    n_samples = 9; n_rows = 3; n_columns = 3
    labeled_samples = edge_detection.edge_detection(crop_frame, n_samples)
    regprops = edge_detection.regprop(labeled_samples, crop_frame, n_rows, n_columns)
    sorted_regprops = edge_detection.sort_regprops(regprops, n_columns, n_rows)
    temp, plate_temp = edge_detection.sample_temp(sorted_regprops,crop_frame)
    for i in range(len(temp)):
        x = plate_temp[i]
        y = temp[i]
        length = 2
        gray_image = data_encoding.plot_to_array(x ,y, length)
        assert isinstance(gray_image, np.ndarray), 'Output is not an array'
        assert len(gray_image) == 200, 'Incorrectly sized output array'
    return

def test_plot_to_array1():
    '''Test for function which generates an array of the
       image of derivative of the temperature profile'''
    file_name = ('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    frames = edge_detection.input_file(file_name)
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[35:85,40:120])
    n_samples = 9; n_rows = 3; n_columns = 3
    labeled_samples = edge_detection.edge_detection(crop_frame, n_samples)
    regprops = edge_detection.regprop(labeled_samples, crop_frame, n_rows, n_columns)
    sorted_regprops = edge_detection.sort_regprops(regprops, n_columns, n_rows)
    temp, plate_temp = edge_detection.sample_temp(sorted_regprops,crop_frame)
    for i in range(len(temp)):
        x = plate_temp[i]
        length = 2
        gray_image = data_encoding.plot_to_array1(x, length)
        assert isinstance(gray_image, np.ndarray), 'Output is not an array'
        assert len(gray_image) == 200, 'Incorrectly sized output array'
    return

def test_noise_prediction():
    '''Test for function which classifies images as noisy or
       not noisy'''
    file_name = ('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    frames = edge_detection.input_file(file_name)
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[35:85,40:120])
    n_samples = 9; n_rows = 3; n_columns = 3
    labeled_samples = edge_detection.edge_detection(crop_frame, n_samples)
    regprops = edge_detection.regprop(labeled_samples, crop_frame, n_rows, n_columns)
    sorted_regprops = edge_detection.sort_regprops(regprops, n_columns, n_rows)
    temp, plate_temp = edge_detection.sample_temp(sorted_regprops,crop_frame)
    data_encoding.noise_image(temp, plate_temp, 'musicalrobot/data/')
    file_path = 'musicalrobot/data/noise_images/'
    result_df, nonoise_index = data_encoding.noise_prediction(file_path)
    assert isinstance(result_df, pd.DataFrame), 'Output is not a dataframe'
    assert len(result_df) == len(temp), 'Incorrect number of samples in the dataframe'
    assert isinstance(nonoise_index, list), 'Output is not a list'

def test_inf_prediction():
    '''Test for function which classifies images as with and
       without inflection'''
    file_name = ('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    frames = edge_detection.input_file(file_name)
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[35:85,40:120])
    n_samples = 9; n_rows = 3; n_columns = 3
    labeled_samples = edge_detection.edge_detection(crop_frame, n_samples)
    regprops = edge_detection.regprop(labeled_samples, crop_frame, n_rows, n_columns)
    sorted_regprops = edge_detection.sort_regprops(regprops, n_columns, n_rows)
    temp, plate_temp = edge_detection.sample_temp(sorted_regprops,crop_frame)
    data_encoding.noise_image(temp, plate_temp, 'musicalrobot/data/')
    file_path = 'musicalrobot/data/noise_images/'
    result_df, nonoise_index = data_encoding.noise_prediction(file_path)
    data_encoding.inf_images(temp, plate_temp, 2, nonoise_index,'musicalrobot/data/')
    file_path = 'musicalrobot/data/inf_images/'
    inf_pred, inf_index = data_encoding.inf_prediction(file_path)
    assert isinstance(inf_pred, dict),'Output is not a dictionary'
    assert isinstance(inf_index, list), 'Output is not a list'
    return

def test_final_result():
    '''Test for the wrappung function'''
    file_name = ('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    frames = edge_detection.input_file(file_name)
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[35:85,40:120])
    n_samples = 9; n_rows = 3; n_columns = 3
    labeled_samples = edge_detection.edge_detection(crop_frame, n_samples)
    regprops = edge_detection.regprop(labeled_samples, crop_frame, n_rows, n_columns)
    sorted_regprops = edge_detection.sort_regprops(regprops, n_columns, n_rows)
    temp, plate_temp = edge_detection.sample_temp(sorted_regprops,crop_frame)
    path = 'musicalrobot/data/'
    result_df = data_encoding.final_result(temp, plate_temp, path)
    assert isinstance(result_df, pd.DataFrame), 'Output is not a dataframe'
    assert len(result_df) == len(temp), 'Incorrect number of samples classfied'
    return