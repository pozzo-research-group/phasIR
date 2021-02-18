import os
import unittest


# import os,sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

import irtemp
import edge_detection
import data_encoding

#######################################
#######################################
# Test functions for Image Processing #
#######################################
#######################################

# Change path to this file!!
file_name = (
    '../musical-robot/musicalrobot/data/24_conical_empty_plate.HDF5')


class TestSimulationTools(unittest.TestCase):

    def test_input_file(self):
        '''
        Test for function which loads the input file
        '''
        frames = edge_detection.input_file(file_name)
        assert isinstance(frames, np.ndarray), 'Output is not an array'
        return

    def test_flip_frame(self):
        '''
        Test for function which flips the frames horizontally
        and vertically to correct for the mirroring during recording.
        '''
        frames = edge_detection.input_file(file_name)
        crop_frame = []
        for frame in frames:
            crop_frame.append(frame[35:85, 40:120])
        flip_frames = edge_detection.flip_frame(crop_frame)
        assert isinstance(flip_frames, list), 'Output is not a list'
        return

    def test_edge_detection(self):
        '''
        Test for function which detects edges,fills and labels the samples
        '''
        frames = edge_detection.input_file(file_name)
        crop_frame = []
        for frame in frames:
            crop_frame.append(frame[35:85, 40:120])
        # flip_frames = edge_detection.flip_frame(crop_frame)
        n_samples = 24
        labeled_samples = edge_detection.edge_detection(crop_frame, n_samples)
        assert isinstance(labeled_samples, np.ndarray),\
            'Output is not an array'
        assert crop_frame[0].shape == labeled_samples.shape,\
            'Input and Output array shapes are different.'
        return

    # def test_regprop(self):
    #     '''
    #     Test for function which determines centroids of all the samples
    #     and locations on the plate to obtain temperature from
    #     '''
    #     frames = edge_detection.input_file(file_name)
    #     crop_frame = []
    #     for frame in frames:
    #         crop_frame.append(frame[35:85, 40:120])
    #     # flip_frames = edge_detection.flip_frame(crop_frame)
    #     n_samples = 24
    #     n_rows = 4
    #     n_columns = 6
    #     labeled_samples = edge_detection.edge_detection(crop_frame, n_samples)
    #     regprops = edge_detection.regprop(labeled_samples, crop_frame, n_rows, n_columns)
    #     assert isinstance(regprops,dict),'Output is not a dictionary'
    #     assert len(regprops)==len(crop_frame),'The number of dataframes in the dictionary is not equal to number of frames input.'
    #     for i in range(len(crop_frame)):
    #         assert len(regprops[i])==n_samples,'Wrong number of samples detected'
    #     return

    def test_sort_regprops(self):
        '''Test for function which sorts the dataframes in the dictionary regprops'''
        file_name = ('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
        frames = edge_detection.input_file(file_name)
        crop_frame = []
        for frame in frames:
            crop_frame.append(frame[35:85,40:120])
        # flip_frames = edge_detection.flip_frame(crop_frame)
        n_samples = 9; n_rows = 3; n_columns = 3
        labeled_samples = edge_detection.edge_detection(crop_frame, n_samples)
        regprops = edge_detection.regprop(labeled_samples, crop_frame, n_rows, n_columns)
        sorted_regprops = edge_detection.sort_regprops(regprops, n_columns, n_rows)
        assert isinstance(sorted_regprops,dict),'Output is not a dictionary'
        assert len(sorted_regprops)==len(crop_frame),'The number of dataframes in the dictionary is not equal to number of frames input.'
        for i in range(len(crop_frame)):
            assert len(sorted_regprops[i])==n_samples,'Wrong number of samples detected'
        return


    def test_sample_temp(self):
        '''Test for function which obtaines temperature of samples and plate temperature'''
        file_name = ('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
        frames = edge_detection.input_file(file_name)
        crop_frame = []
        for frame in frames:
            crop_frame.append(frame[35:85,40:120])
        # flip_frames = edge_detection.flip_frame(crop_frame)
        n_samples = 9; n_rows = 3; n_columns = 3
        labeled_samples = edge_detection.edge_detection(crop_frame, n_samples)
        regprops = edge_detection.regprop(labeled_samples, crop_frame, n_rows, n_columns)
        sorted_regprops = edge_detection.sort_regprops(regprops, n_columns, n_rows)
        temp, plate_temp = edge_detection.sample_temp(sorted_regprops,crop_frame)
        assert isinstance(temp,list),'Sample temperature output is not a list'
        assert isinstance(plate_temp,list),'Plate temperature output is not a list'
        assert len(temp) == n_samples,'Temperature obtained for wrong number of samples'
        assert len(plate_temp) == n_samples,'Temperature obtained for wrong number of plate locations'
        return

    def test_sample_peaks(self):
        ''' Test for function which obtains the peaks in the sample temperature profile'''
        file_name = ('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
        frames = edge_detection.input_file(file_name)
        crop_frame = []
        for frame in frames:
            crop_frame.append(frame[35:85,40:120])
        # flip_frames = edge_detection.flip_frame(crop_frame)
        n_samples = 9; n_rows = 3; n_columns = 3
        labeled_samples = edge_detection.edge_detection(crop_frame, n_samples)
        regprops = edge_detection.regprop(labeled_samples, crop_frame, n_rows, n_columns)
        sorted_regprops = edge_detection.sort_regprops(regprops, n_columns, n_rows)
        temp, plate_temp = edge_detection.sample_temp(sorted_regprops, crop_frame)
        s_peaks, s_infl = edge_detection.peak_detection(temp, plate_temp, 'Sample')
        assert isinstance(s_peaks, list), 'Output is not a list'
        assert isinstance(s_infl, list), 'Output is not a list'
        assert len(s_peaks) == n_samples, 'Wrong number of peaks detected'
        assert len(s_infl) == n_samples, 'Wrong number of inflection temperatures detected'
        return
self

# def test_inflection_point():
#     '''Test for function which obtains the melting point of all the samples'''
#     file_name = ('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
#     frames = edge_detection.input_file(file_name)
#     crop_frame = []
#     for frame in frames:
#         crop_frame.append(frame[35:85,40:120])
#     # flip_frames = edge_detection.flip_frame(crop_frame)
#     n_samples = 9; n_rows = 3; n_columns = 3
#     labeled_samples = edge_detection.edge_detection(crop_frame, n_samples)
#     regprops = edge_detection.regprop(labeled_samples, crop_frame, n_rows, n_columns)
#     sorted_regprops = edge_detection.sort_regprops(regprops, n_columns, n_rows)
#     temp, plate_temp = edge_detection.sample_temp(sorted_regprops, crop_frame)
#     s_peaks, s_infl = edge_detection.peak_detection(temp, plate_temp, 'Sample')
#     p_peaks, p_infl = edge_detection.peak_detection(temp, plate_temp, 'Plate')
#     inf_temp = edge_detection.inflection_point(temp, plate_temp, s_peaks, p_peaks)
#     assert isinstance(inf_temp, list), 'Output is not a list'
#     assert len(inf_temp) == n_samples, 'Wrong number of melting points determined'
#     return

def test_inflection_temp():
    '''Test for wrapping function'''
    file_name = ('../musical-robot/musicalrobot/data/10_17_19_PPA_Shallow_plate.tiff')
    frames = edge_detection.input_file(file_name)
    crop_frame = []
    for frame in frames:
        crop_frame.append(frame[35:85,40:120])
    n_samples = 9; n_rows = 3; n_columns = 3
    path = 'musicalrobot/data/'
    sorted_regprops, s_temp, p_temp, s_infl, result_df = edge_detection.inflection_temp(crop_frame, n_rows, n_columns, path)
    assert isinstance(crop_frame,list),'Output is not a list'
    assert isinstance(s_infl, list),'Output is not a list'
    assert len(s_infl) == n_samples,'Wrong number of samples detected'
    for i in range(len(crop_frame)):
        assert len(sorted_regprops[i])==n_samples,'Wrong number of samples detected'
    assert isinstance(s_temp,list),'Sample temperature output is not a list'
    assert isinstance(p_temp,list),'Plate temperature output is not a list'
    assert len(s_temp) == n_samples,'Temperature obtained for wrong number of samples'
    assert len(p_temp) == n_samples,'Temperature obtained for wrong number of plate locations'
    assert isinstance(result_df,pd.DataFrame),'Output is not a dataframe'
    assert len(result_df) == n_samples, 'Inflection temperatures obtained for wrong number of samples'
    return
