import os
import unittest

from phasIR import image_analysis as IA
import numpy as np
from unittest.mock import patch

image_file = './phasIR/data/images/24_conical_empty_plate.png'
test_image = IA.input_file(image_file)
samples = IA.edge_detection(test_image[12:105, 17:148], 24)
sample_location = IA.sample_locations(samples, 24)
sorted = IA.sort_samples(sample_location, 6, 4)
mock_coords = [[1, 13], [24, 25]]


class TestSimulationTools(unittest.TestCase):
    # def test_input_file(self):
    #     '''Test for function which loads the input file'''
    #     frames = IA.input_file(test_image)
    #     assert isinstance(frames, np.ndarray), 'Output is not an array'
    #
    #     return

    def test_flip_frame(self):
        frames = IA.input_file(image_file)
        assert isinstance(frames, np.ndarray), 'Output is not an array'
        return

    def test_edge_detection(self):
        # test canny method
        samples = IA.edge_detection(test_image[12:105, 17:148], 24)
        assert isinstance(samples, list), 'the output should be a list'
        assert len(samples) == 24, \
            'number of samples found exceeds the expected one'
        # test sobel method
        samples = IA.edge_detection(test_image[12:105, 17:148], 24,
                                    method='sobel')
        assert isinstance(samples, list), 'the output should be a list'
        assert len(samples) == 24, \
            'number of samples found exceeds the expected one'
        return

    def test_sample_location(self):
        # test datatype as 'float'
        sample_location = IA.sample_locations(samples, 24)
        assert len(sample_location) == 24, ' the expected array is not the' +\
            ' lenght. Check input sample array'
        assert isinstance(sample_location['Column'].loc[0],
                          (float, np.float32))
        # test datatype as 'int'
        sample_location = IA.sample_locations(samples, 24, dtype='int')
        assert len(sample_location) == 24, ' the expected array is not the' +\
            ' lenght. Check input sample array'
        assert isinstance(sample_location['Column'].loc[0],
                          (float, np.float32, int, np.int64)),\
            'sample coordinates should be in numeric form'
        assert len(sample_location.columns) == 3, 'the dataframe should ' +\
            'contain 3 columns'
        return

    def test_sort_samples(self):
        samples = IA.edge_detection(test_image[12:105, 17:148], 24)
        sample_location = IA.sample_locations(samples, 24)
        sorted = IA.sort_samples(sample_location, 6, 4)
        assert len(sorted) == 24, 'resulting dataframe is not the ' +\
            'correct length'
        assert len(sorted.columns) == 2, 'the dataframe contains extra ' +\
            'columns. Expected number was 2, found {}'.format(
                len(sorted.columns))
        return

    def test_plate_location(self):
        coordinates = IA.plate_location(sorted, 6, 4)
        assert len(coordinates) == 24*4, 'resulting dataframe is not the ' +\
            'correct length'
        assert len(coordinates.columns) == 2, 'the dataframe contains ' +\
            'extra columns. Expected number was 2, found {}'.format(
                len(coordinates.columns))
        return

    @patch('matplotlib.pyplot.ginput', return_value=mock_coords)
    @patch('matplotlib.pyplot.waitforbuttonpress')
    def test_manual_centroid(self, patched_input1, patched_input2):

        coordinates = IA.manual_centroid(test_image[12:105, 17:148])

        assert patched_input1.called_with(-1, True, -1, 1, 3, 2)
        assert coordinates['Column'].loc[0] == mock_coords[0][0],\
            'the selected points are not the right ones'

        return
