# import os
import unittest

import pandas as pd
from phasIR import image_analysis as IA
from phasIR import thermal_analysis as TA


sample_location = pd.DataFrame(
    {'Column': [10], 'Row': [10]})
plate_location = pd.DataFrame(
    {'Plate_row': [5, 15, 5, 15], 'Plate_col': [5, 15, 5, 15]})
image_file = './phasIR/data/images/24_conical_empty_plate.png'
test_image = IA.input_file(image_file)


class TestSimulationTools(unittest.TestCase):
    def test_pixel_intensity(self):
        sample, plate = TA.pixel_intensity(
            sample_location, plate_location, [test_image])
        assert isinstance(sample, list), 'sample temperature should be a list'
        assert isinstance(plate, list), 'plate temperature should be a list'
        assert len(sample) == len(plate), \
            'the plate temprature list and the sample temperature list ' +\
            'should be the same length'
        return

    def test_find_temp_peak(self):

        return

    def test_get_temperature(self):

        return

    def test_baseline_subtraction(self):

        return

    def test_phase_transition_temperature(self):

        return

    def test_visualize_results(self):

        return
