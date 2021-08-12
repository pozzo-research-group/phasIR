import unittest

import numpy as np
import pandas as pd

from phasIR import image_analysis as IA
from phasIR import thermal_analysis as TA
from phasIR import data_management as DM


sample_location = pd.DataFrame(
    {'Column': [10, 10], 'Row': [10, 10]})
plate_location = pd.DataFrame(
    {'Plate_row': [5, 15, 5, 15, 5, 15, 5, 15],
     'Plate_col': [5, 15, 5, 15, 5, 15, 5, 15]})
image_file = './doc/data/empty_plates_images/24_conical_empty_plate.png'
test_image = IA.input_file(image_file)
test_path = './phasIR/test/data/'
test_file = 'Test_data.csv'


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
        data_df = DM.load_csv(test_path, test_file)
        temp_df = TA.baseline_subtraction(
            data_df['Plate_temp 1'], data_df['Sample_temp 1'])
        peak_onset, peak_max = TA.find_temp_peak(
            temp_df['Delta_T'])
        assert isinstance(peak_onset, list), \
            'The peak onset is not correctly identified'
        assert isinstance(peak_max, list), \
            'The peak maximum is not correctly identified'
        if peak_onset != []:
            assert isinstance(peak_onset[0], int), \
                'the peak finding function did not return the correct object'
        if peak_max != []:
            assert isinstance(peak_onset[0], int), \
                'the peak finding function did not return the correct object'
        return

    def test_get_temperature(self):
        sample, plate = TA.pixel_intensity(
            sample_location, plate_location, [test_image])
        temp_df = TA.baseline_subtraction(
            plate[0], sample[0])
        peak_onset, peak_max = TA.find_temp_peak(
            temp_df['Delta_T'])
        onset_temp, peak_temp = TA.get_temperature(
            temp_df, peak_onset, peak_max)
        assert isinstance(onset_temp, np.ndarray), \
            "the tempteraute was not correctly extracted from the data"

        if len(onset_temp) != 0:
            assert isinstance(onset_temp[0], int), \
                'the onset tempeature is not correctly identified'
        if len(peak_temp) != 0:
            assert isinstance(peak_tempt[0], int), \
                'the peak temperature is not correclty identified'
        return

    def test_baseline_subtraction(self):
        sample, plate = TA.pixel_intensity(
            sample_location, plate_location, [test_image])
        temp_df = TA.baseline_subtraction(
            plate[0], sample[0])
        assert isinstance(temp_df, pd.DataFrame), \
            'the Delta T curve was not correctly subtracted'
        assert 'Delta_T' in temp_df.columns,\
            'the Delta T curve was not saved in the dataframe'
        return

    def test_phase_transition_temperature(self):
        sample, plate = TA.pixel_intensity(
            sample_location, plate_location, [test_image])
        final_temp_df = TA.phase_transition_temperature(
            plate, sample)

        assert isinstance(final_temp_df, pd.DataFrame), \
            'The phase transition temperature is not extracted correctly. ' + \
            'The final result is not a dataframe'
        return

    def test_visualize_results(self):
        sample, plate = TA.pixel_intensity(
            sample_location, plate_location, [test_image])
        temp_df = TA.baseline_subtraction(
            plate[0], sample[0])
        final_temp_df = TA.phase_transition_temperature(
            plate, sample)
        ax = TA.visualize_results(
            temp_df, final_temp_df['Sample_temp_peak'],
            final_temp_df['Sample_temp_onset'],
            final_temp_df['Plate_temp_peak'],
            final_temp_df['Plate_temp_onset'])
        assert ax, 'The graph was not generated'
        return
