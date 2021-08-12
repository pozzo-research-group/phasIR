import os
import unittest

import pandas as pd

from phasIR import data_management as dm


class TestSimulationTools(unittest.TestCase):
    def test_load_csv(self):
        test_path = './phasIR/test/data/'
        test_file = 'Test_data.csv'
        df = dm.load_csv(test_path, test_file)

        assert isinstance(df, pd.DataFrame), \
            'the file was not correctly loaded'

        # Chack that if file extention is not given, it will still load
        # correctly
        test_path = './phasIR/test/data/'
        test_file_2 = 'Test_data.csv'
        df_2 = dm.load_csv(test_path, test_file_2)

        assert isinstance(df_2, pd.DataFrame), \
            'the file was not correctly loaded'
        return

    def test_save_to_csv(self):
        path = './phasIR/test/data/'
        test_file = 'Test_data.csv'
        df = dm.load_csv(path, test_file)

        # save the file
        save_filename = 'file_to_be_removed.csv'
        dm.save_to_csv(df, path, save_filename)

        files = os.listdir(path)
        assert save_filename in files, 'the file was not correctly saved'

        os.remove(path + save_filename)

        return

    def test_save_results(self):
        test_path = './phasIR/test/data/'
        test_file = 'Test_data.csv'
        df = dm.load_csv(test_path, test_file)

        dm.save_results([list(df['Sample_temp 1'])],
                        [list(df['Plate_temp 1'])],
                        test_path, 'test_file', 1, 1)

        files = os.listdir(test_path)
        assert 'test_file.h5' in files, 'the file was not correctly saved'
        return

    def read_results(self):
        test_path = './phasIR/test/data/'
        test_file = 'test_file.h5'
        result_dict = dm.read_results(test_path, test_file)
        assert isinstance(result_dict, dict), \
            'the file was not correctly loaded'
        return

    test_path = './phasIR/test/data/'
    os.remove(test_path + 'test_file.h5')
