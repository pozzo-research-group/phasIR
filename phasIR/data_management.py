import h5py
import itertools
import string

import numpy as np
import pandas as pd

def load_csv(path, file):
    '''
    Function to load a csv file. The results from the thermal analysis can
    be saved directly into a csv file and loaded up with this function.

    Paramters
    ---------
    path : str
        Path to the location of the file to be loaded
    file : str
        Name of the file to be loaded

    Returns
    -------
    dataframe : pd.DataFrame
        Dataframe containing the data saved in the file.

    '''
    if file.endswth('.csv'):
        dataframe = pd.read_csv(path+file, index_col='Unnamed: 0')
    else:
        dataframe = pd.read_csv(path+file+'.csv', index_col='Unnamed: 0')
    return dataframe


def save_to_csv(dataframe, path, filename):
    '''
    File to save a dataframe into a .csv file

    Paramters
    ---------
    path : str
        Path to the location of the file to be saved
    file : str
        Name of the file to be saved
    dataframe : pd.DataFrame
        Dataframe containing to be saved.
    '''
    dataframe.to_csv(path+filename)
    return print('The data was correctly save as' +
                 '\033[1m{}\033[0m '.format(filename) +
                 ' in the following folder \033[1m{}\033[0m'.format(path))


def save_results(sample_temp, plate_temp, path, filename, n_col, n_rows):
    '''
    Funciton to save the extracted temperature profiles for each well.
    The data for each well is sved separately as a dataset with label as the
    well name- Uppercase letter for rows and numbers for columns.


    Parameters
    ----------
    sample_temp : list
        list containing the sample temperature profile for each well
    plate_temp : list
        list containing the plate temperature profile for each well
    path : str
        Path to the location of the file to be saved
    filename : str
        Name of the file to be saved
    n_col : int
    n_rows : int
    '''
    # generate well name as: number fo columns (numbers) and rows (letters)
    columns = [str(i) for i in range(1, n_col+1)]
    rows = list(string.ascii_uppercase)[:n_rows]
    wells = list(itertools.product(columns, rows))
    # well name should be A1,A2,....B1,B2,...
    # The same naming convention is used on the well plates
    well_names = [str(wells[i][0]+wells[i][1]) for i in range(len(wells))]

    assert len(well_names) == len(sample_temp), \
        'the number of wells does not match the number' + \
        ' of tempeature profiles provided'

    # create the hdf5 file.
    result_file = h5py.File(path+filename+'.h5', 'w-')
    try:
        # create a group to store the temperature profiles of each well
        temperature_profile = result_file.create_group('temperature_profiles')

        for i in range(len(well_names)):
            temperature_profile.create_dataset(well_names[i], data=[
                    np.array(sample_temp[i]).astype('float64'),
                    np.array(plate_temp[i]).astype('float64')])
    except OSError:
        result_file.close()

    result_file.close()

    return print('The data was correctly save as' +
                 '\033[1m{}\033[0m '.format(filename) +
                 ' in the following folder \033[1m{}\033[0m'.format(path))


def read_results(path, filename):
    '''
    Function to load the hdf5 file containing the temperature profiles
    The results from the thermal analysis are saved as separate datasets.


    Paramters
    ---------
    path : str
        Path to the location of the file to be loaded
    filename : str
        Name of the file to be loaded

    Returns
    -------
    result_dictionary : dict
        Dictionary cnotaining the sample temperature and the plate temperature
        profiles for each well. The keys of the dictionary are the names of the
        wells as they are indicated on the well plate.
    '''
    if filename.endswith('.h5'):
        result_file = h5py.File(path+filename, 'r')
    else:
        result_file = h5py.File(path+filename+'.h5', 'r')

    temperature_profiles = result_file.get('temperature_profiles')
    well_names = list(temperature_profiles.keys())
    result_dictionary = {}
    for i in range(len(well_names)):
        temperature = temperature_profiles.get(well_names[i])
        data = {'Sample_temp': temperature[0],
                'Plate_temp': temperature[1]}
        result_dictionary[well_names[i]] = data

    return result_dictionary
