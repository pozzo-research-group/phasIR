import numpy as np
import pandas as pd
import pickle
import matplotlib
import json
import os
import sys

matplotlib.use('Agg')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from matplotlib import pyplot as plt
# import edge_detection_ver2 as ed
# from XXXXX import inflection_temperature
from scipy.interpolate import interp1d
from scipy.signal import filtfilt
from scipy.interpolate import BSpline
from skimage import data, color
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import model_from_json


# To calculate derivative of the temp profile
def derivative(sample_temp, plate_temp):
    '''
    Funtion to determine the derivative of the
    of all the sample temperature profiles.

    Parameters
    -----------
        sample_temp : List
            Temperature of all the samples in
            every frame of the video.
        plate_temp : List
            Temperature profiles of all the plate locations

    Returns
    --------
        derivative: List
            Derivative of temperature profiles
            of all the samples
    '''
    derivative = []
    for i in range(len(sample_temp)):
        # Fitting a spline to the temperature profile of the samples.
        # if material == 'Plate':
        #     bspl = BSpline(frames,plate_temp[i],k=3)
        #     # Stacking x and y to calculate gradient.
        #     gradient_array = np.column_stack((frames,bspl(frames)))
        # else:
        f = interp1d(plate_temp[i], sample_temp[i], bounds_error=False)
        x = np.linspace(min(plate_temp[i]),
                        max(plate_temp[i]), len(plate_temp[i]))
        y = f(x)
        n = 25  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        yy = filtfilt(b, a, y)
        gradient_array = np.column_stack((x, yy))
        # Calculating gradient
        first_gradient = np.gradient(gradient_array, axis=0)
        # Calculating derivative
        derivative.append(first_gradient[:, 1]/first_gradient[:, 0])
#         deri_array = img_to_array(plt.plot(derivative))
    return derivative


# First channel for temperature profile
def plot_to_array(x, y, length):
    '''
    Funtion to generate gray image of temperature profile.

    Parameters
    -----------
        x : List
            Sample temperature
        y : List
            Plate temperature
        length: int
            Length and width of the image required for
            neural network input

    Returns
    --------
        gray_image : Array
            Array of the grayscale temperature profile image
    '''
    # Plotting the image
    fig, ax = plt.subplots(figsize=(length, length), dpi=100)
    ax.plot(x, y)
    ax.axis('off')
    # Triggering a canvas to save it as a buffer
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    # Converting it to an array from buffer
    array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    plt.close(fig)
    # Converting into gray scale
    gray_image = color.rgb2gray(array)
    return gray_image


# Second channel for derivative
def plot_to_array1(x, length):
    '''
    Funtion to generate gray image of the derivative
    of the temperature profile.

    Parameters
    -----------
        x : List
            Derivative
        length: int
            Length or width of the image required for
            neural network input

    Returns
    --------
        gray_image : Array
            Array of the grayscale derivative image
    '''
    fig, ax = plt.subplots(figsize=(length, length), dpi=100)
    ax.plot(x)
    ax.axis('off')
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    plt.close(fig)
    gray_image = color.rgb2gray(array)
    return gray_image


# Generating images for noise net
def noise_image(temp, plate_temp, path):
    '''
    Funtion to generate grayscale image of the of the temperature
    profile for every sample.

    Parameters
    -----------
        temp : List
            Temperature of all the samples in
            every frame of the video.
        plate_temp : List
            Temperature profiles of all the plate locations
        path : String
            Path to the location to temporarily store neural
            network input images.

    Returns
    --------
        Creates a directory names 'noise_images' in the current
        directory and saves all the images generated in it.
    '''
    dir_name = path + 'noise_images'
    try:
        # Creating directory to store images for noise net
        os.mkdir(dir_name)
        print("Directory ", dir_name, " Created ")
    except FileExistsError:
        # Removing old files
        filelist = [f for f in os.listdir(dir_name)]
        for f in filelist:
            os.remove(os.path.join(dir_name, f))
        print("Directory ", dir_name, " already exists")
    # Saving plots
    for i in range(len(temp)):
        fig = plt.figure()
        plt.plot(plate_temp[i], temp[i])
        plt.axis('off')
        fig.savefig(path+'noise_images/noise_'+str(i+1)+'.png')
        plt.close()
    return print('Noise images generated')


# Noise prediction
def noise_prediction(file_path):
    '''
    Funtion to classify temperature profiles as noisy or noiseless

    Parameters
    -----------
        file_path : String
            Path to the directory containing the images to be classified
                path : String

    Returns
    --------
        result_df : Dataframe
            Dataframe containing well number and noise net predictions
        nonoise_index : List
            List of sample numbers with noiseless temperature profiles
    '''
    noise_pred = {}
    nonoise_index = []
    files = [f for f in os.listdir(file_path)]
    file_names = list(filter(lambda x: x[-4:] == '.png', files))
    module_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(module_dir, 'models')
    model_labels = os.path.join(model_path, 'noise_net_labels.pkl')
    model_json_path = os.path.join(model_path, 'noise_net_bw5.json')
    model_weights = os.path.join(model_path, 'best_noisenet5.hdf5')
    with open(model_labels, 'rb') as handle:
        labels = pickle.load(handle)
    print(labels)
    # Loading the model
    with open(model_json_path, 'r') as f:
        model_json = json.load(f)
    model = model_from_json(model_json)
    model.load_weights(model_weights)
    for file in file_names:
        image = load_img(file_path+file,
                         target_size=(150, 150), color_mode='grayscale')
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshaping the image
        image = image.reshape((1, image.shape[0],
                               image.shape[1], image.shape[2]))
        # Predicting the class
        prediction = model.predict_classes(image)[0][0]
        # Extracting sample number from file name
        if len(file) == 12:
            sample_number = int(file[6:8])-1
        if len(file) == 11:
            sample_number = int(file[6:7])-1
        # Saving the prediction in a dictionary
        noise_pred[sample_number] = prediction
        # Saving samples with noiseless plots
        if prediction == 0:
            if len(file) == 12:
                nonoise_index.append(int(file[6:8])-1)
            if len(file) == 11:
                nonoise_index.append(int(file[6:7])-1)
    # Creating lists for the dataframe
    well_number = list(noise_pred.keys())
    pred_values = list(noise_pred.values())
    result_df = pd.DataFrame({'Sample number': well_number,
                              'Noise net': pred_values})
    result_df.sort_values(by=['Sample number'], inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    return result_df, nonoise_index


# Generating images for inflection net
def inf_images(temp, plate_temp, n, nonoise_index, path):
    '''
    Funtion to generate grayscale image of the of the temperature
    profile for every sample.

    Parameters
    -----------
        temp : List
            Temperature of all the samples in
            every frame of the video.
        plate_temp : List
            Temperature profiles of all the plate locations
        n : int
            Length or width of the images to be generated
        nonoise_index : List
            List of sample numbers with noiseless temperature profiles
        path : String
            Path to the location to temporarily store neural
            network input images.

    Returns
    --------
        Creates a directory names 'noise_images' in the current
        directory and saves all the images generated in it.
    '''
    dir_name = path + 'inf_images'
    try:
        # Creating directory to store images for inflection net
        os.mkdir(dir_name)
        print("Directory ", dir_name, " Created ")
    except FileExistsError:
        # Removing old files
        filelist = [f for f in os.listdir(dir_name)]
        for f in filelist:
            os.remove(os.path.join(dir_name, f))
        print("Directory ", dir_name, " already exists")
    # Calculating derivative of temp. profiles of all the
    # samples
    deri = derivative(temp, plate_temp)
    # Stacking temp profile and its derivative in a single
    # image
    for i in nonoise_index:
        img1 = plot_to_array(plate_temp[i], temp[i], n)
        img2 = plot_to_array1(deri[i], n)
        img3 = np.zeros([n*100, n*100], dtype=np.uint8)
        img3 = (img3*255).astype(np.uint8)
        new_img = np.dstack((img1, img2, img3))
        # Saving plots
        fig = plt.figure()
        plt.imshow(new_img)
        plt.axis('off')
        fig.savefig(path+'inf_images/inf_'+str(i+1)+'.png')
        plt.close()
    return print('Generated inflection images')


# Inflection prediction
def inf_prediction(file_path):
    '''
    Funtion to classify temperature profiles as with and without an inflection

    Parameters
    -----------
        file_path : String
            Path to the directory containing the images to be classified

    Returns
    --------
        inf_pred : dict
            Dictionary containing the neural net prediction for each sample.
            The sample numbers are used as dictionary keys
        inf_index : List
            List of sample numbers with temperature profiles containing
            an inflection
    '''
    inf_pred = {}
    inf_index = []
    files = [f for f in os.listdir(file_path)]
    file_names = list(filter(lambda x: x[-4:] == '.png', files))
    module_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(module_dir, 'models')
    model_labels = os.path.join(model_path, 'inflection_net_labels.pkl')
    model_json_path = os.path.join(model_path, 'inflection_net_der3.json')
    model_weights = os.path.join(model_path, 'best_derinf3.hdf5')
    with open(model_labels, 'rb') as handle:
        labels = pickle.load(handle)
    print(labels)
    # Loading the model
    with open(model_json_path, 'r') as f:
        model_json = json.load(f)
    model = model_from_json(model_json)
    model.load_weights(model_weights)
    for file in file_names:
        image = load_img(file_path+file,
                         target_size=(200, 200))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshaping the image
        image = image.reshape((1, image.shape[0],
                               image.shape[1], image.shape[2]))
        # Predicting the class
        prediction = model.predict_classes(image)[0][0]
        # Extracting sample number from file name
        if len(file) == 9:
            sample_number = int(file[4:5])-1
        if len(file) == 10:
            sample_number = int(file[4:6])-1
        inf_pred[sample_number] = prediction
        # Saving samples with an inflection
        if prediction == 0:
            inf_index.append(sample_number)
    inf_index.sort()
    return inf_pred, inf_index


# Wrapping function to generate the result dataframe
def final_result(sample_temp, plate_temp, path):
    '''
    Funtion to classify temperature profiles as noisy or noiseless

    Parameters
    -----------
    sampe_temp : List
        Temperature of all the samples in
        every frame of the video.
    plate_temp : List
        Temperature profiles of all the plate locations
    path : String
        Path to the location to temporarily store neural
        network input images.

    Returns
    --------
    result_df : Dataframe
        Dataframe containing well number, predictions of noise net anf
        inflection net and melting point.
    '''
    # Generating noise images
    noise_image(sample_temp, plate_temp, path)
    # Making predictions using noise net
    file_path = path + 'noise_images/'
    result_df, nonoise_index = noise_prediction(file_path)
    # Generating inflection images
    inf_images(sample_temp, plate_temp, 2, nonoise_index, path)
    # Making prediction using inflection net
    file_path = path + 'inf_images/'
    inf_pred, inf_index = inf_prediction(file_path)
    # Extracting melting point
    infl_temp = []
    for i in range(len(sample_temp)):
        infl = inflection_temp(plate_temp, sample_temp, plot=False)
        infl_temp.append(infl)

    melting_point = np.asarray(infl_temp)
    # Adding inflection and melting point data to the dataframe
    result_df['Inf net'] = '-'
    result_df['Melting point'] = '-'
    for i in nonoise_index:
        result_df['Inf net'].loc[i] = inf_pred[i]
    for i in inf_index:
        result_df['Melting point'].loc[i] = melting_point[i]
    result_df['Inf net'].replace(0, 'Inflection', inplace=True)
    result_df['Inf net'].replace(1, 'No Inflection', inplace=True)
    result_df['Noise net'].replace(0, 'Noiseless', inplace=True)
    result_df['Noise net'].replace(1, 'Noisy', inplace=True)
    return result_df
