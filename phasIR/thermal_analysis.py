
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# from phasIR.irtemp import centikelvin_to_celsius
from irtemp import centikelvin_to_celsius
from skimage.draw import disk
from scipy.signal import find_peaks, peak_widths


def pixel_intensity(sample_location, plate_location, frames, r=2):
    '''
    Function to find pixel intensity at all sample locations
    and plate locations in each frame.

    Parameters
    -----------
    sample_location : Dataframe
        A dataframe containing the coordinates of all samples.
    plate_location : Dataframe
        A dataframe containing the coordinates of the plate to use for each
        sample.
    frames : Array
        An array of arrays containing the frames of the IR video.
    r : Int
       Integer value to use for the radius of the disk applied to each
       sample location. The final sample temperature is obtained as an average
       value.

    Returns
    --------
    sample_temp : List
        A list containig the temperature of all the samples over the
        entire IR video.
    plate_temp : List
        Temperature of the plate next to every sample in every
        frame of the video.
    '''
    sample_temp = []
    plate_temp = []
    # Row values of the peaks
    row = sample_location['Row']
    # Column values of the peaks
    col = sample_location['Column']
    # Row value of the plate location
    p_row = plate_location['Plate_row']
    # Row value of the plate location
    p_col = plate_location['Plate_col']
    for i in range(len(sample_location)):
        temp_well = []
        plate_well_temp = []
        for frame in frames:
            rr, cc = disk((row[i], col[i]), radius=r)
            sample_intensity = np.mean(frame[rr, cc])
            plate_intensity = np.mean(
                frame[p_row[i*4:(i*4)+4], p_col[i*4:(i*4)+4]])
            temp_well.append(centikelvin_to_celsius(sample_intensity))
            plate_well_temp.append(centikelvin_to_celsius(plate_intensity))
        sample_temp.append(temp_well)
        plate_temp.append(plate_well_temp)
    return sample_temp, plate_temp


def baseline_subtraction(plate_temperature, sample_temperature, n=None):
    """
    Function to store the data in a pandas DataFrame and perform the baseline
    subtraction. In this case, the baseline is defined as the plate temperature
    as it is the reference material for the system.

    Parameters
    ----------
    plate_temperature : list
        List containing the plate temperature at every frame of the IR video
    sample_temperature : list
        List containing the sample temperature at every frame of the IR video
    n : int
        number of point to use for smoothing of the data. The smoothing is
        performed using a rolling average over n points. Default is 5% of
        the data

    Returns
    -------
    temperature_dataframe : pd.DataFrame
        Datafrane containing the temperature data for plate, sample, baseline
        and smoothed data
    """
    temperature_dataframe = pd.DataFrame(
        {'Frames': np.linspace(1, len(plate_temperature),
         len(plate_temperature)), 'Plate_temp': plate_temperature,
         'Sample_temp':  sample_temperature})
    if n:
        n = n
    else:
        n = int(0.05*len(temperature_dataframe['Plate_temp']))
    temperature_dataframe['Plate_avg'] = \
        temperature_dataframe.iloc[:, 1].rolling(window=n).mean()
    temperature_dataframe['Sample_avg'] = \
        temperature_dataframe.iloc[:, 2].rolling(window=n).mean()
    temperature_dataframe['Delta_T'] = \
        temperature_dataframe['Plate_avg']-temperature_dataframe['Sample_avg']

    return temperature_dataframe


def find_temp_peak(baseline_array, height=2, prominence=1):
    '''
    Funciton to evaluate the Delta Temperature curve obtained from the
    subtraction of the baseline. The peak(s) of the curve are determined,
    as well as their width to extract their onset.

    Parameters
    ----------
    baseline_array : pd.Series
        Delta Temperature curve obtained from the subtraction of the baseline
    heigth : int
        Minimum height of the peaks in the delta temperature curve
    prominence : int
        minimum prominence of the peaks in the delta temperature curve

    Returns
    -------
    peak_left_onset : list
        List containing the index of the left-side onset of the peak(s)
    peak_max : list
        List containing the index of the peak(s)
    '''
    peaks, properties = find_peaks(baseline_array, height=height,
                                   prominence=prominence)
    peak_left_onset = []
    peak_max = []
    for i in range(len(peaks)):
        peaks_w = peak_widths(baseline_array, [peaks[i]])
        peak_left_onset.append(int(round(peaks_w[2][0])))
        peak_max.append(peaks[i])
    return peak_left_onset, peak_max


def get_temperature(dataframe, peak_onset_index, peak_max_index, sample=True):
    """
    Function to extract the sample and plate temparature given the indices
    of the peak and its onset of the delta temperature curve

    Parameters
    ----------
    dataframe : pd.Dataframe
        Temperature dataframe containing the raw and delta temprature data
    peak_onset_index : list
        List of indices of the left-hand onset of the peak(s)
    peak_max_index : list
        List fo indices of the peak(s)
    sample : Boolean
        If True, temperature will be extracted using the sample temperature
        data, otherwise the plate data will be used

    Returns
    -------
    onset_temp : list
        List containing the temperature at the onset of the peak(s)
    peak_temp : list
        List containing the temperature at the peak(s)
    """
    if sample:
        onset_temp = dataframe['Sample_avg'].iloc[peak_onset_index].values
        peak_temp = dataframe['Sample_avg'].iloc[peak_max_index].values
    else:
        onset_temp = dataframe['Plate_avg'].iloc[peak_onset_index].values
        peak_temp = dataframe['Plate_avg'].iloc[peak_max_index].values

    return onset_temp, peak_temp


def phase_transition_temperature(plate_temperatures, sample_temperatures,
                                 plot=False):
    """
    Wrapping function to extract the phase transition temperature given the
    list of plate and sample temperatures of each well on the plate.
    The data will be converted into a dataframe. The delta temperature curve
    will be extracted and evaluated for peaks. Finally, the data can be plotted
    for visual inspection of results. The output of the function is a dataframe
    containing only the phase transition temperature -onset and peak

    Parameters
    ----------
    plate_temperature: list
        List containing the plate temperature at every frame of the IR video
    sample_temperature: list
        List containing the sample temperature at every frame of the IR video
    plot: boolean
        if True, a plot of the temprature profiles for each well identified
        on the plate will be displayed.

    Returns
    -------
    phase_transition_df: pd.DataFrame
        dataframe containing only the phase transition temperature(s)
    """
    assert len(plate_temperatures) == len(sample_temperatures),\
        'The temperature arrays provided are not the same length.'

    n = len(plate_temperatures)

    stemp_onset = []
    stemp_peak = []
    ptemp_onset = []
    ptemp_peak = []

    for i in range(n):
        dataframe = baseline_subtraction(
            plate_temperatures[i], sample_temperatures[i])
        peak_onset, peak_max = find_temp_peak(dataframe['Delta_T'])
        s_temp_onset, s_temp_peak = get_temperature(
            dataframe, peak_onset, peak_max, sample=True)
        p_temp_onset, p_temp_peak = get_temperature(
            dataframe, peak_onset, peak_max, sample=False)

        if len(s_temp_onset) == 0:
            stemp_onset.append('-')
            ptemp_onset.append('-')
            stemp_peak.append('-')
            ptemp_peak.append('-')
        else:
            stemp_onset.append(np.round(s_temp_onset, 2))
            ptemp_onset.append(np.round(p_temp_onset, 2))
            stemp_peak.append(np.round(s_temp_peak, 2))
            ptemp_peak.append(np.round(p_temp_peak, 2))

        if plot:
            visualize_results(
                dataframe, p_temp_onset, s_temp_onset,
                p_temp_peak, s_temp_peak)

    phase_transition_df = pd.DataFrame({'Sample_temp_onset': stemp_onset,
                                        'Sample_temp_peak': stemp_peak,
                                        'Plate_temp_onset': ptemp_onset,
                                        'Plate_temp_peak': ptemp_peak})

    return phase_transition_df


def visualize_results(raw_dataframe, plate_onset, sample_onset,
                      plate_peak, sample_peak):
    """
    Function to visualize the temperature profiles, delta temperature curve and
    phase transition temperature(s)

    Parameters
    ----------
    raw_dataframe : pd.DataFrame
        Temperature dataframe containing the raw and delta temprature data
    plate_onset : list
        List of the phase transition onset plate temperature(s)
    sample_onset : list
        List of the phase transition onset sample temperature(s)
    plate_peak : list
        List of the phase transition peak plate temperature(s)
    sample_peak : list
        List of the phase transition peak sample temperature(s)

    Returns
    -------
    ax : matplotlib axis
        Matplotlib axis object
    """
    # generate figure and axis object
    fig, ax = plt.subplots()
    # plot the raw dataset
    ax.scatter(raw_dataframe['Plate_temp'], raw_dataframe['Sample_temp'],
               c='lightgray', label='Raw data', s=3)
    # Plot the two temperatures identified as melting temperature
    ax.scatter(plate_onset, sample_onset, c='r', label='peak_onset', s=15)
    ax.scatter(plate_peak, sample_peak, c='b', label='peak_max', s=15)
    # create a secondary axis object to plot the 'Delta T' data
    ax1 = ax.twinx()
    ax1.scatter(raw_dataframe['Plate_avg'], raw_dataframe['Delta_T'],
                c='orange', label='$\Delta$ T', s=3)  # noqa: W605, W1401
    ax.set_xlabel('Plate Temperature [$^{\circ}$C]')  # noqa: W605, W1401
    ax.set_ylabel('Sample Temperature [$^{\circ}$C] ')  # noqa: W605, W1401
    ax.set_title('Temperature Profiles and Inflection Points')
    ax1.set_ylabel('$\Delta$ T [$^{\circ}$C]')  # noqa: W605,W1401

    data_0, labels_0 = ax.get_legend_handles_labels()
    data_1, labels_1 = ax1.get_legend_handles_labels()

    data = data_0 + data_1
    labels = labels_0 + labels_1

    ax.legend(data, labels, loc=0)
    return ax
