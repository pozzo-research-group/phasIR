
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from .irtemp import centikelvin_to_celsius
from skimage.draw import circle
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
    r: Int
       Integer value to use for the radius of the circle applied to each
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
            rr, cc = circle(row[i], col[i], radius=r)
            sample_intensity = np.mean(frame[rr, cc])
            plate_intensity = np.mean(
                frame[p_row[i*4:(i*4)+4], p_col[i*4:(i*4)+4]])
            temp_well.append(centikelvin_to_celsius(sample_intensity))
            plate_well_temp.append(centikelvin_to_celsius(plate_intensity))
        sample_temp.append(temp_well)
        plate_temp.append(plate_well_temp)
    return sample_temp, plate_temp


def find_temp_peak(baseline_array, height=2):
    '''

    Parameters
    ----------

    Returns
    -------

    '''
    peaks, properties = find_peaks(baseline_array, height=height)
    peak_heights = properties['peak_heights']
    peak_max = [i for i in range(len(peak_heights))
                if peak_heights[i] == max(peak_heights)]
    peaks_w = peak_widths(baseline_array, peaks[peak_max])

    peak_left_onset = int(round(peaks_w[2][0]))

    return peak_left_onset, peaks[peak_max]


def get_temperature(dataframe, peak_onset_index, peak_max_index, sample=True):
    """
    Parameters
    ----------

    Returns
    -------
    """
    if sample:
        onset_temp = dataframe['Sample_avg'].loc[peak_onset_index]
        peak_temp = dataframe['Sample_avg'].loc[peak_max_index[0]]
    else:
        onset_temp = dataframe['Plate_avg'].loc[peak_onset_index]
        peak_temp = dataframe['Plate_avg'].loc[peak_max_index[0]]

    return onset_temp, peak_temp


def baseline_subtraction(plate_temperature, sample_temperature, n=None):
    """
    Parameters
    ----------

    Returns
    -------
    """
    dataframe = pd.DataFrame({'Frames': np.linspace(1, len(plate_temperature),
                                                    len(plate_temperature)),
                              'Plate_temp': plate_temperature,
                              'Sample_temp':  sample_temperature})
    if n:
        n = n
    else:
        n = int(0.05*len(dataframe['Plate_temp']))
    dataframe['Plate_avg'] = dataframe.iloc[:, 1].rolling(window=n).mean()
    dataframe['Sample_avg'] = dataframe.iloc[:, 2].rolling(window=n).mean()
    dataframe['Delta_T'] = dataframe['Plate_avg']-dataframe['Sample_avg']

    return dataframe


def phase_transition_temperature(plate_temperatures, sample_temperatures,
                                 plot=False):
    """
    Parameters
    ----------

    Returns
    -------

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
        stemp_onset.append(np.round(s_temp_onset, 2))
        stemp_peak.append(np.round(s_temp_peak, 2))
        ptemp_onset.append(np.round(p_temp_onset, 2))
        ptemp_peak.append(np.round(p_temp_peak, 2))
        if plot:
            visualize_results(
                dataframe, p_temp_onset, s_temp_onset,
                p_temp_peak, s_temp_peak)

    temperatures_dataframe = pd.DataFrame({'Sample_temp_onset': stemp_onset,
                                           'Sample_temp_peak': stemp_peak,
                                           'Plate_temp_onset': ptemp_onset,
                                           'Plate_temp_peak': ptemp_peak})

    return temperatures_dataframe


def visualize_results(raw_dataframe, plate_onset, sample_onset,
                      plate_peak, sample_peak):
    """

    Parameters
    ----------

    Returns
    -------
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
