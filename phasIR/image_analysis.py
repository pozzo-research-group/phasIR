import os
import numpy as np
import pandas as pd
import random

from skimage import io, feature, filters
from skimage.measure import label, regionprops
from scipy.ndimage.morphology import binary_fill_holes, remove_small_objects


# Function to load the input file
def input_file(file_name):
    '''
    To load the input video as an array.

    Parameters
    -----------
    file_name : String
        Name of the file to be loaded as it is saved on the disk.
        Provide file path if it is not in the same directory as
        the jupyter notebook.

    Returns
    --------
    frames : array
        In case of a video, returns an array for each frame
        in the video. In case of an image, return an array.
    '''
    frames = io.imread(file_name)
    return frames


# Function to flip the frames horizontally and vertically to correct
# for the mirroring during recording.
def flip_frame(frames):
    '''
    To flip all the loaded frames horizontally and vertically
    to correct for the mirroring effect during recording.

    Parameters
    -----------
    frames : Array
        An array containing an array for each frame
        in the video or just a single array in case of an image.

    Returns
    --------
    flip_frames : Array
        Flipped frames that can be processed to get temperature data.
    '''
    flip_frames = []
    for frame in frames:
        f_frame = np.fliplr(frame)
        flip_frames.append(np.flipud(f_frame))
    return flip_frames


def edge_detection(frame, n_samples, method='canny', sigma=1):
    """
    Function to detect the edges of the wells, fill and label them to
    determine their centroids.

    Parameters
    -----------
    frames : Array
        The frames to be processed and determine the
        sample temperature from.
    n_samples : Int
        The number of samples in the input video.
    method : String
        Edge detection algorithm to be used
    sigma : float
        Standard deviation of the Gaussian filter applied during the
        skimage.feature.canny method to detect the wells on the plates.

    Returns
    --------
    samples : Array
        Array containing all samples in the frame stored as skimage objects.
    """

    # Define placeholder variables
    broken = False

    # use canny edge detection method
    if method is 'canny':
        for size in range(15, 9, -1):
            image = frame - frame.min()
            image = image/image.max()

            edges = feature.canny(image, sigma=sigma)

            filled_samples = binary_fill_holes(edges)
            cleaned_samples = \
                remove_small_objects(filled_samples, min_size=size)
            labeled_samples = label(cleaned_samples)
            samples = \
                regionprops(labeled_samples, intensity_image=frame)

            # Check that the number of objects found in the image is not
            # higher than the number of wells.
            # If higher, increase the threshold [thres] even further.
            if len(samples) > n_samples:
                broken = True

            if len(samples) == n_samples:
                break

    elif method is 'sobel':
        for size in range(15, 9, -1):
            # use sobel
            edges = filters.sobel(frame)
            edges = edges > edges.mean() * 2  # booleanize data
            filled_samples = binary_fill_holes(edges)
            cleaned_samples = \
                remove_small_objects(filled_samples, min_size=size)
            labeled_samples = label(cleaned_samples)
            samples = \
                regionprops(labeled_samples, intensity_image=frame)

            # Check that the number of objects found in the image is not
            # higher than the number of wells.
            # If higher, increase the threshold [thres] even further.
            if len(samples) > n_samples:
                broken = True

            if len(samples) == n_samples:
                break

    if broken:
        print('The number of objects found on the image is higher than' +
              ' the sample number provided ', n_samples)

    return samples


def sample_locations(samples, n_samples, dtype='float'):
    """
    Function used to extract the centroid coordinates of each well.

    Paramters
    ---------
    samples: skimage.measure._regionprops.RegionProperties
        skimage object containing the properties of each well found
        using the edge detection function. The centroid of the well
        is an attribute of this object.
    n_samples: int
        Integer number representing the number of samples in a plate.
        This number shuold match the number fo wells

    Returns
    -------
    sample_location : pd.DataFrame
        DataFrame containing the sample centroid's coordinates
        ordered column wise.
    """
    # Extract centroid of the samples.
    sample_col = []
    sample_row = []
    if dtype == 'int':
        for i in range(len(samples)):
            sample_col.append(int(round(samples[i].centroid[1])))
            sample_row.append(int(round(samples[i].centroid[0])))
    elif dtype == 'float':
        for i in range(len(samples)):
            sample_col.append(samples[i].centroid[1])
            sample_row.append(samples[i].centroid[0])

    # Save the sample coordinates and order them
    # Create a unique index to help with the sorting of the samples
    unique_index = random.sample(range(100), n_samples)
    location_dict = {'Column': sample_col, 'Row': sample_row,
                     'unique_index': unique_index}
    # Convert dictionary in a DataFrame and sort it
    sample_location = pd.DataFrame(location_dict)
    sample_location.sort_values(['Column', 'Row'],
                                inplace=True, ignore_index=True)

    return sample_location


def sort_samples(sample_location, n_columns, n_rows):
    '''
    Function to sort the samples location to match the order in which
    the samples are pipetted.

    Parameters
    ------------
    regprops : Dict
        A dictionary of dataframes containing information about the sample.
    n_columns : Int
        Number of columns of samples
    n_rows : Int
        Number of rows of samples

    Returns
    --------
    sorted_location : Dict
        A dataframe with information about samples. The order of the samples
        is sorted from top to bottom and from left to right, following the
        pipetting order.
    '''
    sorted_rows = []
    # Sorting the dataframe according to the row coordinate in each column.
    # The samples are pipetted out top to bottom from left to right.
    # The order of the samples in the dataframe
    # should match the order of pipetting.
    for j in range(0, n_columns):
        sorted_df = sample_location[j*n_rows:(j+1)*n_rows].sort_values(['Row'])
        sorted_rows.append(sorted_df)
    new_location = pd.concat(sorted_rows)
    # Creating an index to be used for reordering all the dataframes.
    # The unique index is the sum of row and column coordinates.
    reorder_index = new_location.unique_index
    sample_location.set_index('unique_index', inplace=True)
    sorted_location = sample_location.reindex(reorder_index)

    sorted_location = sorted_location.reset_index(drop=True)
    return sorted_location


def diagonal_points(x_coord, y_coord, radius):
    """
    Function to generate four diagonla points given a center and a radius

    Paramters
    ---------
    x_coord: float or int
        X-coordinates of center point
    y_coord: float or int
        Y-coordinates of center point
    radius: int or float
        Distance to add and subtract to the both coordinates of the center
        point to obtain the four diagonal points

    Returns
    -------
    x_coords: list
        List containing the X-coordinates of the diagonal points
    y_coords: list
        List containing the Y-coordinates of the diagonal points
    """

    x_coords = [np.round(x_coord+radius), np.round(x_coord-radius),
                np.round(x_coord+radius), np.round(x_coord-radius)]
    y_coords = [np.round(y_coord+radius), np.round(y_coord+radius),
                np.round(y_coord-radius), np.round(y_coord-radius)]

    return x_coords, y_coords


def plate_location(sample_location, n_columns, n_rows):
    """
    Function to obtain the coordinate of the plate points to use for each
    sample. The plate points are the four diagonal coordinates for each
    well.

    Parameters
    ----------
    sample_location: pd.DataFrame
        Dataframe containing the coordinates of the each sample on the plate
    n_columns: int
        Number representing the columns of wells composng the wellplate
    n_row: int
        Number representing the rows of wells composng the wellplate

    Returns
    -------
    plate_coordinates: pd.DataFrame
        Dataframe containig the coordinate of the plate to use for the
        evaluation of the melting point of each sample.
    """
    plate_x_coords = []
    plate_y_coords = []

    rows = np.array(sample_location['Row'])
    columns = np.array(sample_location['Column'])

    for i in range(len(sample_location)):
        if i >= len(sample_location)-(n_rows+1):
            radius = (columns[i]-columns[i-n_rows])/2
        else:
            radius = (columns[i+n_rows]-columns[i])/2
        diagonal_x, diagonal_y = diagonal_points(columns[i], rows[i], radius)
        plate_x_coords.append(diagonal_x)
        plate_y_coords.append(diagonal_y)

    plate_x = [float(x) for x in np.array(plate_x_coords).reshape(-1, 1)]
    plate_y = [float(y) for y in np.array(plate_y_coords).reshape(-1, 1)]

    plate_coordinates = pd.DataFrame({'Plate_col': plate_x,
                                      'Plate_row': plate_y}, dtype=int)
    return plate_coordinates


def manual_centroid(image):
    """
    Function to manually identify the centroid of the wells on a plate.

    Function will require user input to select the coordinates of
    each centroid. Left click to select, middle click to remove last point,
    and right click to finish manual input. Finally, using a key on
    the keyboard, the coordinates will eb save in a dataframe, already
    sorted following the same order they were inputted.
    Note: suggested order-> top-bottom, left-right


    Parameters
    ----------
    image: np.ndarray
        Array representing an image. Thsi image will be use to manually
        select the centroid position of each well.

    Returns
    -------
    sample_location: pd.DataFrame
        Dataframe containing the coordinates of each centroid identified on
        the plate.

    """

    # Set preferred matplolib visualization as pop out window
    %matplotlib qt  # noqa: E999

    prompt = 'Select the centroid of each well. Right click once' + \
        'done. Middle mouse button removes most recent point.'
    message = "Press keyboard button to save points and exit."
    print(prompt)

    # generate image
    fig, ax = plt.subplots()
    plt.setp(plt.gca(), autoscale_on=True)
    ax.imshow(image)

    # create empty lists to collect the centroid coordinates
    sample_col = []
    sample_row = []
    while True:
        plt.title(prompt, wrap=True)
        fig.canvas.draw()
        while True:
            # This command allows the user to manually select any point on
            # the image. n is set to a negative number, indicating that there
            # is no limit for how many centroid to select. Selection is
            # interrupted using the right click of the mouse.
            points = plt.ginput(n=-1, show_clicks=True, timeout=-1,
                                mouse_add=1, mouse_stop=3, mouse_pop=2)
            break
        plt.title(message, wrap=True)
        fig.canvas.draw()
        # print("Saved points = ", points)
        print(message)
        # Extract coordinates of centroids.
        if plt.waitforbuttonpress():
            for i in range(len(points)):
                point_col = int(np.round(points[i][0]))
                point_row = int(np.round(points[i][1]))
                sample_col.append(point_col)
                sample_row.append(point_row)
            plt.close()
            break

    # Reset preferred matplolib visualization as inline
    %matplotlib inline

    # save the centroid location in a dataframes
    location_dict = {'Column': sample_col, 'Row': sample_row}
    sample_location = pd.DataFrame(location_dict)

    return sample_location
