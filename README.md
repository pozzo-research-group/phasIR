# phasIR
Python modules for high-throughput measurement of melting point using IR bolometry


        **_Note_**: the package is currently under development. New functions are being added and new
        modules will be created and added in the next few days. This package is a second version of
        [`musical robot `](https://github.com/pozzocapstone/musical-robot.git). The following readme
        still needs to be edited and new information will be added. Stay tuned!


![Github musical robot image](https://user-images.githubusercontent.com/46472196/60206415-51243b00-9808-11e9-9668-d65843ce377d.png)

## Introduction
Deep eutectic solvents (DES) are novel solvents that can be easily produced at low-cost for several important applications, such as chemical synthesis, extractions, electrochemistry, and even pharmaceutical drug delivery. The design space for DES is enormous and high throughput measurement of melting points is required to rapidly identify DES with melting points that are feasible for their specific application. High throughput measurement of melting points was made possible through the use of an infrared camera and subsequent image analysis. The melting point of the DES was obtained by recording the temperature profile of the sample as it was heated, and locating the inflection point in the profile that results from an increase in thermal conductivity as the sample melts.  The python package developed is able to obtain accurate melting points of multiple samples at once, while only requiring a matter of minutes to perform the physical measurement, and at low-cost. In contrast, standard melting point determination techniques utilize equipment that is orders of magnitude more expensive and can take up to an hour for individual samples.

## Installation
* Currently this package can be installed using the following command:
` python setup.py install`
    Note: you will have to first `cd your-path/phasIR`


## Usage

#### The python package adopts the following two techniques to obtain the temperature profile of the samples and sample holder to determine the melting point of the samples:

1. Temperature profile through edge detection

* This method can be used for images(video frames) with high contrast and minimal noise which will allow for detection of edges of just the samples.
* The temperature profile of the samples and plate is determined by detecting the edges, filling and labeling them, and monitoring the temperature at their centroids.
* This technique can be adapted by using the functions `input_file` and `centroid_temp` from the `musicalrobot.edge_detection` module to load the recorded video and obtain the temperature profile of the samples and sample holder.

2. Temperature profile through pixel value analysis.

* This is an alternative technique for low contrast images(video frames). In some situations, the contrast between the image and sample maybe too low for edge detection, even with contrast enhancement.
* Alternatively, centroid location for each sample can be found by summing pixel values over individual rows and columns of the sample holder(well plate).
* This technique can be adapted by using the functions `input_file` and `pixel_temp` from the `musicalrobot.pixel_analysis` module to load the recorded video and obtain the temperature profile of the samples and sample holder.

#### Melting point determination

* An inflection is observed at the melting point in the temperature profile of the samples due to the following reasons
1. Change in thermal conductivity of the sample
2. Increase in thermal contact between the sample and the well plate
* The point of inflection in the temperature profile is determined by detecting the peak in the second derivative of the temperature profile. Since an analytical approach is used to determine the melting point, the temperature profile plots have to classified to eliminate plots with noise and without an inflection in them.
* The module `musicalrobot.data_encoding` can be used to classify the temperature profiles.

An example of adapting the above mentioned modules and functions using the `musicalrobot` package can be found in the ipython notebook `Tutorial.ipynb` found in the examples folder.

## For Development
* Install python version 3.6
* Clone the repository on your machine using git clone https://github.com/pozzo-research-group/phasIR.git . This will create a copy of this repository on your machine.
* Go to the repository folder using cd musical-robot.
* Install the python dependencies by using pip install -r requirements.txt
