# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np


def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0)

# TODO: define functions to compute more features

def _compute_variance_features(window):
    return np.var(window, axis = 0)

def _compute_dft_features(window):
    arr = np.fft.rfft(window, axis = 0)
    temp = arr.astype(float)
    return temp[0]

def _compute_entropy_features(window):
    data = []
    for num in window:
        data.append(num)
    
    num_bins = 2

    hist, bin_edges = np.histogram(data , bins = num_bins)

    hist = hist/np.sum(hist)

    entropy = -np.sum(hist * np.log2(hist))

    return entropy

    

def _compute_peaks_features(window):
    upper_q = np.percentile(window , 75 , axis = 0)
    result = 0
    for arr in window:
        if(arr >= upper_q):
            result+=1
    return result



def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """

    """
    Statistical
    These include the mean, variance and the rate of zero- or mean-crossings. The
    minimum and maximum may be useful, as might the median
    
    FFT features
    use rfft() to get Discrete Fourier Transform
    
    Entropy
    Integrating acceleration
    
    Peak Features:
    Sometimes the count or location of peaks or troughs in the accelerometer signal can be
    an indicator of the type of activity being performed. This is basically what you did in
    assignment A1 to detect steps. Use the peak count over each window as a feature. Or
    try something like the average duration between peaks in a window.
    """

    #mag might not be needed !!!
    
    x = []
    feature_names = []
    win = np.array(window)

    #mean
    x.append(_compute_mean_features(win[:,0]))
    feature_names.append("x_mean")

    x.append(_compute_mean_features(win[:,1]))
    feature_names.append("y_mean")

    x.append(_compute_mean_features(win[:,2]))
    feature_names.append("z_mean")

    # x.append(_compute_mean_features(win[:,3]))
    # feature_names.append("mag_mean")


    #variance
    x.append(_compute_variance_features(win[:,0]))
    feature_names.append("x_variance")

    x.append(_compute_variance_features(win[:,1]))
    feature_names.append("y_variance")

    x.append(_compute_variance_features(win[:,2]))
    feature_names.append("z_variance")

    # x.append(_compute_variance_features(win[:,3]))
    # feature_names.append("mag_variance")

    #Discrete Fourier Transform
    x.append(_compute_dft_features(win[:,0]))
    feature_names.append("x_dft")

    x.append(_compute_dft_features(win[:,1]))
    feature_names.append("y_dft")

    x.append(_compute_dft_features(win[:,2]))
    feature_names.append("z_dft")

    # x.append(_compute_dft_features(win[:,3]))
    # feature_names.append("mag_dft")


    #entropy
    x.append(_compute_entropy_features(win[:,0]))
    feature_names.append("x_entropy")

    x.append(_compute_entropy_features(win[:,1]))
    feature_names.append("y_entropy")

    x.append(_compute_entropy_features(win[:,2]))
    feature_names.append("z_entropy")

    # x.append(_compute_entropy_features(win[:,3]))
    # feature_names.append("mag_entropy")

    #number of peaks
    x.append(_compute_peaks_features(win[:,0]))
    feature_names.append("x_peaks")

    x.append(_compute_peaks_features(win[:,1]))
    feature_names.append("y_peaks")

    x.append(_compute_peaks_features(win[:,2]))
    feature_names.append("z_peaks")

    # x.append(_compute_peaks_features(win[:,3]))
    # feature_names.append("mag_peaks")


    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names

    feature_vector = list(x)
    return feature_names, feature_vector

