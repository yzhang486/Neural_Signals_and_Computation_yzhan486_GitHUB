## Problem 4 find time traces

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import imageio
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.ndimage import label, generate_binary_structure
import matplotlib.pyplot as plt


def extract_time_traces(frames, roi_masks):
    """
    Extracts time traces of relative brightness for given ROIs across a sequence of frames.
    
    Parameters:
    - frames: A 3D NumPy array of shape (num_frames, height, width), representing the sequence of frames.
    - roi_masks: A list of 2D NumPy arrays, each of shape (height, width), representing the binary masks for each ROI.
    
    Returns:
    - A list of 1D NumPy arrays, each representing the time-trace of relative brightness for an ROI.
    """
    time_traces = []
    for mask in roi_masks:
        # Ensure the mask is boolean for indexing
        #mask = mask.astype(bool)
        # Extract the time-trace for the current ROI
        time_trace = [np.mean(frame[mask]) for frame in frames]
        time_traces.append(np.array(time_trace))
    
    return time_traces

