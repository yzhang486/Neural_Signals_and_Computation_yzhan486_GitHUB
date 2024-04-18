## Problem 4 find time traces

import numpy as np
import matplotlib.pyplot as plt
import tifffile



def read_tif_frames(tif_path):
      # Use tifffile to open and read the tif file
    with tifffile.TiffFile(tif_path) as tif:
        # Assuming the image data is in the first page/series
        # You might need to adjust this depending on your TIFF file structure
        frames = tif.asarray()


    return frames


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


# Path to your .tif file
tif_path = "TEST_MOVIE_00001-small.tif"

# Read the frames from the .tif file
frames = read_tif_frames(tif_path)

time_traces = extract_time_traces(frames, roi_masks)