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
    # Time trace initialization
    time_traces = []
    for mask in roi_masks:
        # Ensure the mask is boolean for indexing
        #mask = mask.astype(bool)
        # Extract the time-trace for the current ROI
        time_trace = [np.mean(frame[mask]) for frame in frames]
        time_traces.append(np.array(time_trace))
    
    return time_traces


def plot_time_traces(time_traces):
    # Generate sample data: 5 rows, 500 columns
    data = time_traces

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Colors for different curves
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    curve_names = ['ROI1','ROI2','ROI3','ROI4','ROI5']
    # Plot each row of the array on the same graph
    for i in range(np.shape(time_traces)[0]):
        plt.plot(data[i], color=colors[i], label=curve_names[i])

    # Add title and labels
    plt.title('Time traces of the 5 ROIs')
    plt.xlabel('Time')
    plt.ylabel('Pixel intensities')
    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

    
