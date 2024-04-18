## Problem 4 find time traces

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import imageio
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.ndimage import label, generate_binary_structure
import matplotlib.pyplot as plt


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



# Load the variance image
image_path = 'variance_image.tif'
image = imageio.imread(image_path)

# Apply Otsu's thresholding
thresh = threshold_otsu(image)
binary_image = image > thresh

# Label connected components
structure = generate_binary_structure(2, 2)
labeled_array, num_features = label(binary_image, structure=structure)

# Select the 5 largest ROIs based on area
props = regionprops(labeled_array)
props.sort(key=lambda x: x.area, reverse=True)
largest_props = props[:5]

# Initialize a list to hold the binary masks for each ROI
roi_masks = []

# Generate and visualize a binary mask for each ROI
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, prop in enumerate(largest_props):
    # Create a binary mask for the ROI
    roi_mask = np.zeros_like(image, dtype=bool)
    roi_mask[labeled_array == prop.label] = True
    roi_masks.append(roi_mask)



# Path to your .tif file
tif_path = "/content/drive/MyDrive/Neural_Signals_and_Computation_yzhan486/TEST_MOVIE_00001-small.tif"

# Read the frames from the .tif file
frames = read_tif_frames(tif_path)

time_traces = extract_time_traces(frames, roi_masks)

print(time_traces)
plt.figure(figsize=(10,5))
plt.plot(time_traces)