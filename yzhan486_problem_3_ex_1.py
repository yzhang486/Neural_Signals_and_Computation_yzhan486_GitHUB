## Problem 3

import numpy as np
import imageio
from skimage.filters import threshold_otsu
from scipy.ndimage import label, generate_binary_structure
from skimage.measure import regionprops
import matplotlib.pyplot as plt

def plot_roi(image):
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
	    
	    # Visualization
	    axes[i].imshow(roi_mask, cmap='gray')
	    axes[i].set_title(f'ROI {i+1}')
	    axes[i].axis('off')

	plt.tight_layout()
	plt.show()

	return roi_masks