## Problem 3 Drawing Regions of Interests (ROIs)


import numpy as np
import imageio
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.pyplot as plt

# Load the variance image
variance_image_path = 'variance_image.tif'
image = imageio.imread(variance_image_path)

# Apply Otsu's thresholding
thresh = threshold_otsu(image)
binary_image = image > thresh

# Label connected components
labeled_image = label(binary_image)
regions = regionprops(labeled_image)

# Select the 5 largest connected components based on area
regions_sorted_by_area = sorted(regions, key=lambda x: x.area, reverse=True)
largest_regions = regions_sorted_by_area[:5]

# Create an empty image to store the ROIs
rois_image = np.zeros_like(image, dtype=np.uint8)

# Fill the ROIs image with the labels of the 5 largest regions
for i, region in enumerate(largest_regions, start=1):
    for coord in region.coords:
        rois_image[coord[0], coord[1]] = i

# Visualize the original image with ROIs outlined
plt.figure(figsize=(10, 5))
plt.imshow(image, cmap='gray')
plt.imshow(label2rgb(rois_image, bg_label=0, bg_color=None, colors=['red', 'green', 'blue', 'yellow', 'cyan']), alpha=0.5)
plt.title('Variance Image with 5 ROIs Outlined')
plt.axis('off')
plt.show()