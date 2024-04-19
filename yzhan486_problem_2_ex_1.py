## Problem 2 Summary images
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

### PROBLEM 2 PART A

def read_tif_frames(tif_path):
      # Use tifffile to open and read the tif file
    with tifffile.TiffFile(tif_path) as tif:
        # Assuming the image data is in the first page/series
        # You might need to adjust this depending on your TIFF file structure
        frames = tif.asarray()
    return frames

## Calculate mean, median and variance images
def calculate_summary_images(frames):
	mean_img = np.mean(frames,axis=0)
	median_img = np.median(frames, axis=0)
	variance_img = np.var(frames, axis=0)

	## Write into TIF image files and save them for future use
	tifffile.imwrite('mean_image.tif', mean_img.astype(np.float32))
	tifffile.imwrite('median_image.tif', median_img.astype(np.float32))
	tifffile.imwrite('variance_image.tif', variance_img.astype(np.float32))

	return mean_img, median_img, variance_img

def plot_summary_images(mean_img, median_img, variance_img):
	# Visualization
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))

	# Displaying the mean image
	axes[0].imshow(mean_img, cmap='gray')
	axes[0].set_title('Mean Image')
	axes[0].axis('off')  # Hide axis ticks and labels

	# Displaying the median image
	axes[1].imshow(median_img, cmap='gray')
	axes[1].set_title('Median Image')
	axes[1].axis('off')  # Hide axis ticks and labels

	# Displaying the variance image
	axes[2].imshow(variance_img, cmap='gray')
	axes[2].set_title('Variance Image')
	axes[2].axis('off')  # Hide axis ticks and labels

	# Show the summary images
	plt.tight_layout()
	plt.show()


### PROBLEM 2 PART B
def std_skew_images(frames):
##  I would like to try standard deviation and skewness

	std_dev_img = np.std(frames, axis=0)
	skewness_img = skew(frames, axis=0, bias=False)

	# Visualization using matplotlib
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))

	# Displaying the Standard Deviation image
	axes[0].imshow(std_dev_img, cmap='gray')
	axes[0].set_title('Standard Deviation Image')
	axes[0].axis('off')  # Hide axis ticks and labels

	# Displaying the Skewness image
	axes[1].imshow(skewness_img, cmap='gray')
	axes[1].set_title('Skewness Image')
	axes[1].axis('off')  # Hide axis ticks and labels

	plt.tight_layout()
	plt.show()

