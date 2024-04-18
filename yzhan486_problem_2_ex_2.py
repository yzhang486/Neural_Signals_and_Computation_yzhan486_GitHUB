## Part B
from scipy.stats import skew
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

# Path to your .tif file
tif_path = "/content/drive/MyDrive/Neural_Signals_and_Computation_yzhan486/TEST_MOVIE_00001-small.tif"

# Read the frames from the .tif file
frames = read_tif_frames(tif_path)

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