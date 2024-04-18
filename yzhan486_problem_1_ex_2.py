import plotly.graph_objs as go
import tifffile
import numpy as np
import numpy as np
from scipy.signal import correlate2d

# Function to read frames from a .tif file
def read_tif_frames(tif_path):
      # Use tifffile to open and read the tif file
    with tifffile.TiffFile(tif_path) as tif:
        # Assuming the image data is in the first page/series
        # You might need to adjust this depending on your TIFF file structure
        frames = tif.asarray()


    return frames

def compute_shifted_correlation(frame1, frame2, max_shift):
   
    # Initialize an array to hold the correlation results
    correlation_map = np.zeros((2*max_shift+1, 2*max_shift+1))
    
    # Loop over all possible shifts within the specified range
    for dy in range(-max_shift, max_shift+1):
        for dx in range(-max_shift, max_shift+1):
            # Shift frame2 relative to frame1
            shifted_frame2 = np.roll(frame2, shift=(dy, dx), axis=(0, 1))
            
            # Compute correlation coefficient for this shift
            correlation = correlate2d(frame1, shifted_frame2, mode='valid')
            
            # Store the correlation result
            correlation_map[dy+max_shift, dx+max_shift] = correlation
            
    return correlation_map

tif_path = "/content/drive/MyDrive/Neural_Signals_and_Computation_yzhan486/TEST_MOVIE_00001-small-motion.tif"
frames = read_tif_frames(tif_path)


# Initialization
frame1 = frames[1]  # First frame data here
frame2 = frames[200]  # Second frame data here
max_shift = 10  # Maximum shift in pixels

correlation_map = compute_shifted_correlation(frame1, frame2, max_shift)

# To find the shift with the maximum correlation
max_correlation_value = np.max(correlation_map)
peak_idx = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
print(f"Maximum correlation value: {max_correlation_value}")
print(f"Location of the correlation peak (shift_y, shift_x): {peak_idx}")
