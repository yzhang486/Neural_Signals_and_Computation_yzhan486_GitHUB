import numpy as np
from scipy.signal import correlate2d

# Function to read frames from a .tif file

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

