#Import all the libraries
import plotly.graph_objs as go
import tifffile
import numpy as np
from scipy.signal import correlate2d

### Function read_tif_frames and plot_video are for solving Problem 1A
# Function to read frames from a .tif file
def read_tif_frames(tif_path):
      # Use tifffile to open and read the tif file
    with tifffile.TiffFile(tif_path) as tif:
        # Assuming the image data is in the first page/series
        # You might need to adjust this depending on your TIFF file structure
        frames = tif.asarray()
    return frames

def plot_video(frames):
  # Setup initial frame
  fig = go.Figure(
      data=[go.Heatmap(z=frames[0])],
      layout=go.Layout(
          updatemenus=[dict(
              type="buttons",
              buttons=[dict(label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                       dict(label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])])],
          sliders=[{
              "steps": [{"args": [[f"frame{i}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                         "label": str(i), "method": "animate"} for i in range(len(frames))]}]
      ),
      frames=[go.Frame(data=[go.Heatmap(z=frame)], name=f"frame{i}") for i, frame in enumerate(frames)]
  )

  # Show the figure
  fig.show()


### Function compute_shifted_correlation is for solving Problem 1B
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
