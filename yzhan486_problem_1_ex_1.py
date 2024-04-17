#Import all the libraries
import plotly.graph_objs as go
import tifffile
import numpy as np

## Part A: Write a script to play the data as a video

# Function to read frames from a .tif file
def read_tif_frames(tif_path):
      # Use tifffile to open and read the tif file
    with tifffile.TiffFile(tif_path) as tif:
        # Assuming the image data is in the first page/series
        # You might need to adjust this depending on your TIFF file structure
        frames = tif.asarray()


    return frames

# Path to your .tif file
tif_path = "/content/drive/MyDrive/Neural_Signals_and_Computation_yzhan486/TEST_MOVIE_00001-small-motion.tif"

# Read the frames from the .tif file
frames = read_tif_frames(tif_path)

frames = frames[0:100]
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