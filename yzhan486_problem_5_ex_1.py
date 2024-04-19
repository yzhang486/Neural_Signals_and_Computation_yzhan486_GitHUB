import numpy as np
import tifffile as tiff
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_and_vectorize_tiff(file_path):
    with tiff.TiffFile(file_path) as tif:
        images = tif.asarray()  # Load all frames; shape (frames, height, width)
    num_frames, height, width = images.shape
    reshaped_images = images.reshape(num_frames, height * width)  # Reshape to (frames, pixels)
    return reshaped_images

def perform_pca(data, n_components=5):
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    return transformed_data, explained_variance, pca

def plot_results(metrics, title, ylabel):
    plt.figure(figsize=(10, 5))
    for key, values in metrics.items():
        plt.plot(range(1, len(values)+1), values, label=key)
    plt.xlabel('Number of Components')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def perform_nmf(data, n_components=5):
    model = NMF(n_components=n_components, init='random', random_state=0)
    W = model.fit_transform(data)  # Basis vectors
    H = model.components_  # Coefficients
    return W, H, model

def perform_ica(data, n_components=5):
    ica = FastICA(n_components=n_components, random_state=0)
    S = ica.fit_transform(data)  # Independent components
    A = ica.mixing_  # Mixing matrix
    return S, A, ica
