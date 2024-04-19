import numpy as np
import tifffile as tiff
from sklearn.decomposition import PCA, NMF, FastICA
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


def plot_results(values, title='Component Analysis', xlabel='Number of Components', ylabel='Value', plot_type='line'):
    plt.figure()
    if plot_type == 'line':
        plt.plot(values)
    elif plot_type == 'bar':
        plt.bar(range(len(values)), values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()
