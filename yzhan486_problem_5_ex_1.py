import numpy as np
import tifffile as tiff
from sklearn.decomposition import PCA, NMF, FastICA
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage import io, filters, measure, color, morphology

def load_and_vectorize_tiff(file_path):
    with tiff.TiffFile(file_path) as tif:
        images = tif.asarray()  # Load all frames; shape (frames, height, width)
    num_frames, height, width = images.shape
    reshaped_images = images.reshape(num_frames, height * width)  # Reshape to (frames, pixels)
    return images,reshaped_images, height, width

### Three functions performing PCA, NMF and ICA repsectively here:
def perform_pca(pixel_matrix,target_components):
    pca = PCA(n_components=target_components)  # Adjust based on how many components you want to consider
    pca.fit(pixel_matrix)
    components = pca.components_
    return components

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

##########################################################################


def plot_roi(image):
# Apply Otsu's thresholding
    thresh = threshold_otsu(image)
    binary_image = image > thresh
    # Combine the masks, eliminating the gray region
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
            axes[i].imshow(roi_mask)
            axes[i].set_title(f'ROI {i+1}')
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    return roi_masks


def plot_images(images, title):
    plt.figure(figsize=(10, 10))
    for i, image in enumerate(images, 1):
        plt.subplot(len(images),1, i)
        plt.imshow(image)
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


def plot_component_images(components_imgs):
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) #Plot top 5 components always

    # Displaying the img 1
    axes[0].imshow(components_imgs[0].reshape(height, width), cmap='gray')
    axes[0].set_title('Component 1')
    axes[0].axis('off')  # Hide axis ticks and labels

    # Displaying the median image
    axes[1].imshow(components_imgs[1].reshape(height, width), cmap='gray')
    axes[1].set_title('Component 2')
    axes[1].axis('off')  # Hide axis ticks and labels

    # Displaying the variance image
    axes[2].imshow(components_imgs[2].reshape(height, width), cmap='gray')
    axes[2].set_title('Component 3')
    axes[2].axis('off')  # Hide axis ticks and labels

    # Show the summary images
    plt.tight_layout()
    plt.show()
 