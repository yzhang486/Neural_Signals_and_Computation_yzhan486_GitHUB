## PART A libraries
import scipy.io as sio
import seaborn as sns
import matplotlib.pyplot as plt


## Functions in part B
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

#Compute the time step between consecutive points
def compute_time_steps(times):
    return np.mean(np.diff(times))
    
# Define function to compute the Maximum Likelihood Estimation (MLE) of matrix A
def estimate_A(datasets, time_list):
    sum_xt_xt1 = np.zeros((datasets[0].shape[1], datasets[0].shape[1]))
    sum_xt_xt = np.zeros((datasets[0].shape[1], datasets[0].shape[1]))
    
    for dataset, times in zip(datasets, time_list):
        dt = compute_time_steps(times)
        X_t = dataset[:-1, :]
        X_t1 = dataset[1:, :]
        sum_xt_xt1 += X_t.T @ X_t1
        sum_xt_xt += X_t.T @ X_t
    
    A_estimated = sum_xt_xt1 @ np.linalg.inv(sum_xt_xt)
    return A_estimated


# Function to predict the next state given current state and estimated A
def predict_next_state(x_t, A_estimated, dt):
    return x_t @ (np.eye(A_estimated.shape[0]) + A_estimated * dt)


## Functions in part C & D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.io import loadmat
# Function to compute the time step between consecutive points


# Function to apply PCA and keep top n components
def apply_pca(X, n_components=6):
    pca = PCA(n_components=n_components)
    X_centered = X - X.mean(axis=0)
    X_pca = pca.fit_transform(X_centered)
    return X_pca, pca

# Function to compute the Maximum Likelihood Estimation (MLE) of matrix A
def estimate_A_pca(datasets_pca):
    n_components = datasets_pca[0].shape[1]
    sum_xt_xt1 = np.zeros((n_components, n_components))
    sum_xt_xt = np.zeros((n_components, n_components))
    
    for X_pca in datasets_pca:
        X_t = X_pca[:-1, :]
        X_t1 = X_pca[1:, :]
        sum_xt_xt1 += X_t.T @ X_t1
        sum_xt_xt += X_t.T @ X_t
    
    A_estimated = sum_xt_xt1 @ np.linalg.inv(sum_xt_xt)
    return A_estimated

# Function to predict the next state given current state and estimated A
def predict_next_state_pca(x_t, A_estimated, dt):
    return x_t @ (np.eye(A_estimated.shape[0]) + A_estimated * dt)

# Function to extrapolate the neural system forward in time
def extrapolate_dynamics(initial_state, A_estimated, num_steps, dt):
    trajectory = [initial_state]
    for _ in range(num_steps - 1):
        next_state = predict_next_state_pca(trajectory[-1], A_estimated, dt)
        trajectory.append(next_state)
    return np.array(trajectory)


## Functions in part E
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.io import loadmat
import jPCA
from jPCA.util import load_churchland_data, plot_projections


def extrapolate_dynamics_jPCA(initial_state, A_jpca, num_steps, dt):
    trajectory = [initial_state]
    for _ in range(num_steps - 1):
        next_state = trajectory[-1] @ (np.eye(A_jpca.shape[1]) + A_jpca.T * dt)
        trajectory.append(next_state)
    return np.array(trajectory)



def compare_projections(part_d_trajectories, part_e_trajectories, title_d='Part D (PCA)', title_e='Part E (jPCA)'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))


# Compare Eigenspectra
def plot_eigenspectra(A1, A2, title1='Matrix 1', title2='Matrix 2'):
    eigvals1 = np.linalg.eigvals(A1)
    eigvals2 = np.linalg.eigvals(A2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(np.real(eigvals1), np.imag(eigvals1))
    axes[0].set_title(title1)
    axes[0].set_xlabel('Real Part')
    axes[0].set_ylabel('Imaginary Part')
    axes[0].grid(True)

    axes[1].scatter(np.real(eigvals2), np.imag(eigvals2))
    axes[1].set_title(title2)
    axes[1].set_xlabel('Real Part')
    axes[1].set_ylabel('Imaginary Part')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()