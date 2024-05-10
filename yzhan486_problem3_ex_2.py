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
from scipy.linalg import expm  
import jPCA
from jPCA.util import load_churchland_data, plot_projections

# Function to normalize the initial state to prevent NAN being generated
def normalize_initial_state(initial_state):
    norm = np.linalg.norm(initial_state)
    return initial_state / norm if norm != 0 else initial_state

# Function to extrapolate dynamics using matrix exponentiation for jPCA
def extrapolate_dynamics_jPCA(initial_state, A_jpca, num_steps, dt):
    initial_state = normalize_initial_state(initial_state)
    trajectory = [initial_state]
    scaling_factor = 1e-3 #ensure initial state nonzero
    skew_symm_exp = expm(A_jpca * dt * scaling_factor) #Take skewness into consideration
    for _ in range(num_steps - 1):
        next_state = trajectory[-1] @ skew_symm_exp
        trajectory.append(next_state)
    return np.array(trajectory)



def compare_projections(part_d_trajectories, part_e_trajectories, title_d='Part D (PCA)', title_e='Part E (jPCA)'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Part D Plot
    for trajectory in part_d_trajectories:
        pca = PCA(n_components=2)
        trajectory_2d = pca.fit_transform(trajectory)
        axes[0].plot(trajectory_2d[:, 0], trajectory_2d[:, 1], marker='o')
    axes[0].set_title(title_d)
    axes[0].set_xlabel('Principal Dimension 1')
    axes[0].set_ylabel('Principal Dimension 2')
    axes[0].grid(True)

    # Part E Plot
    for trajectory in part_e_trajectories:
        pca = PCA(n_components=2)
        trajectory_2d = pca.fit_transform(trajectory)
        axes[1].plot(trajectory_2d[:, 0], trajectory_2d[:, 1], marker='o')
    axes[1].set_title(title_e)
    axes[1].set_xlabel('Principal Dimension 1')
    axes[1].set_ylabel('Principal Dimension 2')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

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
