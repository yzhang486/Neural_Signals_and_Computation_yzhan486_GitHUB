## Import libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from numpy.random import poisson
from scipy.optimize import minimize
from numpy.random import normal, poisson
from scipy.stats import poisson as poisson_dist
from scipy.optimize import minimize
from numpy.linalg import norm

def gausswin(N, alpha):
    # This function is an approximation of MATLAB's gausswin
    return gaussian(N, std=alpha*N/10)

def custom_gausswin(N, alpha=2.5):
    # Create a customized Gaussian window similar to MATLAB's gausswin
    std = alpha * N / 10
    return gaussian(N, std=std, sym=False) * np.cos(2 * np.pi * np.arange(N) / 10)

# Define the negative log-likelihood function
def negLogLikelihood(g, X, r):
    lambda_exp = np.exp(X.T @ g)
    return -np.sum(r * np.log(lambda_exp) - lambda_exp)


def negLogPosterior(g, X, response, sigma_g):
    Xg = X.T @ g
    regularization = (1/(2*sigma_g**2)) * np.sum(np.diff(g)**2)
    return -np.sum(response * Xg - np.exp(Xg)) + regularization


def experiment_with_parameters(N, M, A, sigma_g):
    sigma = 0.1  # Standard deviation of Gaussian noise in the model

    # Generate the true tuning curve g
    g_true = custom_gausswin(N, 5)

    # Generate random stimuli and scale by A
    X = A * 2 * np.random.rand(N, M)

    # Generate responses based on the Poisson model
    lambda_ = np.exp(X.T @ g_true)
    responses = poisson(lambda_)

    # Define the negative log-posterior for the Poisson case with smoothing prior
    def negLogPosterior(g):
        Xg = X.T @ g
        regularization = (1 / (2 * sigma_g ** 2)) * np.sum(np.diff(g) ** 2)
        return -np.sum(responses * np.log(np.exp(Xg)) - np.exp(Xg)) + regularization

    # Initial guess for g
    g_initial = np.zeros(N)

    # Perform optimization
    result = minimize(negLogPosterior, g_initial, method='BFGS')
    g_estimated = result.x

    # Plot true and estimated g for comparison
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(g_true, 'b-', label='True g')
    axs[0].plot(g_estimated, 'r--', label='Estimated g (MAP with smoothing)')
    axs[0].legend()
    axs[0].set_title(f'Estimation with A = {A:.2f}, M = {M}, N = {N}, sigma_g = {sigma_g:.2f}')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Tuning Curve Value')

    # Plot histogram of the spike counts
    axs[1].hist(responses, bins='auto', density=True, alpha=0.7, color='g')
    axs[1].set_title(f'Histogram of Spike Counts (Responses) with A = {A:.2f}, M = {M}')
    axs[1].set_xlabel('Spike Counts')
    axs[1].set_ylabel('Probability Density')

    plt.tight_layout()
    plt.show()
