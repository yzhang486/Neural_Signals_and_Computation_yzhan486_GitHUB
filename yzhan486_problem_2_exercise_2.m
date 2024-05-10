% Problem 2:Dimensionality reduction

%% Part A

% Part A: Load the data and compute PSTH

% Load data
data = load('sample_dat.mat');

% Each trial matrix is of size N (neurons) by T (time bins)
S = data.dat;
sample_trial = S(1).spikes;

% Initialize parameters
N = size(sample_trial, 1); % Number of neurons
T = size(sample_trial, 2); % Number of time bins
num_trials = length(S);

% Compute the PSTH for each neuron by averaging across trials
PSTH = zeros(N, T);
for i = 1:num_trials
    PSTH = PSTH + S(i).spikes;
end
PSTH = PSTH / num_trials;

% Plot PSTHs
figure;
imagesc(PSTH);
colorbar;
xlabel('Time (ms)');
ylabel('Neuron index');
title('PSTHs Before Smoothing');

% Part B: Gaussian Process Smoothing
% Define the Gaussian Process Covariance matrix K
A = 1; % Amplitude parameter
l = 50; % Length-scale parameter (can be tuned)

time_points = 1:T;
K = A * exp(-(bsxfun(@minus, time_points(:), time_points(:)').^2) / (2 * l^2));

% Smooth the PSTHs using Gaussian Process prior
smoothed_PSTH = zeros(N, T);
for n = 1:N
    % Prior distribution
    prior_mean = zeros(T, 1);
    prior_cov = K;

    % Observed data (PSTH of a single neuron)
    observed_spikes = PSTH(n, :)';

    % Gaussian Process Regression
    noise_covariance = 0.1 * eye(T);
    posterior_cov = prior_cov + noise_covariance;
    posterior_mean = prior_cov / posterior_cov * observed_spikes;

    % Store smoothed PSTH for neuron n
    smoothed_PSTH(n, :) = posterior_mean';
end

% Plot smoothed PSTHs
figure;
imagesc(smoothed_PSTH);
colorbar;
xlabel('Time (ms)');
ylabel('Neuron index');
title('PSTHs After Gaussian Process Smoothing');

% Visual comparison
figure;
subplot(2, 1, 1);
imagesc(PSTH);
colorbar;
xlabel('Time (ms)');
ylabel('Neuron index');
title('PSTHs Before Smoothing');

subplot(2, 1, 2);
imagesc(smoothed_PSTH);
colorbar;
xlabel('Time (ms)');
ylabel('Neuron index');
title('PSTHs After Gaussian Process Smoothing');

%% Answer for Part A
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Before Gaussian Process Smoothing:
%   - The PSTHs before smoothing are more noisy due to trial-to-trial variability.
%   - It is hard to identify patterns of neural activity before smoothing because of noticeable variability in some neurons' spike rates over time.
% 
% After Gaussian Process Smoothing:
%   - The Gaussian Process smoothed PSTHs show a much clearer trend over time.
%   - The smoothing process effectively reduces noise obeserved in the non-smoothed condition and yields more consistent patterns of activity across neurons.
%   - I obesvered that for instance, neuron 3,9,33 exhibit distinctive temporal patterns that are not identifiable in the unsmoothed PSTHs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part B Apply PCA

% Compute PCA on unsmoothed PSTHs
[coeff_raw, score_raw, ~, ~, explained_raw] = pca(PSTH');

% Compute PCA on smoothed PSTHs
[coeff_smooth, score_smooth, ~, ~, explained_smooth] = pca(smoothed_PSTH');

% Select the top 3 principal components
num_components = 3;

% Plot top 3 principal components side-by-side
figure;
subplot(1, 2, 1);
plot3(score_raw(:, 1), score_raw(:, 2), score_raw(:, 3), '-o');
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
title('Top 3 PCs (Original PSTH)');
grid on;

subplot(1, 2, 2);
plot3(score_smooth(:, 1), score_smooth(:, 2), score_smooth(:, 3), '-o');
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
title('Top 3 PCs (Smoothed PSTH)');
grid on;

% Display explained variance for both
disp('Explained Variance (Original PSTH):');
disp(explained_raw(1:num_components));

disp('Explained Variance (Smoothed PSTH):');
disp(explained_smooth(1:num_components));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Answer for PART B: Comparing the two PCAs between non-smoothed and smoothed PSTH

% Comparison of Principal Components Analysis (PCA)
% 
% PCAs of non-smoothed PSTHs:
%   - The 3D plot of the non-smoothed PSTHs exhibits significant variability and noise and have scattered distribution.
%   - The principal components do not capture consistent neural activity patterns.
% 
% PCAs of smoothed PSTHs:
%   - The smoothed PSTHs reveal clearer and more consistent trajectories in the 3D plot.
%   - The principal components appear more structured, suggesting that Gaussian Process smoothing aids in revealing underlying patterns of neural activity.
% 
% In summary, the Gaussian process smoothing reduces noise of PSTHs and
% improve the PCA's ability to uncover consistent temporal patterns across
% trials.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Part C

% Please execute the following addpath line if you do not have GPFA added
% Add GPFA functions to MATLAB path
% addpath(genpath('path/to/gpfa'));  % Replace with the actual path to the GPFA package

% Set parameters for GPFA
runIdx = 1;
binWidth = 1;  % Bin size of 1 ms
method = 'gpfa';
xDim = 8;  % Number of latent dimensions
kernSD = 30;  % Smoothing parameter (standard deviation of Gaussian kernel)

% % Set parameters for GPFA
% params.method = 'gpfa'; % Gaussian Process Factor Analysis
% params.binWidth = 1; % 1 ms bin width
% params.kernSD = 30; % Standard deviation of Gaussian kernel (tunable parameter)
% params.xDim = 3; % Number of latent dimensions
% runIdx = 1; % Identifier for the run

% Run GPFA using neuralTraj function
result = neuralTraj(runIdx, S, 'method', method, 'xDim', xDim,... 
                    'binWidth',binWidth,'kernSDList', kernSD);
[estParams, seqTrain] = postprocess(result, 'kernSD', kernSD);
% Visualize results using plot3D function
plot3D(seqTrain, 'xorth', 'dimsToPlot', 1:3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Observation in Part C
% Observations of GPFA Analysis
% 
% - I observed smooth, consistent trajectories of latent neural activity across trials.
% - The trajectories showed shared patterns of neural activity underlying different trials.

% Remarks:
% - GPFA provides a clear representation of consistent patterns across trials that are not easily observable through conventional PCA.
% - GPFA can reduce the noise and identifying latent dimensions to help interpret neural population dynamics.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part D

% Possible sources of trial-by-trial variability:
%   - Shifts in subject's attential during data collection
%   - Pysiological noise can lead to variability
%   - Variations in target location may lead to different neural trajectory.


% Chosen hypothesis: difference in attention of the subject during the data collection task could cause variations in neural trajectories.

% Design a theoretical way to test this: 

% Recruit a set of participants and
% let them perform a task that requires sustained attention. Split the
% groups into two groups: high attention (without distraction) and low attention (with distraction). 
% Record neural activities from relevant motor regions. Use GPFA to obtain
% trial-by-trial latent trajectories. Calculate euclidean distance between each trial's trajectory for each group. 
% Compare the quantified deviations between the two groups.
% If there is statistical difference between the two groups, that supports the hypothesis. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
