""" Binary DGP based on a physical diffusion model. """

import os
import numpy as np
import argparse
import scipy.ndimage
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky

#################
### DGP Utils ###
#################

# Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Function to compute the Laplacian (for diffusion)
def laplacian(C, dx, dy):
    return (np.roll(C, -1, axis=0) + np.roll(C, 1, axis=0) - 2 * C) / dx**2 + \
           (np.roll(C, -1, axis=1) + np.roll(C, 1, axis=1) - 2 * C) / dy**2

# Function to compute the gradient (for advection)
def gradient(C, dx, dy):
    grad_x = (np.roll(C, -1, axis=1) - np.roll(C, 1, axis=1)) / (2 * dx)
    grad_y = (np.roll(C, -1, axis=0) - np.roll(C, 1, axis=0)) / (2 * dy)
    return grad_x, grad_y

# Functions to simulate Gaussian processes 
def rbf_kernel(X, length_scale, variance):
    """Compute the RBF kernel matrix."""
    dists = squareform(pdist(X, 'euclidean'))**2
    K = variance * np.exp(-0.5 * dists / length_scale**2)
    return K

def sample_conditional_gp(K, prev_sample, length_scale_time, 
                          offset, variance, nx, ny):
    """
    Sample from the GP conditioned on the previous time step.

    Parameters:
    K (ndarray): Covariance matrix.
    prev_sample (ndarray): Previous sample from the GP.
    length_scale_time (float): Temporal length scale.
    offset (float): Mean offset for the GP.
    variance (float): Variance of the GP.
    nx (int): Number of grid points in x direction.
    ny (int): Number of grid points in y direction.
    """
    # Sample from the Gaussian process
    L = cholesky(K + 1e-6 * np.eye(K.shape[0]), lower=True)
    samples = L @ np.random.randn(K.shape[0])
    
    # Combine with the previous sample using temporal correlation
    temporal_correlation = np.exp(-0.5 / length_scale_time**2)
    samples = temporal_correlation * (prev_sample.ravel() - offset) + np.sqrt(1 - temporal_correlation**2) * samples
    return samples.reshape(ny, nx) + offset

def sample_gp(K, length_scale_space, offset, nx, ny):
    """Sample from the GP conditioned on the previous time step."""    
    # Sample from the Gaussian process
    L = cholesky(K + 1e-6 * np.eye(K.shape[0]), lower=True)
    samples = L @ np.random.randn(K.shape[0]) + offset
    return samples.reshape(ny, nx)

########################
### Simulation Utils ###
########################
# Function to generate next treatment
def generate_next_A(t, A, X, Y, alpha_A, beta_A, K_Y, K_T, L):
    _, _, temperature = X
    # Update A
    lagged_avg_Y = np.sum(Y[max(t-L+1, 0):t+1], axis=0) / min(t, L)
    lagged_avg_temp = np.sum(temperature[max(t-L+1, 0):t+1], axis=0) / min(t, L)
    mu = - alpha_A * (
        scipy.ndimage.convolve(lagged_avg_Y, K_Y, mode='constant', cval=0)/K_Y.sum() +\
        scipy.ndimage.convolve(lagged_avg_temp, K_T, mode='constant', cval=0)/K_T.sum() +\
        beta_A
    )
    A[t] = np.random.binomial(n=1, p=sigmoid(mu))

# Function to generate next outcome
def generate_next_Y(t, A, X, Y, alpha_Y, beta_Y, L, dt, dx, dy):
    wind_x, wind_y, temperature = X
    D = 2 + 0.05 * (temperature[t-1] - 20)
    # Update Y
    # Diffusion term
    diffusion = D * laplacian(Y[t-1], dx, dy)    
    # Advection term
    advection_x = -wind_x[t-1] * (np.roll(Y[t-1], -1, axis=1) - np.roll(Y[t-1], 1, axis=1)) / (2 * dx)
    advection_y = -wind_y[t-1] * (np.roll(Y[t-1], -1, axis=0) - np.roll(Y[t-1], 1, axis=0)) / (2 * dy)
    # Update Y
    Y[t] = Y[t-1] + alpha_Y * (1-A[t-1]) + dt * (diffusion + advection_x + advection_y)
    Y[t] -= beta_Y * dt * Y[t]  # Added a sink
    #Y[t] += np.random.uniform(size=Y[t].shape, high=0.05)       

# Functions to simulate the diffusion process
def generate_covariates(grid_params, gp_params, random_seed=42, from_file=False, data_folder=None):
    """
    Generate temperature and wind covariates for the diffusion model.

    Parameters:
    grid_params (dict): Dictionary containing grid parameters (nx, ny, dx, dy, nt).
    gp_params (dict): Dictionary containing Gaussian process parameters (length_scale_space, length_scale_time, 
                      temp_offset, temp_variance, wind_x_offset, wind_y_offset, wind_variance).
    random_seed (int): Seed for random number generation.
    """
    if not from_file:
        np.random.seed(random_seed)
        nx, ny, dx, dy, nt = grid_params['nx'], grid_params['ny'], grid_params['dx'], grid_params['dy'], grid_params['nt']
        length_scale_space, length_scale_time = gp_params['length_scale_space'], gp_params['length_scale_time']
        temp_offset, temp_variance = gp_params['temp_offset'], gp_params['temp_variance']
        wind_x_offset, wind_y_offset, wind_variance = gp_params['wind_x_offset'], gp_params['wind_y_offset'], gp_params['wind_variance']
        
        x = np.linspace(0, nx * dx, nx)
        y = np.linspace(0, ny * dy, ny)
        X, Y = np.meshgrid(x, y)
        coords = np.vstack([X.ravel(), Y.ravel()]).T
        K_temp = rbf_kernel(coords, length_scale_space, temp_variance)
        K_wind = rbf_kernel(coords, length_scale_space, wind_variance)

        # Initialize temperature field and wind fields using Gaussian processes
        temperature = np.zeros((nt, ny, nx))
        wind_x = np.zeros((nt, ny, nx))
        wind_y = np.zeros((nt, ny, nx))

        # Generate the initial fields
        temperature[0] = sample_gp(K_temp, length_scale_space, temp_offset, nx, ny)
        wind_x[0] = sample_gp(K_wind, length_scale_space, wind_x_offset, nx, ny)
        wind_y[0] = sample_gp(K_wind, length_scale_space, wind_y_offset, nx, ny)

        # Generate fields sequentially over time
        print("Covariate time step: 0", end=",")
        for t in range(1, nt):
            temperature[t] = sample_conditional_gp(K_temp, temperature[t-1], length_scale_time, temp_offset, temp_variance, nx, ny)
            wind_x[t] = sample_conditional_gp(K_wind, wind_x[t-1], length_scale_time, wind_x_offset, wind_variance, nx, ny)
            wind_y[t] = sample_conditional_gp(K_wind, wind_y[t-1], length_scale_time, wind_y_offset, wind_variance, nx, ny)
            print(f" {t}", end=",")
    else:
        X = np.load(os.path.join(data_folder, "X.npy")).transpose(1, 0, 2, 3)
        wind_x, wind_y, temperature = X

    return wind_x, wind_y, temperature

# Function to generate treatment and outcome
def generate_treatment_and_outcome(grid_params, gp_params, eq_params, covariates, random_seed=42):
    """
    Generates treatment and outcome data for a spatiotemporal diffusion process.

    Parameters:
    grid_params (dict): Dictionary containing grid parameters (nx, ny, dx, dy, nt).
    gp_params (dict): Dictionary containing Gaussian process parameters (length_scale_space_A, variance_A, offset_A).
    eq_params (dict): Dictionary containing equation parameters (alpha_Y, beta_Y, L, alpha_A, beta_A, K_Y, K_T).
    random_seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
    tuple: A tuple containing:
        - A (ndarray): Treatment assignment array of shape (nt, ny, nx).
        - Y (ndarray): Outcome array of shape (nt, ny, nx).
    """
    np.random.seed(random_seed)
    nx, ny, dx, dy, nt, dt = grid_params['nx'], grid_params['ny'], grid_params['dx'], grid_params['dy'], grid_params['nt'], grid_params['dt']
    length_scale_space_A, variance_A, offset_A = gp_params['length_scale_space_A'], gp_params['variance_A'], gp_params['offset_A']
    alpha_Y, beta_Y, L, alpha_A, beta_A = eq_params['alpha_Y'], eq_params['beta_Y'], eq_params['L'], eq_params['alpha_A'], eq_params['beta_A']
    K_Y, K_T = eq_params['K_Y'], eq_params['K_T']

    A = np.zeros((nt, ny, nx))
    Y = np.zeros((nt, ny, nx))

    # Initialize A as a random gaussian process
    x = np.linspace(0, nx*dx, nx)
    y = np.linspace(0, ny*dy, ny)
    x_mesh, y_mesh = np.meshgrid(x, y)
    coords = np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T
    K = rbf_kernel(coords, length_scale_space_A, variance_A)
    mu = sigmoid(sample_gp(K, length_scale_space_A, offset_A, nx, ny))
    A[0] = np.random.binomial(n=1, p=mu)

    # Generate factual data
    for t in range(1, nt):
        # Generate next outcome
        generate_next_Y(t, A, covariates, Y, alpha_Y, beta_Y, L, dt, dx, dy)
        # Generate next treatment
        generate_next_A(t, A, covariates, Y, alpha_A, beta_A, K_Y, K_T, L)
        
    return A, Y

# Function to generate counterfactuals
def generate_counterfactuals(grid_params, gp_params, gp_params_A, eq_params, counter_params, from_file=False, data_folder=None):
    """
    Generate counterfactuals for the diffusion model.
    Parameters:
    grid_params (dict): Dictionary containing grid parameters (nx, ny, dx, dy, nt).
    gp_params (dict): Parameters for the Gaussian process (length_scale_space_A, length_scale_time_A, variance_A, offset_A, counter_offset_A). 
    gp_params_A (dict): Parameters for the Gaussian process for treatment (length_scale_space_A, length_scale_time_A, offset_A, variance_A). 
    eq_params (dict): Equation parameters (alpha_Y, beta_Y, L, alpha_A, beta_A, K_Y, K_T) for the diffusion model.
    counter_params (dict): Parameters for counterfactual generation (random_seed, history_len, n_test, dim_horizon). 

    Returns:
    tuple: A tuple containing:
        - A_counter (ndarray): Counterfactual treatment series.
        - X_test (ndarray): Generated covariates.
        - A_test (ndarray): Generated treatment series.
        - Y_test (ndarray): Generated outcome series.
    """
    nx, ny, dx, dy, dt = grid_params['nx'], grid_params['ny'], grid_params['dx'], grid_params['dy'], grid_params['dt']
    length_scale_space_A, variance_A, offset_A = gp_params_A['length_scale_space_A'], gp_params_A['variance_A'], gp_params_A['counter_offset_A']
    length_scale_time_A = gp_params_A['length_scale_time_A']
    alpha_Y, beta_Y, L, alpha_A, beta_A = eq_params['alpha_Y'], eq_params['beta_Y'], eq_params['L'], eq_params['alpha_A'], eq_params['beta_A']
    K_Y, K_T = eq_params['K_Y'], eq_params['K_T']
    random_seed, history_len, n_test, dim_horizon = counter_params['random_seed'], counter_params['history_len'], counter_params['n_test'], counter_params['dim_horizon']

    seq_len = history_len + dim_horizon
    X_test = np.zeros((n_test, seq_len, 3, nx, ny))
    A_test = np.zeros((n_test, seq_len, nx, ny))
    Y_test = np.zeros((n_test, seq_len, nx, ny))
    A_counter = np.zeros((dim_horizon, nx, ny))

    ### Generate A_counter in as a smoothly varying series of treatments
    # Initialize A as a random gaussian process
    np.random.seed(random_seed)
    x = np.linspace(0, nx*dx, nx)
    y = np.linspace(0, ny*dy, ny)
    x_mesh, y_mesh = np.meshgrid(x, y)
    coords = np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T
    K = rbf_kernel(coords, length_scale_space_A, variance_A)
    mu = sample_gp(K, length_scale_space_A, offset_A, nx, ny)
    A_counter[0] = np.random.binomial(n=1, p=sigmoid(mu))
    mu_prev = mu
    for t in range(1, dim_horizon):
        mu = sample_conditional_gp(K, mu_prev, length_scale_time_A, 
                                   offset_A, variance_A, nx, ny)
        A_counter[t] = np.random.binomial(n=1, p=sigmoid(mu))
        mu_prev = mu
    counter_grid_params = {
        'nx': nx,              
        'ny': ny,               
        'dx': dx,              
        'dy': dy,
        'nt': seq_len    
    }
    if from_file:
        X_test = np.load(os.path.join(data_folder, "X.npy"))
    for i in range(n_test):
        if not from_file:
            wind_x_test, wind_y_test, temp_test = generate_covariates(counter_grid_params, gp_params, random_seed=random_seed + i)
            X_test[i] = np.stack([wind_x_test, wind_y_test, temp_test], axis=0).transpose(1,0,2,3)
        else:
            wind_x_test, wind_y_test, temp_test = X_test[i].transpose(1,0,2,3)                      
        # Generate A[i][0]
        mu = sample_gp(K, length_scale_space_A, -1.0, nx, ny)
        A_test[i][0] = np.random.binomial(n=1, p=sigmoid(mu))

        # Generate counterfactual data
        for t in range(1, seq_len):
            # Generate next outcome
            generate_next_Y(t, A_test[i], (wind_x_test, wind_y_test, temp_test), Y_test[i], alpha_Y, beta_Y, L, dt, dx, dy)
            # Update A
            if seq_len - t - dim_horizon > 0:
                #Generate next treatment
                generate_next_A(t, A_test[i], (wind_x_test, wind_y_test, temp_test), Y_test[i], alpha_A, beta_A, K_Y, K_T, L)
            else:
                A_test[i][t] = A_counter[dim_horizon - (seq_len-t)]
    
    return A_counter, X_test, A_test, Y_test

# Function to save parameters to a file
def save_params(params, filename):
    with open(filename, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

###################
### Main Script ###
###################
def main(alpha_A=1.0, from_file=False):
    # Grid and time parameters
    print("Starting DGP simulation...")
    grid_params = {
        'nx': 64,               # Number of grid points in x direction
        'ny': 64,               # Number of grid points in y direction
        'dx': 1.0,              # Grid spacing in x direction
        'dy': 1.0,              # Grid spacing in y direction
        'dt': 0.1,              # Time step
        'T': 20.0,              # Total simulation time
        'nt': int(20.0 / 0.1)   # Number of time steps
    }

    # Parameters for Gaussian process
    gp_params = {
        'length_scale_space': 5.0,    # Spatial smoothness
        'length_scale_time': 3.0,     # Temporal smoothness
        'temp_offset': 20.0,          # Offset for temperature
        'temp_variance': 4.0,         # Amplitude for temperature
        'wind_x_offset': 0.0,         # Offset for wind in x direction
        'wind_y_offset': 0.0,         # Offset for wind in y direction
        'wind_variance': 0.5          # Amplitude for wind
    }

    # Parameters for Gaussian process for treatment
    gp_params_A = {
        'length_scale_space_A': 7.0,    # Spatial smoothness for treatment
        'length_scale_time_A': 3.0,     # Temporal smoothness for treatment
        'offset_A': -1.0,             # Offset for treatment
        'variance_A': 1.0,            # Amplitude for treatment
        'counter_offset_A': -0.5      # Counterfactual offset for treatment
    }

    # Equation parameters
    eq_params = {
        'alpha_A': alpha_A,                 # Confounding strength for treatment
        'beta_A': -20.5,                # Bias term for treatment
        'alpha_Y': 0.5,                 # Confounding strength for outcome
        'beta_Y': 4.0,                  # Bias term for outcome
        'L': 1,                        # Lag length
        'K_Y': np.array([               # Kernel for outcome
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.3, 0.3, 0.3, 0.1],
            [0.1, 0.3, 0.6, 0.3, 0.1],
            [0.1, 0.3, 0.3, 0.3, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1]
        ]),
        'K_T': np.array([              # Kernel for temperature
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.3, 0.3, 0.3, 0.1],
            [0.1, 0.3, 0.6, 0.3, 0.1],
            [0.1, 0.3, 0.3, 0.3, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1]
        ])
    }

    # Counterfactual parameters
    counter_params = {
        'random_seed': 42,     # Random seed for reproducibility
        'history_len': 10,     # Length of history for counterfactual generation
        'n_test': 10,          # Number of test samples
        'dim_horizon': 10      # Dimension of horizon for counterfactuals
    }

    # Random seed
    random_seed = 42

    # Target folders
    target_folder = "./simulated_data/diffusion"
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    dgp_folder = os.path.join(target_folder, f"dgp_diffusion_alpha_A_{eq_params['alpha_A']:.1f}")
    counter_folder = os.path.join(dgp_folder, "counterfactuals")
    if not os.path.exists(dgp_folder):
        os.makedirs(dgp_folder)
        os.makedirs(counter_folder)

    # Combine all parameters into a dictionary of dictionaries
    all_params = {
        'grid_params': grid_params,
        'gp_params': gp_params,
        'gp_params_A': gp_params_A,
        'eq_params': eq_params,
        'counter_params': counter_params,
        'random_seed': random_seed
    }

    # Save the parameters to the dgp folder
    params_filename = os.path.join(dgp_folder, "params.txt")
    save_params(all_params, params_filename)

    ######################
    ### DGP Simulation ###
    ######################
    # Generate covariates
    if from_file:
        wind_x, wind_y, temperature = generate_covariates(grid_params, gp_params, random_seed=random_seed, from_file=from_file, data_folder=dgp_folder)
    else:
        wind_x, wind_y, temperature = generate_covariates(grid_params, gp_params, random_seed=random_seed)

    print("Done generating covariates.")

    # Generate treatment and outcome
    A, Y = generate_treatment_and_outcome(grid_params, gp_params_A, eq_params, (wind_x, wind_y, temperature), random_seed=random_seed)
    print("Done generating treatment and outcome.")

    # Save the data
    X = np.stack([wind_x, wind_y, temperature], axis=0).transpose(1,0,2,3)
    if not from_file:
        np.save(os.path.join(dgp_folder, "X.npy"), X)
    np.save(os.path.join(dgp_folder, "A.npy"), A)
    np.save(os.path.join(dgp_folder, "Y.npy"), Y)

    # Generate counterfactuals
    if from_file:
        A_counter, X_test, A_test, Y_test = generate_counterfactuals(grid_params, gp_params, gp_params_A, eq_params, counter_params, from_file=from_file, data_folder=counter_folder)
    else:
        A_counter, X_test, A_test, Y_test = generate_counterfactuals(grid_params, gp_params, gp_params_A, eq_params, counter_params)
    print("Done generating counterfactuals.")

    # Save the counterfactuals
    np.save(os.path.join(counter_folder, "A_counter.npy"), A_counter)
    if not from_file:
        np.save(os.path.join(counter_folder, "X.npy"), X_test)
    np.save(os.path.join(counter_folder, "A.npy"), A_test)
    np.save(os.path.join(counter_folder, "Y.npy"), Y_test)
    print("Done saving counterfactuals.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary DGP based on a physical diffusion model.")
    parser.add_argument("--alpha_A", type=float, default=1.0, help="Confounding strength for treatment (alpha_A)")
    parser.add_argument("--from_file", action="store_true", help="Load covariates from file", default=False)
    args = parser.parse_args()

    # Pass the parsed alpha_A to the main function
    main(alpha_A=args.alpha_A, from_file=args.from_file)
