import os
import json
import numpy as np
import argparse
import scipy.ndimage

###################
### DGP Utils   ###
###################

def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

def diffusion_kernel(shape=(3, 3), advection=(0, 0)):
    """
    Create a simple diffusion/advection kernel for updating X.
    """
    kernel = np.zeros(shape)
    center = (shape[0] // 2, shape[1] // 2)
    kernel[center] = 0.0
    # Up/Down
    kernel[center[0] - 1, center[1]] = 0.125 + advection[0]
    kernel[center[0] + 1, center[1]] = 0.125 - advection[0]
    # Left/Right
    kernel[center[0], center[1] - 1] = 0.125 + advection[1]
    kernel[center[0], center[1] + 1] = 0.125 - advection[1]
    return kernel / np.sum(kernel) # Return normalized kernel

def save_params(params_dict, filename):
    """
    Save a dictionary of parameters to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(params_dict, f, indent=4)


###################
### Main DGP    ###
###################
def generate_next_X(
    t, X, A,
    alpha_0, alpha_1, alpha_2, alpha_3,
    epsilon_L, K_X
):
    """
    Update X[t] based on X[t-1], A[t-1], and a diffusion-like process.
    """
    x_spatial = scipy.ndimage.convolve(X[t-1], K_X, mode='reflect')
    X[t] = (
        alpha_0
        + alpha_1 * X[t-1]
        + alpha_2 * A[t-1]      # Effect of last-step intervention on X
        + alpha_3 * x_spatial
        + epsilon_L
    )

def generate_next_A(
    t, X, A, mu,
    beta_0, beta_1,
    K_A, L=1
):
    """
    Update A[t] based on X[t] via a logistic policy, and record logit in mu[t].
    """
    lagged_avg_X = np.sum(X[max(t-L+1, 0):t+1], axis=0) / min(t, L)
    x_convolved = scipy.ndimage.convolve(lagged_avg_X, K_A, mode='reflect')
    A_prob = sigmoid(
        beta_1 * (beta_0 + x_convolved)
    )
    if mu is not None:
        mu[t] = beta_1 * (beta_0 + x_convolved)
    A[t] = np.random.binomial(1, A_prob)

def generate_next_Y(
    t, X, A, Y,
    gamma_0, gamma_1, gamma_2, gamma_3,
    epsilon_Y, K_Y, K_YA, K_YX, L=1
):
    """
    Update Y[t] based on Y[t-1], A[t-1], and X[t-1].
    """
    lagged_avg_X = np.sum(X[max(t-L, 0):t], axis=0) / min(t, L)
    x_spatial = scipy.ndimage.convolve(lagged_avg_X, K_YX, mode='reflect')
    A_spatial = scipy.ndimage.convolve(A[t-1], K_YA, mode='reflect')
    y_spatial = scipy.ndimage.convolve(Y[t-1], K_Y, mode='reflect')
    Y[t] = (
        gamma_0
        + gamma_1 * A_spatial
        + gamma_2 * x_spatial
        + gamma_3 * y_spatial
        + epsilon_Y
    )

def generate_factual_data(
    alpha_0, alpha_1, alpha_2, alpha_3,
    beta_0, beta_1, gamma_0, gamma_1, gamma_2, gamma_3, 
    K_X, K_A, K_Y, K_YA, K_YX, L,
    grid_size=64, time_steps=200, seed=42
):
    """
    Generate factual data (X, A, Y) for a spatio-temporal process.
    
    Returns:
        X (ndarray): shape (time_steps, grid_size, grid_size)
        A (ndarray): shape (time_steps, grid_size, grid_size)
        Y (ndarray): shape (time_steps, grid_size, grid_size)
        mu (ndarray): shape (time_steps, grid_size, grid_size) [treatment logit, optional]
    """
    np.random.seed(seed)

    #offset = 10  # To allow system to reach equlibrium
    #time_steps += offset
    # Initialize arrays
    X = np.zeros((time_steps, grid_size, grid_size))
    A = np.zeros((time_steps, grid_size, grid_size))
    Y = np.zeros((time_steps, grid_size, grid_size))
    mu = np.zeros((time_steps, grid_size, grid_size))  # For debugging/logit

    # 1) Initialize X_0 as a smooth random field
    #X_0_base = np.random.normal(size=(grid_size, grid_size))
    #X[0] = scipy.ndimage.gaussian_filter(X_0_base, sigma=5)
    X[0] = np.random.normal(size=(grid_size, grid_size))

    # 2) Define A_0 based on logistic policy
    A_prob_0 = sigmoid(
        beta_1 * (beta_0 + X[0] + scipy.ndimage.convolve(X[0], K_A, mode='reflect'))
    )
    A[0] = np.random.binomial(1, A_prob_0)

    # 3) Simulate forward
    for t in range(1, time_steps):
        # Noise
        epsilon_L = np.random.normal(scale=1, size=(grid_size, grid_size))
        epsilon_Y = np.random.normal(scale=1, size=(grid_size, grid_size))

        # -- X update (Air Quality)
        generate_next_X(
            t, X, A,
            alpha_0, alpha_1, alpha_2, alpha_3,
            epsilon_L, K_X
        )

        # -- A update (Intervention policy)
        generate_next_A(
            t, X, A, mu,
            beta_0, beta_1, K_A, L)

        # -- Y update (Outcome)
        generate_next_Y(
            t, X, A, Y,
            gamma_0, gamma_1, gamma_2, gamma_3,
            epsilon_Y, K_Y, K_YA, K_YX, L)

    return X, A, Y, mu

def generate_counterfactuals(
    alpha_0, alpha_1, alpha_2, alpha_3,
    beta_0, beta_1, gamma_0, gamma_1, gamma_2, gamma_3, 
    K_X, K_A, K_Y, K_YA, K_YX, X0, L,
    history_len, dim_horizon, n_test,
    grid_size=64, seed=1, mean_path=False
):
    """
    Generate counterfactual data for n_test sequences, each of length history_len+dim_horizon.
    The override_p is a baseline probability for the new intervention after history_len.
    """
    np.random.seed(seed)

    offset = 10 # To allow system to reach equlibrium
    seq_len = offset + history_len + dim_horizon

    # We store shape: (n_test, seq_len+1, grid_size, grid_size)
    X_test = np.zeros((n_test, seq_len+1, grid_size, grid_size))
    A_test = np.zeros((n_test, seq_len+1, grid_size, grid_size))
    Y_test = np.zeros((n_test, seq_len+1, grid_size, grid_size))

    n_trajectories = 100 # Number of counterfactual trajectories to average if mean_path=True

    # A_counter shape (dim_horizon+1, grid_size, grid_size) => same intervention for all i
    A_counter = np.zeros((dim_horizon+1, grid_size, grid_size))
    # Initialize A as a random gaussian process
    A_counter = np.random.binomial(n=1, p=0.5, size=A_counter.shape)

    # Initialize each test with X[0], A[0] as above or from real data
    for i in range(n_test):
        # Start from the same X[0] or a new random one
        # X_test[i][0] = X0
        #X_0_base = np.random.normal(size=(grid_size, grid_size))  # Random noise
        #X_test[i][0] = scipy.ndimage.gaussian_filter(X_0_base, sigma=5)  # Smooth with Gaussian filter
        X_test[i][0] = np.random.normal(size=(grid_size, grid_size))

        # Define A_0 based on X_0
        A_prob_0 = sigmoid(beta_1 * (beta_0 + X_test[i][0] + scipy.ndimage.convolve(X_test[i][0], K_A, mode='reflect')))
        A_test[i][0] = np.random.binomial(1, A_prob_0)

        for t in range(1, history_len + offset):
            # Example is simplistic: replicate the same logic as generate_factual_data
            epsilon_L = np.random.normal(scale=1, size=(grid_size, grid_size))
            epsilon_Y = np.random.normal(scale=1, size=(grid_size, grid_size))

            # X update
            generate_next_X(
                t, X_test[i], A_test[i],
                alpha_0, alpha_1, alpha_2, alpha_3,
                epsilon_L, K_X
            )
            generate_next_A(
                t, X_test[i], A_test[i], mu=None,
                beta_0=beta_0, beta_1=beta_1, K_A=K_A, L=L
            )
            # Y update
            generate_next_Y(
                t, X_test[i], A_test[i], Y_test[i],
                gamma_0, gamma_1, gamma_2, gamma_3,
                epsilon_Y, K_Y, K_YA, K_YX, L
            )
        if mean_path:
            Y_trajectory = np.zeros((n_trajectories, dim_horizon+1, grid_size, grid_size))
            for j in range(n_trajectories):
                for t in range(history_len + offset, seq_len+1):
                    # Replicate the same logic as generate_factual_data
                    epsilon_L = np.random.normal(scale=1, size=(grid_size, grid_size))
                    epsilon_Y = np.random.normal(scale=1, size=(grid_size, grid_size))
                    A_test[i][t] = A_counter[t - history_len - offset]
                    # X update
                    generate_next_X(
                        t, X_test[i], A_test[i],
                        alpha_0, alpha_1, alpha_2, alpha_3,
                        epsilon_L, K_X
                    )
                    # Y update
                    generate_next_Y(
                        t, X_test[i], A_test[i], Y_test[i],
                        gamma_0, gamma_1, gamma_2, gamma_3,
                        epsilon_Y, K_Y, K_YA, K_YX, L
                    )
                    Y_trajectory[j, t - history_len - offset] = Y_test[i][t]
            Y_test[i, history_len + offset:] = np.mean(Y_trajectory, axis=0)
        else:
            for t in range(history_len + offset, seq_len+1):
                # Replicate the same logic as generate_factual_data
                epsilon_L = np.random.normal(scale=1, size=(grid_size, grid_size))
                epsilon_Y = np.random.normal(scale=1, size=(grid_size, grid_size))
                A_test[i][t] = A_counter[t - history_len - offset]
                # X update
                generate_next_X(
                    t, X_test[i], A_test[i],
                    alpha_0, alpha_1, alpha_2, alpha_3,
                    epsilon_L, K_X
                )
                # Y update
                generate_next_Y(
                    t, X_test[i], A_test[i], Y_test[i],
                    gamma_0, gamma_1, gamma_2, gamma_3,
                    epsilon_Y, K_Y, K_YA, K_YX, L
                )
    return X_test[:, offset:, :, :], A_test[:, offset:, :, :], Y_test[:, offset:, :, :], A_counter

###################
### Main Script ###
###################

def main(
        alpha_0=0.5, alpha_1=0.3, alpha_2=-2.0, alpha_3=0.3, 
        beta_0=0.5, beta_1=1.0, 
        gamma_0=1.0, gamma_1=1.5, gamma_2=0.5, gamma_3=0.5, 
        K_X=np.array([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]]),
        K_A=np.array([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]]),
        K_Y=np.array([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]]),
        K_YA=np.array([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]]),
        K_YX=np.array([[0, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0]]),
        L=1, history_len=10, dim_horizon=10, 
        n_test=50, grid_size=64, time_steps=200, seed=42, mean_path=False):
    """
    Main function: 
      - Generate the data (factual)
      - Generate counterfactuals
      - Save arrays and parameters
    """

    # 2) Generate the factual data
    X, A, Y, mu = generate_factual_data(
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        alpha_3=alpha_3,
        beta_0=beta_0,
        beta_1=beta_1,
        gamma_0=gamma_0,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        gamma_3=gamma_3,
        K_X=K_X,
        K_A=K_A,
        K_Y=K_Y,
        K_YX=K_YX,
        K_YA=K_YA,
        L=L,
        grid_size=grid_size,
        time_steps=time_steps,
        seed=args.seed
    )
    print("Factual data generation done.")

    # 3) Generate counterfactuals
    X_test, A_test, Y_test, A_counter = generate_counterfactuals(
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        alpha_3=alpha_3,
        beta_0=beta_0,
        beta_1=beta_1,
        gamma_0=gamma_0,
        gamma_1=gamma_1,
        gamma_2=gamma_2,
        gamma_3=gamma_3,
        K_X=K_X,
        K_A=K_A,
        K_Y=K_Y,
        K_YX=K_YX,
        K_YA=K_YA,
        X0=X[0],
        L=L, 
        history_len=history_len, 
        dim_horizon=dim_horizon, 
        n_test=n_test,
        grid_size=grid_size,
        seed=1,
        mean_path=mean_path
    )
    print("Counterfactual data generation done.")

    # 4) Prepare output directories
    dgp_folder = os.path.join(args.output_dir, f"dgp_linear_beta_1_{beta_1:.1f}")
    counter_folder = os.path.join(dgp_folder, "counterfactuals")
    if not os.path.exists(dgp_folder):
        os.makedirs(dgp_folder)
    if not os.path.exists(counter_folder):
        os.makedirs(counter_folder)

    # 5) Save arrays
    np.save(os.path.join(dgp_folder, "X.npy"), X.reshape((time_steps, 1, grid_size, grid_size)))
    np.save(os.path.join(dgp_folder, "A.npy"), A)
    np.save(os.path.join(dgp_folder, "Y.npy"), Y)

    np.save(os.path.join(counter_folder, "X.npy"), X_test.reshape((n_test, history_len+dim_horizon+1, 1, grid_size, grid_size)))
    np.save(os.path.join(counter_folder, "A.npy"), A_test)
    np.save(os.path.join(counter_folder, "Y.npy"), Y_test)
    np.save(os.path.join(counter_folder, "A_counter.npy"), A_counter)

    print("Arrays saved to", dgp_folder)
    # 6) Save the parameter dictionary
    params_dict = {
        "alpha_0": alpha_0,
        "alpha_1": alpha_1,
        "alpha_2": alpha_2,
        "alpha_3": alpha_3,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "gamma_0": gamma_0,
        "gamma_1": gamma_1,
        "gamma_2": gamma_2,
        "gamma_3": gamma_3,
        "K_X": K_X.tolist(),
        "K_A": K_A.tolist(),
        "K_Y": K_Y.tolist(),
        "K_YX": K_YX.tolist(),
        "K_YA": K_YA.tolist(),
        "history_len": history_len,
        "dim_horizon": dim_horizon,
        "n_test": n_test,
        "grid_size": grid_size,
        "time_steps": time_steps,
        "seed": args.seed
    }
    params_filename = os.path.join(dgp_folder, "params.txt")
    save_params(params_dict, params_filename)
    print("Parameters saved to", params_filename)

###################
### Main Entry  ###
###################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Linear-like DGP with spatio-temporal diffusion and logistic interventions.")
    parser.add_argument("--beta_1", nargs='+', type=float, default=[1.0], help="Confounder strengths for intervention, e.g., --beta_1 0.0, 1.0")
    parser.add_argument("--mean_path", action="store_true", help="Verbose mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_dir", type=str, default="./simulated_data/linear", help="Where to save results.")
    args = parser.parse_args()

    # 1) Unpack or define parameters
    mean_path = args.mean_path
    alpha_0, alpha_1, alpha_2, alpha_3 = 0.5, 0.5, -2.0, 0.2
    beta_0 = -1.0
    all_beta_1 = args.beta_1  # from CLI
    gamma_0, gamma_1, gamma_2, gamma_3 = 2.0, 1.5, 0.5, 0.5
    # Convolutions
    K_X = diffusion_kernel(advection=(0.1, -0.05))  # Diffusion kernel with slight advection
    K_A = np.ones((3, 3)) / 16  # Uniform averaging kernel for intervention spatial dependence
    K_A[1, 1] = 0.5  # Self-effect
    K_Y = np.zeros((3, 3))  # Uniform averaging kernel for outcome spatial dependence
    K_Y[1, 1] = 1.0  # Self-effect
    # New kernels for interference and non-local confounding
    K_YA = np.ones((3, 3)) / 16 # Uniform averaging kernel for interference
    K_YA[1, 1] = 0.5  # Self-effect
    K_YX = np.ones((3, 3)) / 16 # Uniform averaging kernel for interference
    K_YX[1, 1] = 0.5  # Self-effect
    # Lags and simulation parameters
    L = 5 # Number of lags for temporal dependence
    history_len, dim_horizon, n_test = 10, 10, 50
    grid_size, time_steps = 64, 200

    # Generate data for all beta_1
    for beta_1 in all_beta_1:
        main(
            alpha_0=alpha_0, alpha_1=alpha_1, alpha_2=alpha_2, alpha_3=alpha_3,
            beta_0=beta_0, beta_1=beta_1,
            gamma_0=gamma_0, gamma_1=gamma_1, gamma_2=gamma_2, gamma_3=gamma_3,
            K_X=K_X, K_A=K_A, 
            K_Y=K_Y, K_YA=K_YA, K_YX=K_YX,
            L=L, history_len=history_len, dim_horizon=dim_horizon, n_test=n_test,
            grid_size=grid_size, time_steps=time_steps, seed=args.seed, mean_path=mean_path)