import os
import json
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Custom imports
from models import unet
from data import utils as data_utils
from data.data_loaders import SimulatedData


def count_parameters(model):
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_params(params_dict, filename):
    """
    Save a dictionary of parameters to a JSON file.
    """
    with open(filename, "w") as f:
        json.dump(params_dict, f, indent=4)

def main():
    """
    Main function that trains and tests a UNetConvLSTM model using the specified
    hyperparameters and data. It then performs a counterfactual analysis.
    """
    parser = argparse.ArgumentParser(description="Train UNetConvLSTM on linear simulated data and run counterfactual analysis.")
    parser.add_argument("--use_attention", action="store_true", help="Whether to use attention", default=False)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for optimizer.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Number of epochs before early stop.")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="Number of epochs before learning rate reduction.")
    parser.add_argument("--dim_horizon", type=int, default=10, help="Dimension of horizon (future steps for treatment override).")
    parser.add_argument("--beta_1", type=float, default=0.0, help="Confounder strength (beta_1) for the logistic policy in data generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    #########################################
    # Use command-line args for hyperparams #
    #########################################
    attention = args.use_attention
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    dim_horizon = args.dim_horizon
    early_stopping_patience = args.early_stopping_patience
    scheduler_patience = args.scheduler_patience
    beta_1 = args.beta_1
    seed = args.seed

    ########################
    # Various parameters   #
    ########################
    # Data directory
    data_dir = f"../data/simulated_data/linear"
    # We assume you have data in: dgp_linear_beta_1_{beta_1:.1f}
    dgp_data_dir = os.path.join(data_dir, f"dgp_linear_beta_1_{beta_1:.1f}")
    counter_data_dir = os.path.join(dgp_data_dir, "counterfactuals")

    # Saved models directory
    models_dir = os.path.join(dgp_data_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    history_len = 10
    tlen = history_len + dim_horizon

    # Set random seed
    torch.manual_seed(seed)

    ##################################
    # Load the training/test dataset #
    ##################################
    n_train = 150 # TODO: replace with the number of training samples read from the data dictionary
    train_dataset = SimulatedData(dgp_data_dir, n_train=n_train, train=True, tlen=tlen)
    test_dataset = SimulatedData(dgp_data_dir, n_train=n_train, train=False, tlen=tlen)

    # Load counterfactual data
    A_counter = torch.tensor(
        np.load(os.path.join(counter_data_dir, 'A_counter.npy'))[:dim_horizon]
    ).float().unsqueeze(1)

    # Hyper parameters (continued)
    height = 64
    width = 64
    in_channel = 3
    dim_treatments = 1
    dim_outcome = 1

    best_model_name = f'unetlstm_dim_horizon_{dim_horizon}_beta_1_{beta_1:.1f}.pth'

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    ###################
    # Normalization   #
    ###################
    normalizer = data_utils.DataNormalizer(train_loader)
    train_loader_normalized, test_loader_normalized = normalizer.normalize(train_loader, test_loader)

    ###################################
    # Create and Inspect the Model    #
    ###################################
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    unetlstm_model = unet.UNetConvLSTM(
        in_channel=in_channel,
        n_classes=1,
        dim_static=dim_horizon,
        bilinear=False,
        attention=attention
    ).to(device)

    print('Number of model parameters: {}'.format(count_parameters(unetlstm_model)))

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(unetlstm_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=scheduler_patience, factor=0.5)

    # Early stopping parameters
    best_loss = float('inf')
    patience_counter = 0

    ###################################
    # Training loop
    ###################################
    total_step = len(train_loader_normalized)
    for epoch in range(num_epochs):
        unetlstm_model.train()
        for i, (x, A, Y) in enumerate(train_loader_normalized):
            b = x.size(0)
            x = x[:, :tlen - dim_horizon]

            A_past = A[:, :tlen - dim_horizon].reshape(b, tlen - dim_horizon, 1, height, width)
            Y_past = Y[:, :tlen - dim_horizon].reshape(b, tlen - dim_horizon, 1, height, width)
            A_curr = A[:, tlen - dim_horizon:-1]  # future treatments
            Y_out = Y[:, -1]                     # final outcome

            Y_out = Y_out.to(device)
            A_curr = A_curr.to(device)

            # Forward pass
            inputs = torch.cat([x, A_past, Y_past], dim=2).to(device)
            outputs = unetlstm_model(inputs, A_curr).reshape(-1, height, width)
            loss = criterion(outputs, Y_out)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('[epoch: {}/{}] [step {}/{}] MSE: {:.4f}'.format(
                    epoch + 1, num_epochs, i, len(train_loader_normalized), loss.item())
                )

        # Validation / Testing
        unetlstm_model.eval()
        with torch.no_grad():
            test_mse = 0
            for i, (x, A, Y) in enumerate(test_loader_normalized):
                b = x.size(0)
                x = x[:, :tlen - dim_horizon]

                A_past = A[:, :tlen - dim_horizon].reshape(b, tlen - dim_horizon, 1, height, width)
                Y_past = Y[:, :tlen - dim_horizon].reshape(b, tlen - dim_horizon, 1, height, width)
                A_curr = A[:, tlen - dim_horizon:-1]
                Y_out = Y[:, -1]

                Y_out = Y_out.to(device)
                A_curr = A_curr.to(device)

                # Forward pass
                inputs = torch.cat([x, A_past, Y_past], dim=2).to(device)
                outputs = unetlstm_model(inputs, A_curr).reshape(-1, height, width)
                loss = criterion(outputs, Y_out)
                test_mse += loss.item() * b

            avg_test_mse = test_mse / len(test_dataset)
            scheduler.step(avg_test_mse)
            print('[epoch: {}/{}] Test MSE of the model on the {} test set: {:.4f}'.format(
                epoch + 1, num_epochs, len(test_dataset), avg_test_mse)
            )

        # Early stopping logic
        if avg_test_mse < best_loss:
            best_loss = avg_test_mse
            patience_counter = 0
            # Save the best model
            torch.save(unetlstm_model.state_dict(), os.path.join(models_dir, best_model_name))
        else:
            patience_counter += 1
            print(f"No improvement in test MSE for {patience_counter} epoch(s).")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered. Best Test MSE: {best_loss:.4f}")
            break

    ###################################
    # Counterfactual analysis
    ###################################
    # Load data
    X = torch.tensor(np.load(os.path.join(counter_data_dir, 'X.npy'))).float()
    A = torch.tensor(np.load(os.path.join(counter_data_dir, 'A.npy'))).float()
    Y = torch.tensor(np.load(os.path.join(counter_data_dir, 'Y.npy'))).float()
    A_counter = torch.tensor(np.load(os.path.join(counter_data_dir, 'A_counter.npy'))[:dim_horizon]).float().unsqueeze(1)
    n_test = X.shape[0]

    # Normalize
    X, A, Y = normalizer.normalize_batch(X, A, Y)
    Y_out = Y[:, tlen, :, :]

    x = X[:, :tlen - dim_horizon]
    A_past = A[:, :tlen - dim_horizon].reshape(n_test, tlen - dim_horizon, 1, height, width)
    Y_past = Y[:, :tlen - dim_horizon].reshape(n_test, tlen - dim_horizon, 1, height, width)
    inputs = torch.cat([x, A_past, Y_past], dim=2).to(device)
    A_counter = normalizer.normalize_A(A_counter).to(device)

    # Load best model if available
    use_best_model = True
    if use_best_model:
        best_model = unet.UNetConvLSTM(
            in_channel=in_channel,
            n_classes=1,
            dim_static=dim_horizon,
            bilinear=False,
            attention=attention
        ).to(device)
        # load the best model's weights
        state_dict = torch.load(os.path.join(models_dir, best_model_name), weights_only=True)
        best_model.load_state_dict(state_dict)
        best_model.eval()
        outputs = best_model(inputs, A_counter.squeeze(1).repeat(n_test, 1, 1, 1)).squeeze(1)
    else:
        unetlstm_model.eval()
        outputs = unetlstm_model(inputs, A_counter.squeeze(1).repeat(n_test, 1, 1, 1)).squeeze(1)

    # Denormalized loss
    denorm_outputs = normalizer.denormalize_Y(
        outputs.reshape(n_test, 1, height, width).detach().cpu()
    ).reshape(n_test, height, width)
    denorm_Y_out = normalizer.denormalize_Y(
        Y_out.reshape(n_test, 1, height, width)
    ).reshape(n_test, height, width)

    norm_loss = np.sqrt(criterion(outputs.detach().cpu(), Y_out).item())
    denorm_loss = np.sqrt(criterion(denorm_outputs, denorm_Y_out).item())
    denorm_loss_var = np.var([criterion(denorm_outputs[i], denorm_Y_out[i]).item() for i in range(n_test)], ddof=1)
    denorm_loss_sd = (0.5 / denorm_loss) * np.sqrt(denorm_loss_var)

    # Create a dictionary of parameters/results
    params_dict = {
        "Best Test RMSE": np.sqrt(best_loss),
        "Normalized Test RMSE": norm_loss,
        "Denormalized Test RMSE": denorm_loss,
        "Denormalized Test SD": denorm_loss_sd,
        "Epochs trained": epoch + 1,
        "Num Epochs": num_epochs,
        "Learning rate": learning_rate,
        "Batch size": batch_size,
        "Early stopping patience": early_stopping_patience,
        "Scheduler patience": scheduler_patience,
        "Dimension of horizon": dim_horizon
    }

    # Use the save_params function to write the dictionary to file
    save_params(params_dict, os.path.join(models_dir, f"unetlstm_best_loss_dim_horizon_{dim_horizon}.txt"))


    print("Training and counterfactual analysis complete.")
    print(f"Saved best model at {os.path.join(models_dir, best_model_name)}")
    print(f"Normalized test RMSE: {norm_loss:.4f}")
    print(f"Denormalized test RMSE: {denorm_loss:.4f}")
    print(f"Denormalized test SD: {denorm_loss_sd:.4f}")

if __name__ == "__main__":
    main()
