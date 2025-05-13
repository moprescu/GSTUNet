import os
import shutil
import sys
import argparse
import json
import numpy as np
import pandas as pd
import scipy.ndimage
import geopandas as gpd
import shapely.geometry as geom
import matplotlib.patches as mpatches

import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap 
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker

# General imports
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from wildfire_utils import get_grid_gdf, pad_func, crop_func, compute_grid_weights, compute_county_from_grid_area_weighted

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Custom imports
from data.data_loaders import SimulatedData
import models.unet as unet
import models.gstunet as gstunet
import models.stcinet as stcinet
import data.utils as data_utils
import models.utils as model_utils

matplotlib.use('Agg')

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 14

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
    parser = argparse.ArgumentParser(description="Run wildfire experiment and evaluate results.")
    parser.add_argument("--eval", action="store_true", help="Whether to evaluate the current model and skip training.", default=False)
    parser.add_argument("--overall", action="store_true", help="Whether to evaluate the overall respiratory rates. Default is to return daily mean rates.")
    parser.add_argument("--hatch", action="store_true", help="Whether to hash out the counties with small populations.")
    parser.add_argument("--random_seed", default=0, help="Random seed for bootstrap iteration.")
    args = parser.parse_args()

    eval_only = args.eval
    overall_stats = args.overall
    use_hatching = args.hatch
    random_seed = int(args.random_seed)

    # Define folder paths and read in relevant data
    data_folder = "../data/wildfire"
    processed_folder = "../data/wildfire/processed_data"
    bootstrap_folder = "../data/wildfire/results/bootstrap"
    data_name = "CA_hosp_County_2018.csv"
    pop_name = "CA_population.csv"
    data = pd.read_csv(os.path.join(data_folder, data_name))
    pop = pd.read_csv(os.path.join(data_folder, pop_name))
    pop["Population"] = pop['Population'].str.replace(',', '').astype(int)
    pop.rename(columns = {"Population": "pop"}, inplace=True)
    data = data.merge(pop, left_on="countyname", right_on="County").drop(columns=["County", "COUNTY_1", "week2"])
    data["resp_norm"] = (data["resp"] / data["pop"]) * 10000 # Cases per 100
    data_slice = data[(data.year==2018) & (data.week >= 20) & (data.week <= 48)].reset_index(drop=True)
    data_slice['day_of_year'] = pd.to_datetime(data_slice[['year', 'month', 'day']]).dt.dayofyear
    # Read in california counties geo
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    counties = gpd.read_file(url)
    # Filter counties to only include those in California (FIPS code '06')
    counties['state'] = counties['id'].str[:2]  # Extract state FIPS code
    california_counties = counties[counties['state'] == '06'].reset_index()
    california_counties["centroid"] = california_counties.geometry.centroid
    california_counties["countyname"] = california_counties["NAME"].str.title()
    counties_sindex = california_counties.sindex

    #########################
    ### Train UNet model ####
    #########################
    # Parameter counting function
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Data directory
    data_dir = "../data/wildfire/processed_data"
    # Saved models directory
    models_dir = "../data/wildfire/processed_data/models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        # Figure directory
    figs_dir = "../data/wildfire/figures"
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    
    lat_min, lat_max = 32.0, 42.0
    lon_min, lon_max = -125.0, -114.0
    grid_res = 0.25
    lats = np.arange(lat_min, lat_max, grid_res)
    lons = np.arange(lon_min, lon_max, grid_res)
    grid_gdf = get_grid_gdf(lats, lons, grid_res=0.25)
    ### Make weight mask for grid cells to be used in training
    weight_mask, _ = compute_grid_weights(
        california_counties.merge(pop, left_on="NAME", right_on="County").drop(columns=["County"]), 
        grid_gdf)
    weight_mask[np.isnan(weight_mask)] = 0
    weight_mask /= weight_mask.sum()
    # For reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    np_mask = np.load(os.path.join(data_dir, "mask.npy"))*1
    np_weights = np_mask * np.sqrt(weight_mask)

    mask = torch.tensor(np_weights).to(device)
    mask.shape # 40 x 44 expected 
    padded_mask = pad_func(mask)
    padded_mask = padded_mask.unsqueeze(0).unsqueeze(1)
    # Set parameters and read in processed data
    dim_horizon = 10
    history_len = 10
    tlen = history_len + dim_horizon
    target_col = "resp"

    # Set output before loading dataset
    shutil.copy(os.path.join(data_dir, f"Y_{target_col}.npy"), os.path.join(data_dir, "Y.npy"))

    # Datasets, use Carr fire for validation
    train_dataset = SimulatedData(processed_folder, n_train = 50, train = False, tlen = tlen)
    #train_dataset = data_utils.BootstrappedDataset(train_dataset, random_seed=random_seed)
    
    test_dataset = SimulatedData(processed_folder, n_train = 50, train = True, tlen = tlen)
    # Counterfactual data
    h0, w0 = train_dataset[0][0].shape[2], train_dataset[0][0].shape[3]
    height, width = 48, 48
    A_counter = torch.zeros((dim_horizon, height, width)).float().unsqueeze(1)

    # Define model
    # Hyper parameters
    num_epochs = 40
    learning_rate = 0.0005
    batch_size = 4
    early_stopping_patience = 10

    in_channel = 7
    h_size = 16
    fc_layer_sizes = [8] 
    dim_treatments = 1
    dim_outcome = 1
    use_constant_feature = False
    attention = True
    best_model_name = f'stcinet_dim_horizon_{dim_horizon}_wildfire_bootstrap_{random_seed}.pth'

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)

    # Normalization
    normalizer = data_utils.DataNormalizer(train_loader)
    train_loader_normalized, test_loader_normalized = normalizer.normalize(train_loader, test_loader)
    A_counter = normalizer.normalize_A(A_counter).to(device)
    
    model = stcinet.STCINet(
                in_channel=in_channel,
                n_classes=1,
                dim_static=dim_horizon,
                bilinear=False,
                height=height,
                width=width
            ).to(device)

    print('Number of model parameters: {}'.format(count_parameters(model)))

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Early stopping parameters
    early_stopping_patience = early_stopping_patience
    best_loss = float('inf')
    patience_counter = 0

    ###################################
    # Training loop
    ###################################
    for epoch in range(num_epochs):
        model.train()
        for i, (x, A, Y) in enumerate(train_loader_normalized):        
            b = x.size(0)
            x = x[:, :-1]

            x_past = x[:, :tlen - dim_horizon]  # shape [B, Tpast, Xchannels, H, W]
            A_past = A[:, :tlen - dim_horizon].reshape(b, tlen - dim_horizon, 1, h0, w0)
            Y_past = Y[:, :tlen - dim_horizon].reshape(b, tlen - dim_horizon, 1, h0, w0)

            A_curr = A[:, tlen - dim_horizon:-1].to(device)
            Y_out = Y[:, -1].to(device)
            inputs = torch.cat([x_past, A_past, Y_past], dim=2).to(device)

            # Pad inputs
            inputs = pad_func(inputs) 
            A_curr = pad_func(A_curr)
            outputs = model(inputs, A_curr).reshape(b, -1, height, width)
            Y_out = pad_func(Y_out.unsqueeze(1))

            outputs_flat = outputs.view(b, -1)         # [B, H_p * W_p]
            Y_out_flat = Y_out.view(b, -1)

            # Pad mask and flatten  # [1, 1, 48, 48]
            padded_mask_flat = padded_mask.repeat(b, 1, 1, 1).view(b, -1)  # [B, 2304]

            masked_pred = outputs_flat * padded_mask_flat
            masked_true = Y_out_flat * padded_mask_flat
            this_loss = criterion(masked_pred, masked_true)
            loss = this_loss * (height * width)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('[epoch: {}/{}] [step {}/{}] MSE: {:.4f}'.format(epoch+1, num_epochs, i, len(train_loader_normalized), loss.item()))

        # Test the model
        model.eval()  # eval mode 
        with torch.no_grad():
            test_mse = 0
            for i, (x, A, Y) in enumerate(test_loader_normalized):
                b = x.size(0)
                x = x[:, :-1]

                x_past = x[:, :tlen - dim_horizon]  # shape [B, Tpast, Xchannels, H, W]
                A_past = A[:, :tlen - dim_horizon].reshape(b, tlen - dim_horizon, 1, h0, w0)
                Y_past = Y[:, :tlen - dim_horizon].reshape(b, tlen - dim_horizon, 1, h0, w0)

                A_curr = A[:, tlen - dim_horizon:-1].to(device)
                Y_out = Y[:, -1].to(device)
                inputs = torch.cat([x_past, A_past, Y_past], dim=2).to(device)

                # Pad inputs
                inputs = pad_func(inputs) 
                A_curr = pad_func(A_curr)
                outputs = model(inputs, A_curr).reshape(b, -1, height, width)
                Y_out = pad_func(Y_out.unsqueeze(1))

                outputs_flat = outputs.view(b, -1)         # [B, H_p * W_p]
                Y_out_flat = Y_out.view(b, -1)

                # Pad mask and flatten  # [1, 1, 48, 48]
                padded_mask_flat = padded_mask.repeat(b, 1, 1, 1).view(b, -1)  # [B, 2304]

                masked_pred = outputs_flat * padded_mask_flat
                masked_true = Y_out_flat * padded_mask_flat
                this_loss = criterion(masked_pred, masked_true)
                this_loss = this_loss * (height * width)
                test_mse += this_loss.item() * b

            avg_test_mse = test_mse / len(test_dataset)
            scheduler.step(avg_test_mse)
            print('[epoch: {}/{}] Test MSE of the model on the {} test set: {:.4f}'.format(epoch+1, num_epochs, len(test_dataset), test_mse / len(test_dataset)))


        # Early stopping logic
        if avg_test_mse < best_loss:
            best_loss = avg_test_mse
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(models_dir, best_model_name))
        else:
            patience_counter += 1
            print(f"No improvement in test MSE for {patience_counter} epoch(s).")
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered. Best Test MSE: {best_loss:.4f}")
            break

    ### Counterfactual analysis
    aux = np.zeros((10, h0, w0))
    Y_outs = np.zeros((10, h0, w0))
    for i in range(10):
        tlen = history_len + dim_horizon
        target_idx = 312+i-137-(10-dim_horizon) # 324 for the end of the Camp Fire
        
        X = torch.tensor(np.load(os.path.join(data_dir, 'X.npy'))).float()
        A = torch.tensor(np.load(os.path.join(data_dir, 'A.npy'))).float()
        Y = torch.tensor(np.load(os.path.join(data_dir, 'Y.npy'))).float()
        Y_out = Y[target_idx, :, :]
        
        X, A, Y = normalizer.normalize_batch(X.unsqueeze(0), A.unsqueeze(0), Y.unsqueeze(0))
        
        x = X[:, target_idx-dim_horizon-history_len:target_idx-dim_horizon]
        A_past = A[:, target_idx-dim_horizon-history_len:target_idx-dim_horizon].reshape(1, history_len, 1, h0, w0)
        Y_past = Y[:, target_idx-dim_horizon-history_len:target_idx-dim_horizon].reshape(1, history_len, 1, h0, w0)
        
        inputs = torch.cat([x, A_past, Y_past], dim=2).to(device)
        inputs = pad_func(inputs)
        
        use_best_model = True
        if use_best_model:
            # Load best model
            best_model = stcinet.STCINet(
                in_channel=in_channel,
                n_classes=1,
                dim_static=dim_horizon,
                bilinear=False,
                height=height,
                width=width).to(device)
            state_dict = torch.load(os.path.join(models_dir, best_model_name), weights_only=True)
            best_model.load_state_dict(state_dict)
            best_model.eval()
            # Outputs from the best model
            outputs = best_model.forward(inputs, A_counter.squeeze(1).unsqueeze(0)).squeeze(1).reshape(height, width)
        else:
            # Outputs from the best model
            model.eval()
            outputs = model.forward(inputs, A_counter.squeeze(1).unsqueeze(0)).reshape(height, width)
        denorm_outputs = normalizer.denormalize_Y(outputs.reshape(1, 1, height, width).detach().cpu()).reshape(height, width)
        outputs_cropped = crop_func(denorm_outputs)
        aux[i] = outputs_cropped.numpy()
        Y_outs[i] = Y_out = Y_out.numpy()
                
    ### Analysis of counterfactuals
    denorm_outputs = normalizer.denormalize_Y(outputs.reshape(1, 1, height, width).detach().cpu()).reshape(height, width)
    outputs_cropped = crop_func(denorm_outputs)
    outputs_cropped = aux.sum(axis=0)
    Y_out = Y_outs.sum(axis=0)
    Y_out[mask.cpu()==0] = np.nan
    outputs_cropped[mask.cpu()==0] = np.nan
    counties_gdf = compute_county_from_grid_area_weighted(
        california_counties.copy(deep=True),
        grid_gdf.copy(deep=True),
        Y_out - outputs_cropped,
        "resp")
    data_slice_mean = data_slice[(data_slice.day_of_year <= target_idx+137) & (data_slice.day_of_year >= target_idx+137-10)].groupby("countyname").agg(
        {"smoke": "mean"}
    ).reset_index()
    counties_gdf = counties_gdf.merge(data_slice[data_slice.day_of_year == target_idx+137][["countyname", "resp", "pop"]], on="countyname")
    counties_gdf = counties_gdf.merge(data_slice_mean, on="countyname")
    exposed_counties = np.array(['Glenn', 'Lake', 'Napa', 'Stanislaus', 'Yuba', 'Butte',
       'Mendocino', 'San Francisco', 'Yolo', 'San Mateo', 'Santa Cruz',
       'Solano', 'Sutter', 'Tehama', 'Contra Costa', 'Kings', 'Colusa',
       'Merced', 'San Joaquin', 'Tulare', 'Marin', 'Placer', 'Fresno',
       'Sacramento', 'Alameda', 'Sonoma', 'Santa Clara'])
    exposed_filter = counties_gdf.countyname.isin(exposed_counties)

    # Compute additional respiratory hospitalizations and save to file
    additional_resp = int(counties_gdf.loc[exposed_filter, "resp_x"].sum())
    print("Additional respiratory hospitalizations over 10 days: ", additional_resp)
    os.makedirs(bootstrap_folder, exist_ok=True)
    filename = os.path.join(bootstrap_folder, f"stcinet_{random_seed}.txt")
    d = {}
    d["additional_resp"] = additional_resp
    for county in exposed_counties:
        d[county] = {
            "additional_resp": int(counties_gdf.loc[counties_gdf.countyname == county, "resp_x"].values[0]),
            "pop": int(counties_gdf.loc[counties_gdf.countyname == county, "pop"].values[0])}
    d["train_epochs"] = epoch
    save_params(d, filename)

    # Remove unexposed counties
    counties_gdf.loc[~exposed_filter, "resp_y"] = np.nan
    counties_gdf.loc[~exposed_filter, "resp_x"] = np.nan

    if overall_stats:
        counties_gdf["resp_x"] = counties_gdf["resp_x"]/counties_gdf["pop"]*10000
        counties_gdf["resp_y"] = counties_gdf["resp_y"]/counties_gdf["pop"]*10000
    else:
        counties_gdf["resp_x"] = counties_gdf["resp_x"]/counties_gdf["pop"]*10000/10
        counties_gdf["resp_y"] = counties_gdf["resp_y"]/counties_gdf["pop"]*10000/10

    # Create a figure 
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    class MidpointNormalize(Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
            x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    norm_resp = TwoSlopeNorm(
        vmin=np.percentile(counties_gdf["resp_x"][~np.isnan(counties_gdf["resp_x"])], 0), 
        vcenter=0, 
        vmax=np.percentile(counties_gdf["resp_x"][~np.isnan(counties_gdf["resp_x"])], 99))
    norm_resp = MidpointNormalize(midpoint = 0, vmin=np.percentile(counties_gdf["resp_x"][~np.isnan(counties_gdf["resp_x"])], 0), 
                                vmax=np.percentile(counties_gdf["resp_x"][~np.isnan(counties_gdf["resp_x"])], 100))
    cmap = plt.cm.RdBu_r

    cmap.set_bad(color='gray')

    im0 = counties_gdf.plot(
        column="resp_x",
        ax=axes,
        edgecolor='black',
        cmap=cmap,
        norm=norm_resp,
        missing_kwds={'color': 'lightgray'} )

    axes.set_title(
        "Camp Fire: Factual vs. Counterfactual Respiratory Illness",
        fontsize=14, pad=10
    )
    cbar0 = fig.colorbar(axes.collections[0], ax=axes, shrink=0.85)
    cbar0.set_label(
        "Factual - Counterfactual\nRespiratory Illness Incidence (cases per 10,000)",
        fontsize=14
    )

    # Hash out small population counties
    small_pop_threshold = 60000
    counties_gdf["small_pop"] = (counties_gdf["pop"] < small_pop_threshold) & exposed_filter
    hatching_df = counties_gdf[counties_gdf["small_pop"]]
    hatching = False
    if use_hatching:
        hatching_df.plot(
            ax=axes,
            hatch='///',
            facecolor='none',
            color=None,
            edgecolor='black',
            linewidth=1,
            alpha=1.0,
            zorder=2  
        )

    # Get the current tick positions
    current_ticks = cbar0.get_ticks()
    # Convert to list for easier manipulation
    current_ticks = list(current_ticks)
    current_ticks=current_ticks[1:]
    if overall_stats:
        # Now manually set these tick positions and their labels:
        current_ticks[-1] = 35
        cbar0.set_ticks(current_ticks)
        cbar0.set_ticklabels([f"{int(tick)}" for tick in current_ticks])
    else:
        # Now manually set these tick positions and their labels:
        current_ticks[-1] = 3.5
        cbar0.set_ticks(current_ticks)
        cbar0.set_ticklabels([f"{tick:.1f}" for tick in current_ticks])
    plt.tight_layout()
    suffix = ""
    if overall_stats:
        suffix = "_overall"
    else:
        suffix = "_daily"
    if use_hatching:
        suffix += "_hashed"

    plt.xlabel(r"Longitude ($^o$)")
    plt.ylabel(r"Latitude ($^o$)")
    #plt.savefig(os.path.join(figs_dir, f"Factual_vs_counterfactual_respiratory_illness{suffix}.pdf"), dpi=200, bbox_inches="tight")
    #plt.savefig(os.path.join(figs_dir, f"Factual_vs_counterfactual_respiratory_illness{suffix}.png"), dpi=200, bbox_inches="tight")

    print(figs_dir)
    print(f"Counterfactual analysis completed.")

if __name__ == "__main__":
    main()