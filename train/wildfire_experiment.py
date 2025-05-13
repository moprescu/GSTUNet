import os
import shutil
import sys
import argparse
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Custom imports
from data.data_loaders import SimulatedData
import models.unet as unet
import models.gstunet as gstunet
import data.utils as data_utils
import models.utils as model_utils

matplotlib.use('Agg')

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 18

#######################################
### Training and Experiment Utils   ###
#######################################
def get_grid_gdf(lats, lons, grid_res=0.25):
    """Obtains grid geometry"""
    cells = []
    grid_rows = []
    grid_cols = []

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # Each cell is a square polygon of size grid_res deg
            # corners: (lon,lat), (lon+grid_res,lat), (lon+grid_res,lat+grid_res), (lon,lat+grid_res)
            poly = geom.Polygon([
                (lon,     lat),
                (lon+grid_res, lat),
                (lon+grid_res, lat+grid_res),
                (lon,     lat+grid_res)
            ])
            cells.append(poly)
            grid_rows.append(i)
            grid_cols.append(j)

    grid_gdf = gpd.GeoDataFrame({
        "grid_row": grid_rows,
        "grid_col": grid_cols,
        "geometry": cells
    }, crs="EPSG:4326")
    grid_gdf["center"] = grid_gdf.geometry.centroid
    return grid_gdf

def pad_func(x):
    """
    x: Tensor of shape [B, C, 40, 44]
    returns: Tensor of shape [B, C, 48, 48]
    """
    # F.pad order = (left, right, top, bottom)
    # Here: left=2, right=2, top=4, bottom=4
    x_padded = F.pad(x, (2, 2, 4, 4))
    return x_padded

def crop_func(x):
    """
    x: Tensor of shape [B, C, 48, 48]
    returns: Tensor of shape [B, C, 40, 44]
    """
    # PyTorch tensor shape convention: [B, C, H, W]
    # remove 4 rows top/bottom, 2 columns left/right
    return x[4:-4, 2:-2]

def compute_grid_weights(counties_gdf, grid_gdf):
    """
    For each grid cell, assign it to a county based on which county has the maximum intersection area.
    Then, for each grid cell, assign a weight equal to:
        (county_pop / total_pop) / (# of grid cells assigned to that county)
        
    Parameters
    ----------
    counties_gdf : GeoDataFrame
        Contains the county polygons and a population column "pop".
    grid_gdf : GeoDataFrame
        Contains grid cells with columns "grid_row", "grid_col", and "geometry".
        
    Returns
    -------
    grid_weight_array : ndarray
        A 2D NumPy array of shape (num_rows, num_cols) such that each cell 
        contains its computed weight.
    assignments : list
        A list (of the same length as grid_gdf) of the assigned county index for each grid cell.
    """
    # Create a spatial index for counties for faster queries.
    counties_sindex = counties_gdf.sindex

    # Prepare a list to store the dominant county assignment (by county index) for each grid cell.
    dominant_county_assignments = []  # same length as grid_gdf
    # Also store the (row, col) for each grid cell
    grid_rows = grid_gdf["grid_row"].tolist()
    grid_cols = grid_gdf["grid_col"].tolist()
    
    # Loop over every grid cell and determine its dominant county.
    for idx, cell in grid_gdf.iterrows():
        cell_poly = cell.geometry

        # Use the spatial index to query possible intersecting counties.
        possible_idx = list(counties_sindex.intersection(cell_poly.bounds))
        candidates = counties_gdf.iloc[possible_idx]

        # Filter candidates by actual intersection.
        intersecting = candidates[candidates.intersects(cell_poly)]
        
        if intersecting.empty:
            dominant_county_assignments.append(None)
            continue

        # For each candidate, compute the intersection area.
        max_area = -1
        dominant_county = None
        for county_idx, county_row in intersecting.iterrows():
            inter_poly = cell_poly.intersection(county_row.geometry)
            if not inter_poly.is_empty:
                area = inter_poly.area
                if area > max_area:
                    max_area = area
                    dominant_county = county_idx  # using the county's index as its ID
        dominant_county_assignments.append(dominant_county)

    # Add the dominant county assignment to grid_gdf (optional, for debugging)
    grid_gdf = grid_gdf.copy()
    grid_gdf["dominant_county"] = dominant_county_assignments

    # Count the number of grid cells assigned to each county.
    county_cell_counts = {}
    for county_idx in dominant_county_assignments:
        if county_idx is None:
            continue
        county_cell_counts[county_idx] = county_cell_counts.get(county_idx, 0) + 1

    # Compute the total population of all counties (only consider those with at least one assigned cell).
    total_pop = 0.0
    for county_idx, cell_count in county_cell_counts.items():
        total_pop += counties_gdf.loc[county_idx, "pop"]

    # Now compute the weight for each grid cell.
    # Weight for a grid cell = (county_pop/total_pop) / (number of grid cells assigned to that county)
    num_rows = int(grid_gdf["grid_row"].max() + 1)
    num_cols = int(grid_gdf["grid_col"].max() + 1)
    grid_weight_array = np.full((num_rows, num_cols), np.nan, dtype=float)

    for idx, cell in grid_gdf.iterrows():
        r = int(cell["grid_row"])
        c = int(cell["grid_col"])
        county_idx = cell["dominant_county"]
        if np.isnan(county_idx):
            continue  # No intersection, weight remains NaN.
        county_pop = counties_gdf.loc[county_idx, "pop"]
        cell_count = county_cell_counts.get(county_idx, 1)  # avoid division by zero
        # Compute grid cell weight.
        #weight = (county_pop / total_pop) / cell_count
        weight = 1 / cell_count
        grid_weight_array[r, c] = weight
        
    return grid_weight_array, dominant_county_assignments

def compute_county_from_grid_area_weighted(
    counties_gdf,
    grid_gdf,
    grid_values,
    feat_name
):
    # Reproject if needed to a projected CRS (e.g. EPSG:3310 for California)
    # counties_gdf = counties_gdf.to_crs("EPSG:3310")
    # grid_gdf = grid_gdf.to_crs("EPSG:3310")

    grid_sindex = grid_gdf.sindex
    counties_gdf[feat_name] = np.nan

    for i, county_row in counties_gdf.iterrows():
        cty_poly = county_row.geometry

        possible_matches_index = list(grid_sindex.intersection(cty_poly.bounds))
        candidates = grid_gdf.iloc[possible_matches_index]
        intersecting = candidates[candidates.intersects(cty_poly)]

        if intersecting.empty:
            print("Intersection is empty for this county: ", county_row["countyname"])
            continue

        total_area = 0.0
        weighted_sum = 0.0

        for j, grid_row in intersecting.iterrows():
            cell_poly = grid_row.geometry
            inter_poly = cty_poly.intersection(cell_poly)

            if not inter_poly.is_empty:
                inter_area = inter_poly.area
                val = grid_values[grid_row["grid_row"], grid_row["grid_col"]]
                if not np.isnan(val):
                    total_area += inter_area
                    weighted_sum += val * inter_area

        if total_area > 0:
            counties_gdf.at[i, feat_name] = weighted_sum / total_area

    return counties_gdf

def main():
    """
    Main function that trains and tests a UNetConvLSTM model using the specified
    hyperparameters and data. It then performs a counterfactual analysis.
    """
    parser = argparse.ArgumentParser(description="Run wildfire experiment and evaluate results.")
    parser.add_argument("--eval", action="store_true", help="Whether to evaluate the current model and skip training.", default=False)
    parser.add_argument("--overall", action="store_true", help="Whether to evaluate the overall respiratory rates. Default is to return daily mean rates.")
    parser.add_argument("--hatch", action="store_true", help="Whether to hash out the counties with small populations.")
    args = parser.parse_args()

    eval_only = args.eval
    overall_stats = args.overall
    use_hatching = args.hatch

    # Define folder paths and read in relevant data
    data_folder = "../data/wildfire"
    processed_folder = "../data/wildfire/processed_data"
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

    ############################
    ### Train GSTUNet model ####
    ############################
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
    test_dataset = SimulatedData(processed_folder, n_train = 50, train = True, tlen = tlen)
    # Counterfactual data
    h0, w0 = train_dataset[0][0].shape[2], train_dataset[0][0].shape[3]
    height, width = 48, 48
    A_counter = torch.zeros((dim_horizon, height, width)).float().unsqueeze(1)

    # Define model
    # Hyper parameters
    num_epochs = 100
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
    best_model_name = f'gstunet_dim_horizon_{dim_horizon}_wildfire.pth'

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

    model = gstunet.GSTUNet(in_channel = in_channel, h_size=h_size, A_counter = A_counter, fc_layer_sizes=fc_layer_sizes, 
                            dim_treatments = dim_treatments, dim_outcome = dim_outcome, dim_horizon = dim_horizon, 
                            use_constant_feature = use_constant_feature, attention=attention).to(device)

    print('Number of model parameters: {}'.format(count_parameters(model)))

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Early stopping parameters
    best_loss = float('inf') 
    patience_counter = 0 

    if not eval_only:
        # Train the model
        #### Warm start via slowly increasing the horizon
        num_epochs_warm_start = 5
        for current_head_idx in np.arange(dim_horizon-1, -1, -1):
            print(f"\nHead idx: {current_head_idx}...")
            for epoch in range(num_epochs_warm_start):
                model.train()
                for i, (x, A, Y) in enumerate(train_loader_normalized):
                    b = x.size(0)
                    x = x[:, :-1]

                    A_past = A[:, :-1].reshape(b, tlen, 1, h0, w0)
                    Y_past = Y[:, :-1].reshape(b, tlen, 1, h0, w0)
                    A_curr = A[:, -2].reshape(b, 1, h0, w0)
                    A_curr = A_curr.to(device)
                    
                    # Pad mask and flatten  # [1, 1, 48, 48]
                    padded_mask_flat = padded_mask.repeat(b, 1, 1, 1).view(b, -1)  # [B, 2304]

                    # Forward pass
                    # Calculate loss iteratively
                    Y_out = pad_func(Y[:, -1]).reshape(b, -1).to(device)
                    target = Y_out
                    loss = torch.zeros((), device=device)
                    for head_idx in np.arange(dim_horizon-1, current_head_idx-1, -1):
                        # Trim dataset
                        A_curr = A_past[:, tlen-dim_horizon+head_idx, :, :, :].to(device)
                        A_past_copy = A_past.clone() 
                        inputs = torch.cat([x[:, :tlen-dim_horizon+head_idx+1, :, :, :], 
                                            A_past_copy[:, :tlen-dim_horizon+head_idx+1, :, :, :],
                                            Y_past[:, :tlen-dim_horizon+head_idx+1, :, :, :]
                                        ], dim=2).to(device)
                        # Pad A_curr and inputs
                        A_curr = pad_func(A_curr)
                        inputs = pad_func(inputs)
                        output = model.forward_grad(inputs, A_curr, head_idx)
                        this_loss = criterion(output*padded_mask_flat, target*padded_mask_flat)
                        this_loss = this_loss * (height * width)
                        """
                        if i==0:
                            print(f"Train step 0: Head index = {head_idx}, Loss = {this_loss:0.4f}.")
                        """
                        loss += this_loss
                        target = model.forward_nograd(inputs, head_idx) # for next step

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if i % 10 == 0:
                        print('[epoch: {}/{}] [step {}/{}] MSE: {:.4f}'.format(epoch+1, num_epochs_warm_start, i, len(train_loader_normalized), loss.item()))

            # Test the model
            model.eval()  # eval mode 
            with torch.no_grad():
                test_mse = 0
                for i, (x, A, Y) in enumerate(test_loader_normalized):
                    b = x.size(0)
                    x = x[:, :-1]

                    A_past = A[:, :-1].reshape(b, tlen, 1, h0, w0)
                    Y_past = Y[:, :-1].reshape(b, tlen, 1, h0, w0)
                    A_curr = A[:, -2].reshape(b, 1, h0, w0)
                    A_curr = A_curr.to(device)
                    
                    # Pad mask and flatten
                    padded_mask_flat = padded_mask.repeat(b, 1, 1, 1).view(b, -1)  # [B, 2304]

                    # Forward pass
                    # Calculate loss iteratively
                    Y_out = pad_func(Y[:, -1]).reshape(b, -1).to(device)
                    target = Y_out
                    loss = torch.zeros((), device=device)
                    for head_idx in np.arange(dim_horizon-1, -1, -1):
                        # Trim dataset
                        A_curr = A_past[:, tlen-dim_horizon+head_idx, :, :, :].to(device)
                        A_past_copy = A_past.clone()
                        inputs = torch.cat([x[:, :tlen-dim_horizon+head_idx+1, :, :, :], 
                                            A_past_copy[:, :tlen-dim_horizon+head_idx+1, :, :, :],
                                            Y_past[:, :tlen-dim_horizon+head_idx+1, :, :, :]
                                        ], dim=2).to(device)
                        # Pad A_curr and inputs
                        A_curr = pad_func(A_curr)
                        inputs = pad_func(inputs)
                        output = model.forward_grad(inputs, A_curr, head_idx)
                        this_loss = criterion(output*padded_mask_flat, target*padded_mask_flat)
                        this_loss = this_loss * (height * width)
                        """
                        if i==0:
                            print(f"Test step 0: Head index = {head_idx}, Loss = {this_loss:0.4f}.")
                        """
                        loss += this_loss
                        target = model.forward_nograd(inputs, head_idx) # for next step
                    test_mse += loss.item() * b

                avg_test_mse = test_mse / len(test_dataset)
                print('[epoch: {}/{}] Test MSE of the model on the {} test set: {:.4f}'.format(epoch+1, num_epochs_warm_start, len(test_dataset), test_mse / len(test_dataset)))

        ### Train jointly
        learning_rate = 0.0005
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

        # Early stopping parameters
        best_loss = float('inf') 
        patience_counter = 0
        early_stopping_patience = 10

        # Train the model
        total_step = len(train_loader_normalized)
        for epoch in range(num_epochs):
            model.train()
            for i, (x, A, Y) in enumerate(train_loader_normalized):        
                b = x.size(0)
                x = x[:, :-1]

                A_past = A[:, :-1].reshape(b, tlen, 1, h0, w0)
                Y_past = Y[:, :-1].reshape(b, tlen, 1, h0, w0)
                A_curr = A[:, -2].reshape(b, 1, h0, w0)
                A_curr = A_curr.to(device)
                
                # Pad mask and flatten  # [1, 1, 48, 48]
                padded_mask_flat = padded_mask.repeat(b, 1, 1, 1).view(b, -1)  # [B, 2304]

                # Forward pass
                # Calculate loss iteratively
                Y_out = pad_func(Y[:, -1]).reshape(b, -1).to(device)
                target = Y_out
                loss = torch.zeros((), device=device)
                for head_idx in np.arange(dim_horizon-1, -1, -1):
                    # Trim dataset
                    A_curr = A_past[:, tlen-dim_horizon+head_idx, :, :, :].to(device)
                    A_past_copy = A_past.clone()
                    #A_past_copy[:, tlen-dim_horizon+head_idx, :, :, :] = 0
                    inputs = torch.cat([x[:, :tlen-dim_horizon+head_idx+1, :, :, :], 
                                        A_past_copy[:, :tlen-dim_horizon+head_idx+1, :, :, :],
                                        Y_past[:, :tlen-dim_horizon+head_idx+1, :, :, :]
                                    ], dim=2).to(device)
                    # Pad A_curr and inputs
                    A_curr = pad_func(A_curr)
                    inputs = pad_func(inputs)
                    output = model.forward_grad(inputs, A_curr, head_idx)
                    this_loss = criterion(output*padded_mask_flat, target*padded_mask_flat)
                    this_loss = this_loss * (height * width)
                    if i==0:
                        print(f"Train step 0: Head index = {head_idx}, Loss = {this_loss:0.4f}.")
                    loss += this_loss
                    target = model.forward_nograd(inputs, head_idx) # for next step

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

                    A_past = A[:, :-1].reshape(b, tlen, 1, h0, w0)
                    Y_past = Y[:, :-1].reshape(b, tlen, 1, h0, w0)
                    A_curr = A[:, -2].reshape(b, 1, h0, w0)
                    A_curr = A_curr.to(device)

                    # Pad mask and flatten  # [1, 1, 48, 48]
                    padded_mask_flat = padded_mask.repeat(b, 1, 1, 1).view(b, -1)  # [B, 2304]

                    # Forward pass
                    # Calculate loss iteratively
                    Y_out = pad_func(Y[:, -1]).reshape(b, -1).to(device)
                    target = Y_out
                    loss = torch.zeros((), device=device)
                    for head_idx in np.arange(dim_horizon-1, -1, -1):
                        # Trim dataset
                        A_curr = A_past[:, tlen-dim_horizon+head_idx, :, :, :].to(device)
                        A_past_copy = A_past.clone()
                        #A_past_copy[:, tlen-dim_horizon+head_idx, :, :, :] = 0
                        inputs = torch.cat([x[:, :tlen-dim_horizon+head_idx+1, :, :, :], 
                                            A_past_copy[:, :tlen-dim_horizon+head_idx+1, :, :, :],
                                            Y_past[:, :tlen-dim_horizon+head_idx+1, :, :, :]
                                        ], dim=2).to(device)
                        # Pad A_curr and inputs
                        A_curr = pad_func(A_curr)
                        inputs = pad_func(inputs)
                        output = model.forward_grad(inputs, A_curr, head_idx)
                        this_loss = criterion(output*padded_mask_flat, target*padded_mask_flat)
                        this_loss = this_loss * (height * width)
                        if i==0:
                            print(f"Test step 0: Head index = {head_idx}, Loss = {this_loss:0.4f}.")
                        loss += this_loss
                        target = model.forward_nograd(inputs, head_idx) # for next step
                    test_mse += loss.item() * b

                avg_test_mse = test_mse / len(test_dataset)
                scheduler.step(avg_test_mse)
                print('[epoch: {}/{}] Test MSE of the model on the {} test set: {:.4f}'.format(epoch+1, num_epochs, len(test_dataset), test_mse / len(test_dataset)))

            #scheduler.step()
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
            best_model = gstunet.GSTUNet(in_channel = in_channel, h_size=h_size, A_counter = A_counter, fc_layer_sizes=fc_layer_sizes, 
                                        dim_treatments = dim_treatments, dim_outcome = dim_outcome, dim_horizon = dim_horizon, 
                                        use_constant_feature = use_constant_feature, attention=attention).to(device)
            state_dict = torch.load(os.path.join(models_dir, best_model_name), weights_only=True)
            best_model.load_state_dict(state_dict)
            best_model.eval()
            # Outputs from the best model
            outputs = best_model.forward(inputs).reshape(height, width)
        else:
            # Outputs from the best model
            model.eval()
            outputs = model.forward(inputs).reshape(height, width)
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
    print("Additional respiratory hospitalizations over 10 days: ", int(counties_gdf.loc[exposed_filter, "resp_x"].sum()))
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
        "Camp Fire Respiratory Illness",
        fontsize=18, pad=10
    )
    cbar0 = fig.colorbar(axes.collections[0], ax=axes, shrink=0.85)
    cbar0.set_label(
        "Factual - Counterfactual Incidence\n(cases per 10,000)",
        fontsize=18
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
    plt.xticks([-124, -122, -120, -118, -116, -114], [-124, -122, -120, -118, -116, -114], fontsize=18)
    plt.ylabel(r"Latitude ($^o$)")
    plt.savefig(os.path.join(figs_dir, f"Factual_vs_counterfactual_respiratory_illness{suffix}.pdf"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(figs_dir, f"Factual_vs_counterfactual_respiratory_illness{suffix}.png"), dpi=200, bbox_inches="tight")

    print(figs_dir)
    print(f"Counterfactual analysis completed.")

if __name__ == "__main__":
    main()