import numpy as np
import geopandas as gpd
import torch.nn.functional as F
import shapely.geometry as geom

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

