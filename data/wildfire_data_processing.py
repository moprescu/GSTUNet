import os
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as geom
from joblib import Parallel, delayed

###############################
### Data Processing Utils   ###
###############################
def get_grid_gdf(grid_res=0.25):
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

def get_grid_data(
    data_slice, day_of_year, grid_gdf,
    feature_names=["tmax", "prec", "hum", "shrtwv_rad", "wind", "smoke", "resp", "resp_norm"], 
    grid_res = 0.25):
    
    daily_data = data_slice[data_slice.day_of_year==day_of_year][["countyname"] + feature_names]
    california_counties_daily = california_counties.merge(daily_data, on="countyname")
    grid_gdf_local = grid_gdf.copy(deep=True)
    
    for feat_name in feature_names:
        values = []
        for i, row in grid_gdf.iterrows():
            cell_poly = row.geometry
            cell_center = row["center"]
            
            # 1) Find intersecting counties
            possible_matches_index = list(counties_sindex.intersection(cell_poly.bounds))
            candidates = california_counties_daily.iloc[possible_matches_index]
            intersecting = candidates[candidates.intersects(cell_poly)]
 
            if intersecting.empty:
                # No intersection => NaN
                values.append(np.nan)
                continue

            # 2) For each intersecting county, compute distance from cell_center to county centroid
            w_sum = 0.0
            val_sum = 0.0
            epsilon = 1e-6
            for j, cty in intersecting.iterrows():
                dist = cell_center.distance(cty["centroid"])  # distance in degrees if EPSG:4326
                w = 1.0 / (dist + epsilon)
                w_sum += w
                val_sum += w * cty[feat_name]

            # Weighted average
            cell_value = val_sum / w_sum
            values.append(cell_value)

        grid_gdf[feat_name] = values
        
    num_rows = len(lats)  # from your np.arange
    num_cols = len(lons)

    interp_array = np.full((len(feature_names), num_rows, num_cols), np.nan, dtype=np.float32)

    # Need a quick way to map lat/lon back to row/col indices
    # e.g. row = i, col = j if lat=some lat, lon=some lon

    for i, rowdata in grid_gdf.iterrows():
        # Suppose you stored "grid_row", "grid_col" in the gdf if you created them
        r = rowdata["grid_row"]
        c = rowdata["grid_col"]
        for i, feat_name in enumerate(feature_names):
            interp_array[i, r, c] = rowdata[feat_name]
    return interp_array, ~np.isnan(interp_array[0]) # num_features X H X W and mask


# Define folder paths and read in data
data_folder = "./wildfire"
processed_folder = "./wildfire/processed_data"
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)
data_name = "CA_hosp_County_2018.csv"
pop_name = "CA_population.csv"
data = pd.read_csv(os.path.join(data_folder, data_name))
pop = pd.read_csv(os.path.join(data_folder, pop_name))
pop["Population"] = pop['Population'].str.replace(',', '').astype(int)
pop.rename(columns = {"Population": "pop"}, inplace=True)
data = data.merge(pop, left_on="countyname", right_on="County").drop(columns=["County", "COUNTY_1", "week2"])
data["resp_norm"] = (data["resp"] / data["pop"]) * 10000 # Cases per 10000
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

# Define parameters
lat_min, lat_max = 32.0, 42.0
lon_min, lon_max = -125.0, -114.0
grid_res = 0.25
lats = np.arange(lat_min, lat_max, grid_res)
lons = np.arange(lon_min, lon_max, grid_res)
print("Lats and Lons: ", len(lats), len(lons))

# Generate data for all days in data_slice
grid_gdf = get_grid_gdf(grid_res=0.25)
days = data_slice.day_of_year.unique()
X = np.empty((len(days), 5, len(lats), len(lons)))
A = np.empty((len(days), len(lats), len(lons)))
Y = np.empty((len(days), 2, len(lats), len(lons)))

feature_names=["tmax", "prec", "hum", "shrtwv_rad", "wind", "smoke", "resp", "resp_norm"]
post_process_idx_pairs = [
(3, 25), (3, 26),
(4, 21), (4, 22), (4, 25),
(5, 21), (5, 22), (5, 25), (5, 26),
(7, 19), (7, 20), (7, 21),
(8, 18), (8, 19), (8, 20), (8, 21),
(22, 7), (22, 8), (23, 7)
]

results = Parallel(n_jobs=-1, verbose=1)(
    delayed(get_grid_data)(data_slice, feature_names=feature_names, day_of_year=day_of_year, 
                           grid_gdf=grid_gdf, grid_res=grid_res) 
    for day_of_year in days
)

for i, (interp_array, mask) in enumerate(results):
    X[i] = interp_array[:5, :, :]
    A[i] = interp_array[5, :, :]
    Y[i] = interp_array[6:, :, :]

for pair in post_process_idx_pairs:
    X[:, :, pair[0], pair[1]] = np.nan
    A[:, pair[0], pair[1]] = np.nan
    Y[:, :, pair[0], pair[1]] = np.nan
    mask[pair[0], pair[1]] = False
    
X[:, :, ~mask] = 0
Y[:, :, ~mask] = 0
A[:, ~mask] = 0
np.save(os.path.join(processed_folder, "X.npy"), X)
np.save(os.path.join(processed_folder, "A_full.npy"), A)
np.save(os.path.join(processed_folder, "A.npy"), (A>10)*1)
np.save(os.path.join(processed_folder, "Y_resp.npy"), Y[:, 0, :, :])
np.save(os.path.join(processed_folder, "Y_resp_norm.npy"), Y[:, 1, :, :])
np.save(os.path.join(processed_folder, "mask.npy"), mask)