import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import torch_geometric as ptg
import torch

import constants
import utils
import visualization as vis

parser = argparse.ArgumentParser(description='Pre-process ushcn dataset')

parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--target", type=str, default="prcp",
        help="Which attribute to use as target (prcp/snow/snwd/tmax/tmin)")
parser.add_argument("--split_length", type=int, default=100,
        help="Number of (irregular) time points to include in each sub-series")
parser.add_argument("--train_fraction", type=float, default=0.7,
        help="Fraction of dataset to use for training (0.0-1.0)")
parser.add_argument("--test_fraction", type=float, default=0.2,
        help="Fraction of dataset to use for test (0.0-1.0)")

parser.add_argument("--weight_scaling", type=float, default=4., # Ok for Del. and 10nn
        help="Scaling to standard deviation in edge weight computation")
parser.add_argument("--graph_alg", type=str, default="knn",
        help="Algorithm to use for constructing graph")
parser.add_argument("--n_neighbors", type=int, default=10,
        help="Amount of neighbors to include in k-nn graph generation")

parser.add_argument("--plot", type=int, default=0,
        help="If plots should be made  during generation (and number of plots)")
parser.add_argument("--max_nodes_plot", type=int, default=10,
        help="Maximum nummber of nodes to plot predictions for")

config = vars(parser.parse_args())

target_ids = {
    "prcp": 0,
    "snow": 1,
    "snwd": 2,
    "tmax": 3,
    "tmin": 4,
}

assert config["target"] in target_ids, "Unknown target"
target_id = target_ids[config["target"]]

# Set random seed
_ = torch.random.manual_seed(config["seed"])
np.random.seed(config["seed"])

# File paths
dir_path = os.path.join(constants.RAW_DATA_DIR, "ushcn_daily")
data_path = os.path.join(dir_path, "daily_sporadic.csv")
data_np_path = os.path.join(dir_path, "data_array_tmp.npy")
id_map_path = os.path.join(dir_path, "centers_id_mapping.npy")
station_info_path = os.path.join(dir_path, "ushcn-stations.txt")

print("Loading data...")
try:
    data_array = np.load(data_np_path)
    print("Saved numpy array found!")
except FileNotFoundError:
    print("Loading from raw csv...")
    data_array = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    np.save(data_np_path, data_array)

station_ids = data_array[:,0].astype(int) # 0 - 1122
n_stations = station_ids.max() + 1
times = data_array[:,1]

ys = data_array[:,2:7]
masks = data_array[:,7:].astype(int)
n_features = ys.shape[-1] # 5

# Spatial map and plot
centers_id_map = np.load(id_map_path, allow_pickle=True).item()

station_info = np.genfromtxt(station_info_path, delimiter=(6, 9, 10))
center_ids = station_info[:,0].astype(int)
station_coords = station_info[:, 1:]

new_station_coords = np.zeros((n_stations, 2))
for center_id, pos in zip(center_ids, station_coords):
    if center_id in centers_id_map:
        new_id = centers_id_map[center_id]
        new_station_coords[new_id] = pos

station_pos = utils.project_eqrect(
        np.flip(new_station_coords, axis=1)) # Should be long, lat

# Create graph
print("Creating graph...")
pos_torch = torch.tensor(station_pos, dtype=torch.float32) # (N, 2)
point_data = ptg.data.Data(pos=pos_torch)
if config["graph_alg"] == "delaunay":
    graph_transforms = ptg.transforms.Compose((
        ptg.transforms.Delaunay(),
        ptg.transforms.FaceToEdge(),
    ))
    graph = graph_transforms(point_data)
    graph_name = "delaunay"
elif config["graph_alg"] == "knn":
    graph_transforms = ptg.transforms.Compose((
        ptg.transforms.KNNGraph(k=config["n_neighbors"]),
    ))
    graph = graph_transforms(point_data)
    graph_name = f"{config['n_neighbors']}nn"
else:
    assert False, "Unknown graph algorithm"

# Plot
if config["plot"]:
    vis.plot_graph(graph, show=True)

# Compute edge weights
edge_index = graph.edge_index
edge_relative_pos = pos_torch[edge_index[0,:],:] - pos_torch[edge_index[1,:],:] # (N,2)
edge_dist = torch.norm(edge_relative_pos, dim=1) #(N,)
edge_weight = utils.compute_edge_weights(edge_dist, config["weight_scaling"],
        show_plot=bool(config["plot"]))

# Find all time points
unique_times = np.sort(np.unique(times)) # (18628,)
n_times = len(unique_times)
time_index_map = {t:i for i, t in enumerate(unique_times)}

# Move into dense 3d array
mask_array = np.zeros((n_times, n_stations, n_features))
y_array = np.zeros((n_times, n_stations, n_features))

print("Reorganizing all into 3d numpy array...")
for iteration_i, (t, station_i, y, mask) in enumerate(
        zip(times, station_ids, ys, masks)):
    if iteration_i % 100000 == 0:
        print(f"iteration {iteration_i}/{ys.shape[0]}")

    time_i = time_index_map[t]
    y_array[time_i, station_i, :] = y
    mask_array[time_i, station_i, :] = mask

print(f"Observed per feature: {np.mean(mask_array, axis=(0,1))}")
print(f"Observed per node: {np.mean(mask_array, axis=(0,2))}")

# Select target
y_array = np.expand_dims(y_array[:,:,target_id], axis=-1)
mask_array = mask_array[:,:,target_id]

# To torch and split up into smaller time series
t = torch.tensor(unique_times, dtype=torch.float32) # (N_times,)
y = torch.tensor(y_array, dtype=torch.float32) # (N_times, N, d_y=1)
obs_mask = torch.tensor(mask_array, dtype=torch.float32) # (N_times, N,)

def split_tensor(a):
    splits = torch.split(a, config["split_length"] ,dim=0)
    if splits[-1].shape[0] < config["split_length"]:
        splits = splits[:-1]
    return torch.stack(splits, dim=0)

t = split_tensor(t) # (N_samples, N_t,)
y = split_tensor(y) # (N_samples, N_t, N, d_y)
obs_mask = split_tensor(obs_mask) # (N_samples, N_t, N,)

# Create time arrays
# Subtract time at start of each sub-time-series, so first obs is always at t=0
t = t - t[:,:1] # (N_samples, N_t,)
max_t_len = t.max() # Should be around 13.3
t = t/max_t_len # To get time in [0,1]
delta_t = utils.node_t_deltas(t, obs_mask)

# This has to be done for some reason, to get right shapes?
obs_mask = obs_mask.transpose(1,2) # (N_samples, N, N_T)
delta_t = delta_t.transpose(1,2) # (N_samples, N, N_T)

# Compute dataset sizes
N_data = y.shape[0]
N_train = int(config["train_fraction"]*N_data)
N_test = int(config["test_fraction"]*N_data)
N_val = N_data - (N_train + N_test)

# Print some info
ds_name = f"ushcn_{config['target']}_{graph_name}"
actual_obs_fraction = torch.mean(obs_mask)

print(f"Dataset: {ds_name}")
print(f"Fraction of nodes observed: {actual_obs_fraction}")
print(f"Number of nodes: {y.shape[2]}")
print(f"Number of time steps: {y.shape[1]}")
print(f"N_train: {N_train}")
print(f"N_val: {N_val}")
print(f"N_test: {N_test}")

# Split into subsets
data_save_dict = {
    "train": {},
    "val": {},
    "test": {},
    "edge_index": edge_index,
    "edge_weight": edge_weight,
}

shuffle_perm = torch.randperm(N_data) # Permutation to shuffle dataset
split_subsets = (lambda a: torch.split(a[shuffle_perm], (N_train, N_val, N_test)))

for ar, ar_name in zip((y, t, delta_t, obs_mask), ("y", "t", "delta_t", "mask")):
    ar_train, ar_val, ar_test = split_subsets(ar)

    data_save_dict["train"][ar_name] = ar_train
    data_save_dict["val"][ar_name] = ar_val
    data_save_dict["test"][ar_name] = ar_test

# Plot some things
# Optionally plot
if config["plot"]:
    vis.plot_preprocessed(data_save_dict, n_plots=config["plot"],
            max_nodes_plot=config["max_nodes_plot"])

# Save data
print("Saving data ...")
utils.save_data(ds_name, config, data_save_dict)

