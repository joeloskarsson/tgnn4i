import torch
import os
import torch_geometric as ptg
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch_geometric_temporal as ptgt

import utils
import constants
import visualization as vis

parser = argparse.ArgumentParser(description='Pre-process dataset')

parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--dataset", type=str, default="bay",
        help="Datset to pre-process")
parser.add_argument("--plot", type=int, default=0,
        help="If plots should be made  during generation (and number of plots)")
parser.add_argument("--max_nodes_plot", type=int, default=10,
        help="Maximum nummber of nodes to plot predictions for")

parser.add_argument("--train_fraction", type=float, default=0.7,
        help="Fraction of dataset to use for training (0.0-1.0)")
parser.add_argument("--test_fraction", type=float, default=0.2,
        help="Fraction of dataset to use for test (0.0-1.0)")

parser.add_argument("--obs_nodes", type=float, default=0.25,
        help="Percentage of nodes observed at each timestep (in [0,1]')")
parser.add_argument("--subsample_time", type=float, default=0.25,
        help="How much to subsample the time with, before masking observations")

config = vars(parser.parse_args())

assert (config["train_fraction"]+config["test_fraction"]
        ) < 1.0, "Invalid subset fractions"

# Set random seed
_ = torch.random.manual_seed(config["seed"])
np.random.seed(config["seed"])

# Load dataset
raw_path = os.path.join(constants.RAW_DATA_DIR, config["dataset"])
if config["dataset"] == "bay":
    ds_loader = ptgt.dataset.PemsBayDatasetLoader(raw_data_dir=raw_path)

    edge_index, edge_weight = ptg.utils.dense_to_sparse(ds_loader.A)
    X = ds_loader.X # Shape (N, n_features, N_timeperiod)
    d_y = 1

elif config["dataset"] == "la":
    ds_loader = ptgt.dataset.METRLADatasetLoader(raw_data_dir=raw_path)

    edge_index, edge_weight = ptg.utils.dense_to_sparse(ds_loader.A)
    X = ds_loader.X # Shape (N, n_features, N_timeperiod)
    d_y = 1

else:
    assert False, f"Unknown dataset: {config['dataset']}"

# Graph pre-processing
# Remove self loops
edge_index, edge_weight = ptg.utils.remove_self_loops(edge_index, edge_weight)

# Remove isolated nodes
edge_index, edge_weight, isolation_mask = ptg.utils.remove_isolated_nodes(edge_index,
        edge_weight, num_nodes=X.shape[0])
# Mask out removed nodes from X
X = X[isolation_mask]

# Time series pre-processing
N_T_original = 288 # 1 day of 5-min observations

num_nodes = X.shape[0]
X_splits = torch.split(X, N_T_original, dim=2)
if X_splits[-1].shape[2] < N_T_original:
    # Drop last
    X_splits = X_splits[:-1]

# Reshape to fit into framework
ds_time_series = torch.stack(X_splits,
        dim=0).transpose(1,3) #Shape (N_data,N_T_orig,d_y+d_f,N)
N_data = ds_time_series.shape[0]

# Subsample timeseries
t_ts = (torch.arange(N_T_original + 1)/N_T_original
        )[1:] # First observation at t=(1/N_T), last at t=1.0
t_original = t_ts.unsqueeze(0).repeat(N_data, 1) # Shape (N_data, N_T_orig)

N_T = int(config["subsample_time"]*288)
index_to_keep = torch.stack([torch.randperm(N_T_original)[:N_T]
    for _ in range(N_data)]) # (N_data, N_T)
index_to_keep, _ = torch.sort(index_to_keep, dim=1) # (N_data, N_T)

# Filter data and timesteps
ds_time_series = torch.stack([time_series[keep_is]
    for time_series, keep_is in zip(ds_time_series, index_to_keep)], dim=0)
t = torch.stack([timepoints[keep_is]
    for timepoints, keep_is in zip(t_original, index_to_keep)], dim=0)

y = ds_time_series[:,:,:d_y].transpose(2,3) # Shape (N_data, N_T, N, d_y)
if ds_time_series.shape[2] > d_y:
    # Remaining dimensions are features
    features = ds_time_series[:,:,d_y:].transpose(2,3) # Shape (N_data, N_T, N, d_f)
else:
    features = None

# Note: datasets can already have some missing values
# Smallest value represents missing data, as it was 0 speed before normalization
missing_val = torch.min(torch.min(torch.min(y, dim=0)[0], dim=0)[0],
        dim=0)[0] # Shape (d_y)
missing_mask = torch.any((y == missing_val), dim=3) # Shape (N_data, N_T, N)
data_obs_mask = torch.logical_not(missing_mask) # (N_data, N_T, N)

n_obs = int(config["obs_nodes"]*(N_T * num_nodes))
obs_mask = torch.zeros_like(data_obs_mask, dtype=float) # (N_data, N_T, N), floats (0,1)
for data_i, sample_mask in enumerate(data_obs_mask):
    obs_index = sample_mask.nonzero() # (N_nonzero, 2)

    shuffle_index = torch.randperm(obs_index.shape[0])
    obs_index_shuffled = obs_index[shuffle_index]

    kept_obs_index = obs_index_shuffled[:n_obs]
    obs_mask[data_i, kept_obs_index[:,0], kept_obs_index[:,1]] = 1.

obs_select = (lambda a: torch.stack([torch.index_select(row, 0, row_i)
    for row, row_i in zip(a, selected_index)], dim=0))

# Mask y and features
y = y*obs_mask.unsqueeze(-1) # (N_samples, N_T, N, d_y)
if not (features is None):
    features = features*obs_mask.unsqueeze(-1)

# Get deltas
delta_t = utils.node_t_deltas(t, obs_mask) # (N_samples, N_T, N)
delta_t = delta_t.transpose(1,2) # (N_samples, N, N_T)

obs_mask = obs_mask.transpose(1,2) # (N_samples, N, N_T)

# Compute dataset sizes
N_train = int(config["train_fraction"]*N_data)
N_test = int(config["test_fraction"]*N_data)
N_val = N_data - (N_train + N_test)

# Print some info
ds_name = f"{config['dataset']}_node_{config['obs_nodes']}"
actual_obs_fraction = torch.mean(obs_mask)

print(f"Dataset: {config['dataset']}")
print(f"Chosen fraction of nodes observed: {config['obs_nodes']}")
print(f"Final fraction of nodes observed: {actual_obs_fraction}")
print(f"Number of nodes: {num_nodes}")
print(f"Number of time steps: {N_T}")
print(f"N_train: {N_train}")
print(f"N_val: {N_val}")
print(f"N_test: {N_test}")
print(f"Saving as: {ds_name}")

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

for ar, ar_name in zip((y, t, delta_t, features, obs_mask),
        ("y", "t", "delta_t", "features", "mask")):
    if not (ar is None):
        ar_train, ar_val, ar_test = split_subsets(ar)

        data_save_dict["train"][ar_name] = ar_train
        data_save_dict["val"][ar_name] = ar_val
        data_save_dict["test"][ar_name] = ar_test

# Optionally plot
if config["plot"]:
    vis.plot_preprocessed(data_save_dict, n_plots=config["plot"],
            max_nodes_plot=config["max_nodes_plot"])

# Save data
utils.save_data(ds_name, config, data_save_dict)
print("Data saved")

