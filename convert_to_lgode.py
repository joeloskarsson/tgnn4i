import torch
import torch_geometric as ptg
import numpy as np
import argparse
import os

import utils

SAVE_DIR = "lgode_dataset"
# NOTE: This all assumes d_y=1

parser = argparse.ArgumentParser(description='Convert data to format of LG-ODE')
parser.add_argument("--dataset", type=str, default="bay_node_0.25",
        help="Which dataset to use")

args = parser.parse_args()
config = vars(args)

print(f"Converting {config['dataset']} to LG-ODE format")
all_data = utils.load_data(config["dataset"])

# Edges / Adj.-mat.
edge_index = utils.to_tensor(all_data["edge_index"], dtype=int)

A = ptg.utils.to_dense_adj(edge_index) # (1, N, N)

# Values
for subset in ("train", "val", "test"):
    data = all_data[subset]

    mask = utils.to_tensor(data["mask"]) # (N_set, N, N_t)
    mask = mask.to(bool).numpy() # (N_set, N, N_t)
    N_set = mask.shape[0]

    y = utils.to_tensor(data["y"]) # (N_set, N_t, N, d_y=1)
    y = y.transpose(1,2).numpy() # (N_set, N, N_t, 1)

    t = utils.to_tensor(data["t"]).numpy() # (N_set, N_t)

    # LG-ODE does not use features, just zeros
    f = np.zeros_like(y) # (N_set, N, N_t, 1)

    # Iterate and mask out
    targets = []
    features = []
    times = []
    for y_sample, f_sample, t_sample, mask_sample in zip(y, f, t, mask):
        sample_targets = [y_node[mask_node] for y_node, mask_node in
                zip(y_sample, mask_sample)]
        sample_features = [f_node[mask_node] for f_node, mask_node in
                zip(f_sample, mask_sample)]
        sample_times = [t_sample[mask_node] for mask_node in mask_sample]

        targets.append(sample_targets)
        features.append(sample_features)
        times.append(sample_times)

    # Create stack of adjacency matrices
    edges = A.repeat_interleave(N_set, 0).numpy()

    # Save all
    os.makedirs(SAVE_DIR, exist_ok=True)
    dir_path = os.path.join(SAVE_DIR, config["dataset"])
    os.makedirs(dir_path, exist_ok=True)
    for save_name, np_data in (
            ("loc", np.array(targets, dtype=object)),
            ("vel", np.array(features, dtype=object)), # Use zeros as veolcities
            ("edges", edges),
            ("times", np.array(times, dtype=object)),
            ("joint_times", t),
    ):
        save_path = os.path.join(dir_path,
                f"{save_name}_{subset}_{config['dataset']}.npy")
        np.save(save_path, np_data)

print("Done!")

