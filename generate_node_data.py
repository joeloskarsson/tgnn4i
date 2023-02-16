import torch
import torch_geometric as ptg
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

import utils
import visualization as vis
from models.gru_graph_model import GRUGraphModel
from tueplots import axes, bundles

parser = argparse.ArgumentParser(description='Generate dataset')

parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--plot", type=int, default=0,
        help="If plots should be made  during generation (and number of plots)")
parser.add_argument("--max_nodes_plot", type=int, default=5,
            help="Maximum nummber of nodes to plot predictions for")

# Graph construction
parser.add_argument("--n_nodes", type=int, default=10,
        help="Number of nodes in graph")
parser.add_argument("--graph_alg", type=str, default="delaunay",
        help="Algorithm to use for constructing graph")
parser.add_argument("--n_neighbors", type=int, default=5,
        help="Amount of neighbors to include in k-nn graph generation")

# Signal construction
parser.add_argument("--T", type=float, default=2.0,
        help="Time horizon")
parser.add_argument("--neighbor_weight", type=float, default=1.0,
        help="Weighting on the influence of neighbor signals")
parser.add_argument("--lag", type=float, default=0.1,
        help="Lag of neighbor influence")
parser.add_argument("--noise_std", type=float, default=0.05,
        help="Std.-dev. of added noise")
parser.add_argument("--batch_size", type=int, default=256,
        help="Batch size used for batched computations")

# Dataset stats
parser.add_argument("--n_t", type=int, default=20,
        help="Number of time points to evaluate at")
parser.add_argument("--obs_nodes", type=str, default="0.25",
        help="Percentage of nodes observed at each timestep (in [0,1] or 'single')")
parser.add_argument("--n_train", type=int, default=100,
        help="Number of data points (time series) in training set")
parser.add_argument("--n_val", type=int, default=50,
        help="Number of data points (time series) in validation set")
parser.add_argument("--n_test", type=int, default=50,
        help="Number of data points (time series) in test set")

config = vars(parser.parse_args())

# Set random seed
_ = torch.random.manual_seed(config["seed"])
np.random.seed(config["seed"])

# Generate graph
print("Generating graph ...")
node_pos = torch.rand(config["n_nodes"], 2)
point_data = ptg.data.Data(pos=node_pos)

if config["graph_alg"]== "delaunay":
    graph_transforms = ptg.transforms.Compose((
        ptg.transforms.Delaunay(),
        ptg.transforms.FaceToEdge(),
    ))
elif config["graph_alg"] == "knn":
    graph_transforms = ptg.transforms.Compose((
        ptg.transforms.KNNGraph(k=config["n_neighbors"], force_undirected=True),
    ))
else:
    assert False, "Unknown graph algorithm"
graph_pos = graph_transforms(point_data)

# Make DAG
undir_edges = graph_pos.edge_index
# Only keep edges going from lower order index to higher (let node index be ordering)
edges = undir_edges[:, undir_edges[0] < undir_edges[1]]
if config["plot"]:
    print(f"Edges: {edges}")
    dag = ptg.data.Data(pos=graph_pos.pos, edge_index=edges)
    vis.plot_graph(dag, show=True)

# Sample signals
n_samples = config["n_train"] + config["n_val"] + config["n_test"]

# Base signals
N_DISC = 1000
ts = torch.linspace(0., config["T"], N_DISC) # Discretize over 1000 time points

float_frac = config["T"]/config["lag"]
assert np.isclose(np.round(float_frac), float_frac),"T must be a multiple of lag"
# Expand T-horizon to account for lag begind time 0
T_expanded = config["T"] + config["lag"]*config["n_nodes"]
disc_delta = config["T"]/N_DISC
n_disc_expanded = round(T_expanded/disc_delta)

base_ts = torch.linspace(0, T_expanded, n_disc_expanded) # (N_disc_exp,)
base_ts = base_ts.view(1, n_disc_expanded, 1) # (1, N_disc_exp, 1)

start_pos = torch.randn(n_samples,1, config["n_nodes"]) # (N_samples, 1, N)

f = 20 + torch.rand(1,1,config["n_nodes"])*100 # U([20,100]) # (1, 1, N)
phase = 2*torch.pi*torch.rand(n_samples, 1, config["n_nodes"]) # (N_samples, 1, 1)

angles = f*base_ts + phase # Shape (N_samples, N_disc_exp, N)

rws = start_pos + torch.sin(angles) # Shape (N_samples, N_disc_exp, N)

index_offset = int(np.round(config["lag"] / disc_delta))
# Note: Indexes are topological ordering of DAG
full_signals = rws.clone() # Shape (N_samples, N_disc_exp, N)
for node_i in range(config["n_nodes"]):
    parents = edges[0,edges[1] == node_i]

    if parents.numel() > 0:
        # Safe to 0-pad here by construction
        neighbor_y = torch.cat((
            torch.zeros(n_samples, index_offset, parents.numel()),
            full_signals[:,:-index_offset,parents]
            ), dim=1)
        full_signals[:,:,node_i] = full_signals[:,:,node_i] +\
            config["neighbor_weight"]*torch.mean(neighbor_y, dim=2)

signals = full_signals[:,-N_DISC:]
base_signals = rws[:,-N_DISC:]

# Sample timepoints
sample_p = torch.ones(n_samples, N_DISC)
sample_i = torch.multinomial(sample_p, config["n_t"],
        replacement=False) #Shape (B, N_T)
sample_i, _ = torch.sort(sample_i, dim=1) # Shape (B, N_T)

signal_samples = torch.stack([signal[t_i]for signal, t_i in
    zip(signals, sample_i)], axis=0) # (B, N_T, N)
base_signal_samples = torch.stack([base_signal[t_i]for base_signal, t_i in
    zip(base_signals, sample_i)], axis=0) # (B, N_T, N)
t_samples = torch.stack([ts[t_i] for t_i in sample_i], axis=0) # Shape (B, N_T)

# Sample which nodes are observed at each time points (create masks)
if config["obs_nodes"] == "single":
    n_obs = 1
else:
    n_obs = int(float(config["obs_nodes"])*config["n_nodes"])

sample_p = torch.ones(n_samples*config["n_t"], config["n_nodes"])
# observed nodes at each timestep and sample
obs_nodes = torch.multinomial(sample_p, num_samples = n_obs,
        replacement=False) # (N_samples*N_T, n_obs)
obs_nodes = obs_nodes.reshape(n_samples, config["n_t"], n_obs) # (N_samples, N_T, n_obs)
# obs_mask is True if node observed
obs_mask = torch.zeros(n_samples, config["n_t"], config["n_nodes"])
for obs_sets, mask_slice in zip(obs_nodes, obs_mask):
    for obs_set, mask_vector in zip(obs_sets, mask_slice):
        mask_vector[obs_set] = 1.

# Create time deltas
delta_t = utils.node_t_deltas(t_samples, obs_mask)
delta_t = delta_t.transpose(1,2) # (N_samples, N, N_T)

# Add noise
y = signal_samples + torch.randn(signal_samples.shape)*config["noise_std"]
# Shape (N_samples, N_T, N)

# Mask unobserved nodes, add target dim. Unobserved values are 0
y = (y*obs_mask).unsqueeze(-1) # (N_samples, N_T, N, 1)

if config["plot"]:
    with plt.rc_context(bundles.aistats2023(column="full", nrows=0.7)):
        for plot_i in range(config["plot"]):
            vis_sample = y[plot_i,:,:,0].clone() # Shape (N_T, N)
            vis_t = t_samples[plot_i]

            vis_mask = obs_mask[plot_i] # Shape (N_T, N)
            vis_sample[vis_mask == 0] = torch.nan

            fig, ax = plt.subplots()

            # Last nodes are (generally) more interesting
            vis_signal = signals[plot_i,:,-config["max_nodes_plot"]:]
            ax.plot(ts, vis_signal, zorder=1)

            for n_i, y_n in enumerate(vis_sample.T[-config["max_nodes_plot"]:]):
                ax.scatter(vis_t, y_n, label=n_i, edgecolors="black", marker="o",
                        s=20, zorder=5, linewidths=0.8)

            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$\kappa^n(t)$")

            ax.set_xlim(0., 1.)

            fig.savefig(os.path.join("plotting", "periodic_example.pdf"))
            plt.show()

# Splits are train, val, test
mask = obs_mask.transpose(1,2) # (N_samples, N, N_T)
y_splits = np.split(y.numpy(), (config["n_train"],
    config["n_train"]+config["n_val"])) # Shape of each: (N_set, N_T, N, d_y)
t_splits = np.split(t_samples.numpy(), (config["n_train"],
    config["n_train"]+config["n_val"])) # Shape of each: (N_set, N_T)
delta_t_splits = np.split(delta_t.numpy(), (config["n_train"],
            config["n_train"]+config["n_val"])) # Shape of each: (N_set, N, N_T)
mask_splits = np.split(mask.numpy(), (config["n_train"],
            config["n_train"]+config["n_val"])) # Shape of each: (N_set, N, N_T)

# Also save base signals
base_signal_splits = np.split(
        base_signal_samples.unsqueeze(-1).numpy(), (config["n_train"],
            config["n_train"]+config["n_val"])) # Shape of each: (N_set, N_T, N, 1)

ds_name = f"periodic_{config['n_nodes']}_{config['obs_nodes']}_{config['seed']}"
save_dict = {}
for set_name, y_set, t_set, delta_t_set, mask_set, base_signal_set in zip(
        ("train", "val", "test"),
        y_splits,
        t_splits,
        delta_t_splits,
        mask_splits,
        base_signal_splits,
        ):
    save_dict[set_name] = {
        "y": y_set,
        "t": t_set,
        "delta_t": delta_t_set,
        "mask": mask_set,
        "base_signal": base_signal_set,
    }
save_dict["edge_index"] = edges

utils.save_data(ds_name, config, save_dict)

print("Data saved")

