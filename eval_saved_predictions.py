import torch
import argparse
import numpy as np
from tueplots import axes, bundles
import matplotlib.pyplot as plt

import utils
import pred_dists
import loss
import visualization

parser = argparse.ArgumentParser(description='Evaluate predictions from saved file')

parser.add_argument("--dataset", type=str, default="bay_node_0.25",
        help="Which dataset to use")
parser.add_argument("--test", type=int, default=1,
        help="Evaluate test set (otherwise validation set)")
parser.add_argument("--init_points", type=int, default=5,
        help="Number of points to observe before prediction start")
parser.add_argument("--seed", type=int, default=42,
        help="Seed for random number generator")
parser.add_argument("--loss_weighting", type=str, default="exp,0.04",
        help="Function to weight loss with, given as: name,param1,...,paramK")
parser.add_argument("--max_pred", type=int, default=10,
        help="Maximum number of time indices forward to predict")
parser.add_argument("--plot", type=int, default=0,
        help="Show plot")
parser.add_argument("--load", type=str,
    help="File to load predictions from")
config = vars(parser.parse_args())

# Set all random seeds
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# Load data
device = torch.device("cpu")
data_dict = utils.load_data(config["dataset"])
edge_index = utils.to_tensor(data_dict["edge_index"], device=device, dtype=torch.long)

eval_set = "test" if config["test"] else "val"
eval_y = utils.to_tensor(data_dict[eval_set]["y"],
        device=device).transpose(1,2) # Shape (N_eval, N, N_T, d_y)
eval_t = utils.to_tensor(data_dict[eval_set]["t"], device=device) # (N_eval, N_T)
eval_delta_t = utils.to_tensor(data_dict[eval_set]["delta_t"],
        device=device) # (N_eval, N, N_T)
eval_mask = utils.to_tensor(data_dict[eval_set]["mask"],
        device=device) # (N_eval, N, N_T)

num_nodes = eval_y.shape[1]
N_T = eval_y.shape[2]

# Things neccesary to compute loss
pred_dist, _ = pred_dists.DISTS["gauss_fixed"]
loss_weight_func = utils.parse_loss_weight(config["loss_weighting"])
const_weight_func = utils.parse_loss_weight("const")

# Load predictions
loaded_pred = torch.load(config["load"], map_location=device) # (N_eval, N_T, N, max_pred)
loaded_pred = loaded_pred.transpose(0,1) # (N_T, N_eval, N, max_pred)

# Reshape to correct shape
prediction = loaded_pred.reshape(N_T, -1, config["max_pred"], 1, 1)
# (N_T, BN, max_pred, d_y=1, d_param=1)

# Get delta times for predictions
all_dts = eval_t.unsqueeze(1) - eval_t.unsqueeze(2) # (B, N_T, N_T)
# Index [:, i, j] is (t_j - t_i), time from t_i to t_j
off_diags = [torch.diagonal(all_dts, offset=offset, dim1=1, dim2=2).t()
        for offset in range(config["max_pred"]+1)]
# List of length max_preds, each entry is tensor: (diag_length, B)
padded_off_diags = torch.nn.utils.rnn.pad_sequence(off_diags,
        batch_first=False) # (N_T, max_pred+1, B)

pred_delta_times = padded_off_diags[:,1:].transpose(1,2) # (N_T, B, max_pred)
# Index [i, :, j] is (t_(i+j) - t_i), time from t_i to t_(i+j)

# Compute other tensors for loss
target = eval_y.reshape(-1, N_T, 1) # (BN, N_T, 1)
obs_mask = eval_mask.reshape(-1, N_T).transpose(0,1) # (N_T, BN)

config.update({
    "num_nodes": num_nodes,
    "y_dim": 1,
    "param_dim": 1,
    "time_steps": N_T}
) # Extra to match call from main
wmse = loss.full_future_loss(prediction, target, pred_delta_times, obs_mask,
        pred_dist, loss_weight_func, config, metric="mse")
mse = loss.full_future_loss(prediction, target, pred_delta_times, obs_mask,
        pred_dist, const_weight_func, config, metric="mse")

print(f"Saved predictions from: {config['load']}")
print(f"Dataset: {config['dataset']}")
print(f"Max pred: {config['max_pred']}")
print(f"{eval_set} wmse: {wmse}")
print(f"{eval_set} mse: {mse}")

if config["plot"]:
    config.update({
        "device": "cpu",
        "max_nodes_plot": 3,
        "pred_dist": "gauss_fixed",
    })
    figs = visualization.plot_step_prediction(
            None,
            None,
            config["plot"],
            pred_dist,
            config,
            batch_prediction=prediction,
            data_batch={
                "y": target,
                "t": eval_t,
                "mask": obs_mask,
            },
        )

    plt.show()
