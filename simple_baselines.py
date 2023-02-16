import torch
import argparse
import numpy as np

import utils
import pred_dists
import loss

parser = argparse.ArgumentParser(description='Evaluate prevdict previous baseline')

parser.add_argument("--dataset", type=str, default="bay_node_0.25",
        help="Which dataset to use")
parser.add_argument("--baseline", type=str, default="pred_prev",
        help="Which dataset to use")
parser.add_argument("--test", type=int, default=0,
        help="Evaluate test set (otherwise validation set)")
parser.add_argument("--init_points", type=int, default=5,
        help="Number of points to observe before prediction start")
parser.add_argument("--seed", type=int, default=42,
        help="Seed for random number generator")
parser.add_argument("--loss_weighting", type=str, default="exp,0.04",
        help="Function to weight loss with, given as: name,param1,...,paramK")
parser.add_argument("--max_pred", type=int, default=10,
        help="Maximum number of time indices forward to predict")
config = vars(parser.parse_args())

# Set all random seeds
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# Load data
device = torch.device("cpu")
data_dict = utils.load_data(config["dataset"])
edge_index = utils.to_tensor(data_dict["edge_index"], device=device, dtype=torch.long)

train_y = utils.to_tensor(data_dict["train"]["y"],
        device=device).transpose(1,2) # Shape (N_train, N, N_T, d_y)

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

# Compute predictions
if config["baseline"] == "pred_prev":
    # Fill observations forward until new observation
    ff_obs = utils.forward_fill(eval_y, eval_mask) # (B, N, N_T, d_param)
    # Previous observed values is same for all future predictions
    single_prediction = ff_obs.reshape(-1, N_T, 1) # (BN, N_T, d_param)
elif config["baseline"] == "node_mean":
    node_means = torch.mean(train_y, dim=(0,2,3)) # (N,)
    node_means_repeated = node_means.repeat(eval_t.shape[0]) # (BN,)
    single_prediction = node_means_repeated.unsqueeze(1).repeat_interleave(
            N_T, 1).unsqueeze(2) # (BN, N_T, 1)
elif config["baseline"] == "graph_mean":
    graph_mean = torch.mean(train_y)
    single_pred_template = torch.ones(
        eval_y.shape[0]*num_nodes, N_T, 1) # (BN, N_T, d_param=1)
    single_prediction = graph_mean*single_pred_template
else:
    assert False, f"Unknown baseline {config['baseline']}"

# Expand to all future predictions
prediction = single_prediction.unsqueeze(2).repeat_interleave(
        config["max_pred"], 2) # (BN, N_T, max_pred, d_param)
prediction = prediction.transpose(0,1) # (N_T, BN, max_pred, d_param)

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

print(f"Baseline: {config['baseline']}")
print(f"Dataset: {config['dataset']}")
print(f"Max pred: {config['max_pred']}")
print(f"{eval_set} wmse: {wmse}")
print(f"{eval_set} mse: {mse}")

