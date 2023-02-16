import torch
import torch_geometric as ptg

# Torch constants
OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}

# Second value is if they use edge_weight
GNN_LAYERS = {
    "gcn": (ptg.nn.GCNConv, True),
    "gat": (ptg.nn.GATConv, False),
    "graphconv": ((lambda *args: ptg.nn.GraphConv(*args, aggr="mean")), True),
}

# WANDB
WANDB_PROJECT = "irregular-tgnns"

# Paths
DS_DIR = "dataset"
DATA_FILE_NAME = "data.pickle"
CONFIG_FILE_NAME = "config.json"
PARAM_FILE_NAME = "parameters.pt"
RAW_DATA_DIR = "raw_data"

