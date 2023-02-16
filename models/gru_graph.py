import torch
import torch.nn as nn
import torch_geometric as ptg

import constants
import utils
from models.decay_cell import DecayCell

class GRUGraphCell(DecayCell):
    def __init__(self, input_size, config):
        super(GRUGraphCell, self).__init__(config)

        hidden_dim = config["hidden_dim"]
        out_dim = self.n_states_internal*hidden_dim
        self.input_gnn = utils.build_gnn_seq(config["gru_gnn"], input_size, hidden_dim,
                out_dim, config["gnn_type"])
        self.state_gnn = utils.build_gnn_seq(config["gru_gnn"], hidden_dim, hidden_dim,
                out_dim, config["gnn_type"])

    def compute_inner_states(self, inputs, h_decayed, edge_index,
            edge_weight):
        Wx = self.input_gnn(inputs, edge_index,
                edge_weight=edge_weight) # Shape (B, n_states_internal*d_h)
        Uh = self.state_gnn(h_decayed, edge_index,
                edge_weight=edge_weight) # Shape (B, n_states_internal*d_h)

        return Wx, Uh

