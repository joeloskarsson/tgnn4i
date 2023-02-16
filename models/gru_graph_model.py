import torch.nn as nn
import torch

import utils
from models.gru_graph import GRUGraphCell
from models.gru_node_model import GRUNodeModel

class GRUGraphModel(GRUNodeModel):
    def __init__(self, config):
        self.pred_gnn = bool(config["pred_gnn"])

        super(GRUGraphModel, self).__init__(config)

        if self.pred_gnn:
            pred_gnn_input_dim = super(GRUGraphModel, self).compute_pred_input_dim(config)

            # instantiate a GNN to use for predictions, output dim here is hidden dim
            self.pred_gnn_model = utils.build_gnn_seq(config["pred_gnn"],
                    pred_gnn_input_dim, config["hidden_dim"], config["hidden_dim"],
                    config["gnn_type"])

    def create_cells(self, config):
        if config["gru_gnn"]:
            input_dim = self.compute_gru_input_dim(config)

            return nn.ModuleList([
                    GRUGraphCell(input_dim if layer_i==0 else config["hidden_dim"], config)
                for layer_i in range(config["gru_layers"])])
        else:
            # No GNN for GRU, use normal (decaying) GRU-cell
            return super(GRUGraphModel, self).create_cells(config)

    def compute_pred_input_dim(self, config):
        if self.pred_gnn:
            return config["hidden_dim"]
        else:
            return super(GRUGraphModel, self).compute_pred_input_dim(config)

    def compute_predictions(self, pred_input, edge_index, edge_weight):
        # pred_input: (BN, N_T, pred_input_dim)

        pred_input = pred_input.transpose(0,1)

        if self.pred_gnn:
            post_gnn = self.pred_gnn_model(pred_input, edge_index,
                        edge_weight) # (N_T, N_T, B*N, hidden_dim)

            fc_input = nn.functional.relu(post_gnn) # Activation in-between
        else:
            fc_input = pred_input

        return self.post_gru_layers(fc_input).transpose(0,1) # Shape (B*N, N_T, d_out)

