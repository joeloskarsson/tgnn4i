import torch
import torch.nn as nn

from models.decay_cell import DecayCell

# (Very loosely) Adapted from https://github.com/YuliaRubanova/latent_ode/
# which was adapted from https://github.com/zhiyongc/GRU-D

class GRUDecayCell(DecayCell):
    def __init__(self, input_size, config):
        super(GRUDecayCell, self).__init__(config)
        self.W = nn.Linear(input_size, self.n_states_internal*config["hidden_dim"],
                bias=True) # bias vector from here
        self.U = nn.Linear(config["hidden_dim"],
                self.n_states_internal*config["hidden_dim"], bias=False)

    def compute_inner_states(self, inputs, h_decayed, edge_index, edge_weight):
        Wx = self.W(inputs) # Shape (B, n_states_internal*d_h)
        Uh = self.U(h_decayed) # Shape (B, n_states_internal*d_h)

        return Wx, Uh

