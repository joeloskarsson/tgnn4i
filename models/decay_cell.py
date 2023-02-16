import torch
import torch.nn as nn

import utils

# RNN Cell with decay (to be extended)
class DecayCell(nn.Module):
    def __init__(self, config):
        super(DecayCell, self).__init__()

        self.decay_type = config["decay_type"]
        if config["model"] == "grud_joint":
            # GRU model can be viewed as graph with one node
            self.num_nodes = 1
        else:
            self.num_nodes = config["num_nodes"]
        self.periodic = bool(config["periodic"])

        # Compute number of hidden states needed from W and U
        self.n_states_internal = 3

        if config["decay_type"] == "none":
            if config["model"] == "grud_joint": # Works on all nodes combined
               self.decay_target = torch.zeros(1, config["hidden_dim"],
                       device=config["device"])
            else:
                self.decay_target = torch.zeros(config["num_nodes"],config["hidden_dim"],
                    device=config["device"])
            self.decay_weight = torch.ones(config["hidden_dim"], device=config["device"])
        elif config["decay_type"] == "to_const":
            # Init non-negative
            self.decay_weight = nn.Parameter(torch.rand(config["hidden_dim"],
                device=config["device"]))

            if config["model"] == "grud_joint": # Works on all nodes combined
               self.decay_target = torch.zeros(1, config["hidden_dim"],
                       device=config["device"])
            elif config["node_params"]:
                # Node parameter for decay targets
                self.decay_target = utils.new_param(config["num_nodes"],
                        config["hidden_dim"])
            else:
                self.decay_target = torch.zeros(config["num_nodes"],
                        config["hidden_dim"], device=config["device"])
        elif config["decay_type"] == "dynamic":
            # r, z, \tilde{h} also for drift target
            # Also decay_weight output
            self.n_states_internal += 4
        else:
            assert False, f"Unknown decay type (decay_type): {config['decay_type']}"

        self.DECAY_CAP = 1000 # For numerics

    def compute_inner_states(self, inputs, h_decayed, edge_index):
        raise NotImplementedError()

    def decay_state(self, hidden_state, decay_weight, delta_ts):
        # hidden_state: (BN, d_h)
        # decay_weight: (d_h,) or (BN, d_h)
        # delta_ts: (B, N_t, 1)
        B, N_t, _ = delta_ts.shape
        d_h = hidden_state.shape[-1]

        if self.decay_type == "none":
            # Do not decay state
            return torch.repeat_interleave(hidden_state.unsqueeze(1), N_t,
                    dim=1) # (BN, N_t, d_h)

        if self.periodic:
            # apply time-dependent rotation matrix to pairs of hidden dimensions
            d_h2 = int(d_h/2)

            # In periodic mode half of decay_weight acts as the frequency of rotation
            exp_decay_weight, freq = torch.chunk(decay_weight, 2, dim=-1)
            # each is (d_h/2,) or (BN, d_h/2)

            z1, z2 = torch.chunk(hidden_state, 2, dim=-1) # each is (BN, d_h/2)
            delta_ts = delta_ts.repeat_interleave(self.num_nodes, dim=0) # (BN, N_T, 1)

            freq_rs = freq.view(-1, 1, d_h2) # (BN/1, 1, d_h/2)
            angle = freq_rs*delta_ts # (BN, N_t, d_h/2)
            cos_t = torch.cos(angle) # (BN, N_t, d_h/2)
            sin_t = torch.sin(angle) # (BN, N_t, d_h/2)

            exp_decay_weight_rs = exp_decay_weight.view(-1, 1, d_h2) # (1/BN, 1, d_h/2)
            decay_factor = torch.exp(-1*torch.clamp(delta_ts*exp_decay_weight_rs, min=0.,
                max=self.DECAY_CAP)) # Shape (BN, N_t, d_h/2)

            z1_rs, z2_rs = z1.unsqueeze(1), z2.unsqueeze(1)
            new_z1 = (z1_rs*cos_t - z2_rs*sin_t)*decay_factor # (B, N_t, d_h/2)
            new_z2 = (z1_rs*sin_t + z2_rs*cos_t)*decay_factor # (B, N_t, d_h/2)

            new_dynamic_state = torch.cat((new_z1, new_z2), dim=-1) # (B, N_t, d_h)
        else:
            decay_weight_rs = decay_weight.view(-1,1,d_h) # (BN/1, 1, d_h)

            delta_ts = delta_ts.repeat_interleave(self.num_nodes, dim=0) # (BN, N_T, 1)
            decay_factor = torch.exp(-1*torch.clamp(delta_ts*decay_weight_rs, min=0.,
                max=self.DECAY_CAP)) # Shape (BN, N_t, d_h)

            # hidden-state --> 0 (decaying as decay factor 1 --> 0)
            state_rs = hidden_state.view(-1, self.num_nodes,
                    1, d_h) # (B, N, 1, d_h)
            B = state_rs.shape[0]
            new_dynamic_state = state_rs*decay_factor.view(B, -1,
                    N_t, d_h) # (B, N/1, N_t, d_h)
            new_dynamic_state = new_dynamic_state.view(-1, N_t, d_h) # (BN, N_t, d_h)

        return new_dynamic_state # Shape (BN, N_t, d_h)

    def forward(self, inputs, h_ode, decay_target, decay_weight, delta_ts,
            edge_index, edge_weight):
        # inputs: (B, d_in)
        # h_ode: (B, d_h)
        # decay_target: (B, d_h)
        # decay_weight: (d_h,) or (B, d_h)
        # delta_ts: (B, 1)
        # edge_index: (2, N_edges)
        # edge_weight: (N_edges,)

        h_decayed = self.decay_state(h_ode, decay_weight,
                delta_ts.unsqueeze(1))[:,0]
        Wx, Uh = self.compute_inner_states(inputs, h_decayed+decay_target, edge_index,
                edge_weight)

        W_chunks = Wx.chunk(self.n_states_internal,1)
        U_chunks = Uh.chunk(self.n_states_internal,1)

        Wx_r, Wx_z, Wx_h = W_chunks[:3]
        Uh_r, Uh_z, Uh_h = U_chunks[:3]

        r = torch.sigmoid(Wx_r + Uh_r)
        z = torch.sigmoid(Wx_z + Uh_z)
        h_tilde = torch.tanh(Wx_r + Uh_h*r) # Shape (B, d_h)

        new_h = h_decayed + z*(h_tilde - h_decayed)

        # Compute decay target for time interval
        if self.decay_type == "dynamic":
            # Parameters for decay target parametrization
            Wx_rd, Wx_zd, Wx_hd = W_chunks[3:6]
            Uh_rd, Uh_zd, Uh_hd = U_chunks[3:6]

            rd = torch.sigmoid(Wx_rd + Uh_rd)
            zd = torch.sigmoid(Wx_zd + Uh_zd)
            hd_tilde = torch.tanh(Wx_rd + Uh_hd*rd) # Shape (B, d_h)

            new_decay_target = decay_target + zd*(hd_tilde - decay_target)

            # Decay weight
            Wx_decay_weight = W_chunks[6]
            Uh_decay_weight = U_chunks[6]
            decay_weight = nn.functional.softplus(Wx_decay_weight + Uh_decay_weight)
        else:
            # Compute batch size here (in terms of graphs)
            num_graphs = int(inputs.shape[0] / self.decay_target.shape[0])
            new_decay_target = self.decay_target.repeat(num_graphs, 1)

            decay_weight = self.decay_weight

        new_h_ode = new_h - new_decay_target
        return h_decayed, new_h_ode, new_decay_target, decay_weight

