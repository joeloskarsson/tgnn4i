import torch
import torch.nn as nn

from models.gru_decay import GRUDecayCell
import utils

class GRUModel(nn.Module):
    # Handles all nodes together in a single GRU-unit
    def __init__(self, config):
        super(GRUModel, self).__init__()

        self.time_input = bool(config["time_input"])
        self.mask_input = bool(config["mask_input"])
        self.has_features = config["has_features"]
        self.max_pred = config["max_pred"]

        output_dim = self.compute_output_dim(config)

        self.gru_cells = self.create_cells(config)

        if config["learn_init_state"]:
            self.init_state_param = utils.new_param(config["gru_layers"],
                    config["hidden_dim"])
        else:
            self.init_state_param = torch.zeros(config["gru_layers"],
                    config["hidden_dim"], device=config["device"])

        first_post_dim = self.compute_pred_input_dim(config)
        if config["n_fc"] == 1:
            fc_layers = [nn.Linear(first_post_dim, output_dim)]
        else:
            fc_layers = []
            for layer_i in range(config["n_fc"]-1):
                fc_layers.append(nn.Linear(first_post_dim if (layer_i == 0)
                    else config["hidden_dim"], config["hidden_dim"]))
                fc_layers.append(nn.ReLU())

            fc_layers.append(nn.Linear(config["hidden_dim"], output_dim))

        self.post_gru_layers = nn.Sequential(*fc_layers)

        self.y_shape = (config["time_steps"], -1, config["num_nodes"]*config["y_dim"])
        self.delta_t_shape = (config["time_steps"], -1, config["num_nodes"])
        # Return shape
        self.pred_shape = (config["time_steps"], -1, config["max_pred"],
                config["y_dim"],config["param_dim"])
        self.f_shape = (config["time_steps"], -1,
                config["num_nodes"]*config["feature_dim"])

        self.init_decay_weight = torch.zeros(config["hidden_dim"],
                device=config["device"])

    def create_cells(self, config):
        input_dim = self.compute_gru_input_dim(config)

        return nn.ModuleList([
                GRUDecayCell(input_dim if layer_i==0 else config["hidden_dim"], config)
            for layer_i in range(config["gru_layers"])])

    def compute_gru_input_dim(self, config):
        # Compute input dimension at each timestep
        return config["num_nodes"]*(config["y_dim"] + config["feature_dim"] +
                int(self.mask_input)+ int(self.time_input))  # Add N if time/mask input

    def compute_pred_input_dim(self, config):
        return config["hidden_dim"] + config["num_nodes"]*config["feature_dim"] +\
                int(self.time_input) # Add 1 if delta_t input

    def compute_output_dim(self, config):
        # Compute output dimension at each timestep
        return config["num_nodes"]*config["param_dim"]*config["y_dim"]

    def get_init_states(self, num_graphs):
        return self.init_state_param.unsqueeze(1).repeat(
                1, num_graphs, 1) # Output shape (n_gru_layers, B, d_h)

    def forward(self, batch):
        # Batch is ptg-Batch: Data(
        #  y: (BN, N_T, 1)
        #  t: (B, N_T)
        #  delta_t: (BN, N_T)
        #  mask: (BN, N_T)
        # )

        edge_weight = batch.edge_attr[:,0] # Shape (N_edges,)

        input_y_full = batch.y.transpose(0,1) # Shape (N_T, B*N, d_y)
        input_y_reshaped = input_y_full.reshape(self.y_shape) # (N_T, B, N*d_y)

        delta_time = batch.delta_t.transpose(0,1).unsqueeze(-1) # Shape (N_T, B*N, 1)

        all_dts = batch.t.unsqueeze(1) - batch.t.unsqueeze(2) # (B, N_T, N_T)
        # Index [:, i, j] is (t_j - t_i), time from t_i to t_j
        off_diags = [torch.diagonal(all_dts, offset=offset, dim1=1, dim2=2).t()
                for offset in range(self.max_pred+1)]
        # List of length max_preds, each entry is tensor: (diag_length, B)
        padded_off_diags = torch.nn.utils.rnn.pad_sequence(off_diags,
                batch_first=False) # (N_T, max_pred+1, B)

        pred_delta_times = padded_off_diags[:,1:].transpose(1,2) # (N_T, B, max_pred)
        # Index [i, :, j] is (t_(i+j) - t_i), time from t_i to t_(i+j)

        all_delta_ts = utils.t_to_delta_t(batch.t).transpose(
                0,1).unsqueeze(2) # (N_T, B, 1)

        # Only for mask input here
        obs_mask = batch.mask.transpose(0,1).view(
                *self.delta_t_shape) # Shape (N_T, B, N)

        # List with all tensors for input
        gru_input_tensors = [input_y_reshaped,] # input to gru update

        if self.has_features:
            input_f_full = batch.features.transpose(0,1) # Shape (N_T, B*N, d_f)
            input_f_reshaped = input_f_full.reshape(self.f_shape) # (N_T, B, N*d_f)
            gru_input_tensors.append(input_f_reshaped)
            fc_feature_input = input_f_reshaped.transpose(0,1) # (B, N_T, N*d_f)

            # Pad feature input for last time steps
            feature_padding = torch.zeros_like(fc_feature_input)[:,:self.max_pred,:]
            fc_feature_input = torch.cat((fc_feature_input, feature_padding), dim=1)
            # (B, N_T+max_pred, N*d_f)

        if self.time_input:
            delta_time_inputs = batch.delta_t.transpose(0,1).view(
                self.delta_t_shape) # (N_T, B, N)

            # Concatenated delta_t to input
            gru_input_tensors.append(delta_time_inputs)

        if self.mask_input:
            # Concatenated mask to input (does not always make sense)
            gru_input_tensors.append(obs_mask)
            # Mask should not be in fc_input, we don't
            # know what will be observed when predicting

        init_states = self.get_init_states(batch.num_graphs)

        gru_input = torch.cat(gru_input_tensors, dim=-1) # (N_T, B, d_gru_input)
        for layer_i, (gru_cell, init_state) in enumerate(
                zip(self.gru_cells, init_states)):
            h_ode = torch.zeros_like(init_state)
            decay_target = init_state # Init decaying from and to initial state
            decay_weight = self.init_decay_weight # dummmy (does not matter)

            step_preds = [] # predictions from each step
            hidden_states = [] # New states after observation
            for t_i, (input_slice,\
                delta_time_slice,\
                pred_delta_time_slice)\
            in enumerate(zip(
                gru_input,\
                all_delta_ts,\
                pred_delta_times\
            )):
                # input_slice: (B, d_gru_input)
                # delta_time_slice: (B,1)
                # pred_delta_time_slice: (B, max_pred)

                # STATE UPDATE
                decayed_states, new_h_ode, new_decay_target, new_decay_weight =\
                    gru_cell(input_slice, h_ode, decay_target, decay_weight,
                        delta_time_slice, batch.edge_index, edge_weight)

                # Update for all nodes
                h_ode = new_h_ode
                decay_target = new_decay_target
                decay_weight = new_decay_weight

                # Hidden state is sum of ODE-controlled state and decay target
                hidden_state = h_ode + decay_target
                hidden_states.append(hidden_state)

                # PREDICTION
                if layer_i == (len(self.gru_cells)-1):
                    # Decay to all future time points for prediction
                    decayed_pred_h_ode = gru_cell.decay_state(h_ode,
                            decay_weight, pred_delta_time_slice.unsqueeze(-1))
                    decayed_pred_states = decayed_pred_h_ode + decay_target.unsqueeze(1)
                    # decayed_pred_states is (B, max_pred, d_h)

                    # Perform prediction
                    pred_input_list = [decayed_pred_states]
                    if self.time_input:
                        # Time from now until prediction
                        time_input = pred_delta_time_slice.unsqueeze(
                                -1) # (B, max_pred, 1)
                        pred_input_list.append(time_input)

                    if self.has_features:
                        features_for_time = fc_feature_input[
                                :,(t_i+1):(t_i+1+self.max_pred)] # (B, max_pred, N*d_f)
                        pred_input_list.append(features_for_time)

                    pred_input = torch.cat(pred_input_list,
                            dim=-1) # (B, max_pred, d_h+d_aux)

                    step_prediction = self.post_gru_layers(
                            pred_input) # (B, max_pred, N*d_out)
                    step_preds.append(step_prediction)

            gru_input = hidden_states

        predictions = torch.cat(step_preds, dim=0) # (N_T, BN, max_pred, N*d_out)
        predictions_reshaped = predictions.view(
                *self.pred_shape) # (N_T, BN, max_pred, d_y, d_param)
        return predictions_reshaped, pred_delta_times

