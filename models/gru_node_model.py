import torch

from models.gru_model import GRUModel
import utils

class GRUNodeModel(GRUModel):
    # Handles each node independently with GRU-units
    def __init__(self, config):
        super(GRUNodeModel, self).__init__(config)

        self.num_nodes = config["num_nodes"]
        self.y_shape = (config["time_steps"], -1, config["y_dim"])
        self.f_shape = (config["time_steps"], -1, config["feature_dim"])
        self.state_updates = config["state_updates"]
        assert config["state_updates"] in ("all", "obs", "hop"), (
                f"Unknown state update: {config['state_updates']}")

        # If node-specific initial states should be used
        self.node_init_states = (config["node_params"] and config["learn_init_state"])
        if self.node_init_states:
            # Override initial GRU-states
            self.init_state_param = utils.new_param(config["gru_layers"],
                    config["num_nodes"], config["hidden_dim"])

    def get_init_states(self, num_graphs):
        if self.node_init_states:
            return self.init_state_param.repeat(
                1, num_graphs, 1) # Output shape (n_gru_layers, B*N, d_h)
        else:
            return self.init_state_param.unsqueeze(1).repeat(
                1, self.num_nodes*num_graphs, 1) # Output shape (n_gru_layers, B*N, d_h)

    def compute_gru_input_dim(self, config):
        # Compute input dimension at each timestep
        return config["y_dim"] + config["feature_dim"] +\
            int(self.time_input) + int(self.mask_input) # Add one if delta_t/mask input

    def compute_pred_input_dim(self, config):
        return config["hidden_dim"] + config["feature_dim"] +\
                int(self.time_input) # Add one if delta_t input

    def compute_output_dim(self, config):
        # Compute output dimension at each timestep
        return config["param_dim"]*config["y_dim"]

    def compute_predictions(self, pred_input, edge_index, edge_weight):
        # pred_input: (N_T, B*N, N_T, pred_input_dim)
        return self.post_gru_layers(pred_input) # Shape (N_T, B*N, N_T, d_out)

    def forward(self, batch):
        # Batch is ptg-Batch: Data(
        #  y: (BN, N_T, 1)
        #  t: (B, N_T)
        #  delta_t: (BN, N_T)
        #  mask: (BN, N_T)
        # )

        edge_weight = batch.edge_attr[:,0] # Shape (N_edges,)

        input_y_full = batch.y.transpose(0,1) # Shape (N_T, B*N, d_y)
        input_y_reshaped = input_y_full.reshape(self.y_shape) # (N_T, B*N, d_y)

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
        dt_node_obs = batch.update_delta_t.transpose(0,1).unsqueeze(2) # (N_T, BN, 1)

        obs_mask = batch.mask.transpose(0,1).unsqueeze(-1) # Shape (N_T, B*N, 1)

        if self.state_updates == "hop":
            update_mask = batch.hop_mask.transpose(0,1).unsqueeze(-1) # (N_T, B*N, 1)
        else:
            # "obs" (or "all", but then unused)
            update_mask = obs_mask

        # List with all tensors for input
        gru_input_tensors = [input_y_reshaped,] # input to gru update

        if self.has_features:
            input_f_full = batch.features.transpose(0,1) # Shape (N_T, B*N, d_f)
            input_f_reshaped = input_f_full.reshape(self.f_shape) # (N_T, BN, d_f)
            gru_input_tensors.append(input_f_reshaped)
            fc_feature_input = input_f_reshaped.transpose(0,1) # (BN, N_T, d_f)

            # Pad feature input for last time steps
            feature_padding = torch.zeros_like(fc_feature_input)[:,:self.max_pred,:]
            fc_feature_input = torch.cat((fc_feature_input, feature_padding), dim=1)
            # (BN, N_T+max_pred, d_f)

        if self.time_input:
            gru_input_tensors.append(delta_time)

        if self.mask_input:
            # Concatenated mask to input (does not always make sense)
            gru_input_tensors.append(obs_mask)
            # Mask should not be in fc_input, we don't
            # know what will be observed when predicting

        init_states = self.get_init_states(batch.num_graphs)

        gru_input = torch.cat(gru_input_tensors, dim=-1)
        for layer_i, (gru_cell, init_state) in enumerate(
                zip(self.gru_cells, init_states)):
            h_ode = torch.zeros_like(init_state)
            decay_target = init_state # Init decaying from and to initial state
            decay_weight = self.init_decay_weight # dummmy (does not matter)

            step_preds = [] # predictions from each step
            hidden_states = [] # New states after observation
            for t_i, (input_slice,\
                delta_time_slice,\
                update_mask_slice,\
                pred_delta_time_slice,\
                dt_node_obs_slice,)\
            in enumerate(zip(
                gru_input,\
                all_delta_ts,\
                update_mask,\
                pred_delta_times,\
                dt_node_obs
            )):
                # input_slice: (BN, d_gru_input)
                # delta_time_slice: (B,1)
                # update_mask_slice: (BN,1)
                # pred_delta_time_slice: (B, max_pred)
                # dt_node_obs_slice: (BN, 1)

                # STATE UPDATE
                decayed_states, new_h_ode, new_decay_target, new_decay_weight =\
                    gru_cell(input_slice, h_ode, decay_target, decay_weight,
                        delta_time_slice, batch.edge_index, edge_weight)

                if self.state_updates == "all":
                    # Update for all nodes
                    h_ode = new_h_ode
                    decay_target = new_decay_target
                    decay_weight = new_decay_weight
                else:
                    # GRU update for observed nodes, decay others
                    h_ode = update_mask_slice*new_h_ode +\
                        (1. - update_mask_slice)*decayed_states
                    decay_target = update_mask_slice*new_decay_target +\
                        (1. - update_mask_slice)*decay_target
                    decay_weight = update_mask_slice*new_decay_weight +\
                        (1. - update_mask_slice)*decay_weight

                # Hidden state is sum of ODE-controlled state and decay target
                hidden_state = h_ode + decay_target
                hidden_states.append(hidden_state)

                # PREDICTION
                if layer_i == (len(self.gru_cells)-1):
                    # Decay to all future time points for prediction
                    decayed_pred_h_ode = gru_cell.decay_state(h_ode,
                            decay_weight, pred_delta_time_slice.unsqueeze(-1))
                    decayed_pred_states = decayed_pred_h_ode + decay_target.unsqueeze(1)
                    # decayed_pred_states is (BN, max_pred, d_h)

                    # Perform prediction
                    pred_input_list = [decayed_pred_states]
                    if self.time_input:
                        # Note: Time since last node obs for each prediction is
                        # sum of dt_node_obs and pred_delta_tim
                        # dt_node_obs_slice: (BN, 1)
                        # pred_delta_time_slice: (B, max_pred)

                        # 0 time since update for nodes updated at this time
                        node_time_since_up = dt_node_obs_slice*(1. - update_mask_slice)

                        time_input = node_time_since_up.view(-1, self.num_nodes, 1) +\
                            pred_delta_time_slice.unsqueeze(1) # (B, N, max_pred)
                        BN = dt_node_obs_slice.shape[0]
                        time_input = time_input.view(BN, -1, 1) # (BN, max_pred, 1)
                        pred_input_list.append(time_input)

                    if self.has_features:
                        features_for_time = fc_feature_input[
                                :,(t_i+1):(t_i+1+self.max_pred)] # (BN, max_pred, d_f)
                        pred_input_list.append(features_for_time)

                    pred_input = torch.cat(pred_input_list,
                            dim=-1) # (BN, max_pred, d_h+d_aux)

                    step_prediction = self.compute_predictions(pred_input,
                            batch.edge_index, edge_weight) # (BN, max_pred, d_out)
                    step_preds.append(step_prediction)

            gru_input = hidden_states

        predictions = torch.cat(step_preds, dim=0) # (N_T, BN, max_pred, d_out)
        predictions_reshaped = predictions.view(
                *self.pred_shape) # (N_T, BN, max_pred, d_y, d_param)
        return predictions_reshaped, pred_delta_times

