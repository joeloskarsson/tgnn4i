import torch

def step_future_loss(pred_dist_params, target, pred_delta_times, obs_mask,
        pred_dist, loss_weighter, config, metric="mse", return_errors=False):
    # pred_dist_params: (BN, max_pred, d_params)
    # target: (BN, max_pred)
    # pred_delta_times: (BN, max_pred)
    # obs_mask: (BN, max_pred)

    pred_dist_params = pred_dist_params.view(-1, config["num_nodes"], config["max_pred"],
            config["param_dim"]) # (B, N, max_pred, d_param)
    full_pred_dist = pred_dist(pred_dist_params)

    new_targets = target.view(-1, config["num_nodes"], config["max_pred"])
    if metric == "nll":
        full_loss = -full_pred_dist.log_prob(new_targets) # (B, N, max_pred)
    elif metric == "mse":
        mean_pred = full_pred_dist.mean
        full_loss = (mean_pred - new_targets)**2 # (B, N, max_pred)

    obs_mask_rs = obs_mask.view(-1, config["num_nodes"], config["max_pred"])
    masked_loss = full_loss*obs_mask_rs # (B, N ,max_pred)

    loss_weight = loss_weighter(pred_delta_times).view(-1, config["num_nodes"],
            config["max_pred"]) # (B, N, max_pred)
    # Fix issue if N_obs = 0
    N_obs = torch.clamp(torch.sum(obs_mask_rs, dim=(1,2)), min=1) # (B,)

    # Compile into final loss
    final_terms = (masked_loss*loss_weight) # (B, N, max_pred)
    sample_loss = torch.sum(final_terms, dim=(1, 2))/N_obs # (B,)

    # Mean over samples in batch
    loss = torch.mean(sample_loss)

    if return_errors:
        return loss, masked_loss
    else:
        return loss

def full_future_loss(pred_dist_params, target, pred_delta_times, obs_mask,
        pred_dist, loss_weighter, config, metric="mse", return_errors=False):
    # pred_dist_params: (N_T, BN, max_pred, d_y, d_params)
    # target: (BN, N_T, d_y)
    # pred_delta_times: (N_T, B, max_pred)
    # obs_mask: (N_T, BN)

    pred_dist_params = pred_dist_params.view(config["time_steps"], -1,
            config["num_nodes"], config["max_pred"], config["y_dim"],
            config["param_dim"]) # (N_T, B, N, max_pred, d_y, d_param)
    full_pred_dist = pred_dist(pred_dist_params)

    # Pad and reshape target and mask
    target_padding = torch.zeros_like(target)[
            :,:config["max_pred"]] # (BN, max_pred, d_y)
    target_padded = torch.cat((target, target_padding), dim=1) # (BN, N_T+max_pred, d_y)

    mask_rs = obs_mask.t() # (BN, N_T)
    mask_padding = torch.zeros_like(mask_rs)[:,:config["max_pred"]] # (BN, max_pred)
    mask_padded = torch.cat((mask_rs, mask_padding), dim=1) # (BN, N_T+max_pred)

    step_targets = [target_padded[:,(1+di):(1+di+config["time_steps"]),:]
            for di in range(config["max_pred"])]
    # Lists of length max_pred, Each is (BN, N_T, d_y)
    step_masks = [mask_padded[:,(1+di):(1+di+config["time_steps"])]
            for di in range(config["max_pred"])]
    # Lists of length max_pred, Each is (BN, N_T)

    new_targets = torch.stack(step_targets, dim=2).transpose(
            0,1) #(N_T, BN, max_pred, d_y)
    new_targets = new_targets.view(config["time_steps"], -1, config["num_nodes"],
            config["max_pred"], config["y_dim"]) #(N_T, B, N, max_pred, d_y)
    new_mask = torch.stack(step_masks, dim=2).transpose(0,1) #(N_T, BN, max_pred)
    new_mask = new_mask.view(config["time_steps"], -1, config["num_nodes"],
            config["max_pred"]) #(N_T, B, N, max_pred)

    if metric == "nll":
        full_loss = -full_pred_dist.log_prob(new_targets) # (N_T, B, N, max_pred, d_y)
    elif metric == "mse":
        mean_pred = full_pred_dist.mean
        full_loss = (mean_pred - new_targets)**2 # (N_T, B, N, max_pred, d_y)

    # Masks
    # When pred_delta_times=0 this means: no future obs to predict
    time_mask = (pred_delta_times > 0.).to(torch.float32) # (N_T, B, max_pred)
    time_mask = time_mask.view(config["time_steps"], -1, 1,
            config["max_pred"], 1) # (N_T, B, 1, max_pred, 1)

    obs_mask_rs = new_mask.unsqueeze(-1) # (N_T, B, N, max_pred, 1)
    masked_loss = full_loss*obs_mask_rs*time_mask # (N_T, B, N, max_pred, d_y)
    # [i,:,:,j,:] is prediction t_i -> t_(i+j)

    # Weighted average over all predictions of time point
    n_pred_indexer = (1+torch.arange(config["time_steps"]+config["max_pred"],
        device=target.device)).clamp(max=config["max_pred"])
    # [1, 2, ..., max_pred, max_pred, ..., max_pred]

    n_preds_list = [n_pred_indexer[t_i:(t_i+config["time_steps"])]
            for t_i in range(config["max_pred"])]
    # List of length max_pred, each entry (N_t,)
    n_preds = torch.stack(n_preds_list, dim=1).unsqueeze(1) # (N_t, 1, max_pred)

    loss_weight = loss_weighter(pred_delta_times) # (N_T, B, max_pred)
    full_weighting = loss_weight/n_preds # (N_T, B, max_pred)
    full_weighting = full_weighting.view(config["time_steps"], -1, 1,
            config["max_pred"], 1) # (N_T, B, 1, max_pred, 1)

    # Compile into final loss
    final_terms = (masked_loss*full_weighting)[config["init_points"]:]
    # (N_T-N_init, B, N, max_pred,d_y)
    N_obs = torch.sum(obs_mask.view(config["time_steps"], -1,
        config["num_nodes"])[config["init_points"]:], dim=(0,2)) # (B,)
    sample_loss = torch.sum(final_terms, dim=(0, 2, 3, 4))/(N_obs*config["y_dim"]) # (B,)

    # Mean over samples in batch
    loss = torch.mean(sample_loss)

    if return_errors:
        return loss, masked_loss
    else:
        return loss

