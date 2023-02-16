import torch
import numpy as np

import utils
import loss

def model_forward(model, batch, cond_length, config):
    pred = model.forward(batch, cond_length) # (BN, max_pred, 1)

    # Delta times to make predictions
    times = batch.t.transpose(0,1) # (N_t, B)
    pred_times = times[(cond_length-1):cond_length+config["max_pred"]] # (max_pred+1, B)
    pred_dts = pred_times[1:] - pred_times[:1] # (max_pred, B)

    # Pad with zeros in cases not enough dts
    actual_pred_t = pred_dts.shape[0]
    dt_padding = torch.zeros(config["max_pred"]-actual_pred_t, batch.num_graphs,
        device=pred_dts.device)

    pred_dts = torch.cat((pred_dts, dt_padding), dim=0)
    pred_dts = pred_dts.transpose(0,1).unsqueeze(1).repeat(
            1,config["num_nodes"],1).unsqueeze(-1) # (B, N, max_pred, 1)
    pred_delta_times = pred_dts.view(-1, config["max_pred"], 1) # (BN, max_pred)
    return pred, pred_delta_times

def train_epoch(model, data_loader, opt, pred_dist, config, loss_weighter):
    model.train(True)

    batch_losses = []

    for batch in data_loader:
        batch = batch.to(config["device"]) # Move all graphs to GPU
        opt.zero_grad()

        #  cond_length = int(config["time_steps"] / 2)
        cond_length = torch.randint(config["init_points"],
            config["time_steps"]-config["max_pred"], ()) # Random length for full batch

        pred, pred_delta_times = model_forward(model, batch, cond_length,
                config) # (B*N, max_pred, 1) and (BN, max_pred)
        target = batch.y[:,
                cond_length:cond_length+config["max_pred"], :] # (B*N, max_pred)
        obs_mask = batch.mask[:,
                cond_length:cond_length+config["max_pred"]] # (B*N, max_pred)

        batch_loss = loss.step_future_loss(pred, target, pred_delta_times,
                obs_mask, pred_dist, loss_weighter, config)

        batch_loss.backward()
        opt.step()

        batch_losses.append(batch_loss.detach()*batch.num_graphs)

    # Here mean over samples, to not weight samples in small batches higher
    epoch_loss = torch.sum(torch.stack(batch_losses))/len(data_loader.dataset)

    return epoch_loss.item()

@torch.no_grad()
def val_epoch(model, data_loader, pred_dist, loss_weighter, config):
    model.train(False)

    const_weighter = utils.parse_loss_weight("const")
    batch_metrics = {
        "wmse": [],
        "mse": [],
    }
    for batch in data_loader:
        batch = batch.to(config["device"]) # Move all graphs to GPU

        # Fixed length
        cond_length = int(config["time_steps"] / 2)
        pred, pred_delta_times = model_forward(model, batch, cond_length,
                config) # (B*N, max_pred, 1) and (BN, max_pred)
        target = batch.y[:,
                cond_length:cond_length+config["max_pred"], :] # (B*N, max_pred)
        obs_mask = batch.mask[:,
                cond_length:cond_length+config["max_pred"]] # (B*N, max_pred)

        batch_wmse = loss.step_future_loss(pred, target, pred_delta_times,
                obs_mask, pred_dist, loss_weighter, config, metric="mse")
        batch_mse = loss.step_future_loss(pred, target, pred_delta_times,
                obs_mask, pred_dist, const_weighter, config, metric="mse")

        for val, name in zip((batch_wmse, batch_mse), ("wmse", "mse")):
            batch_metrics[name].append(val.detach()*batch.num_graphs)

    epoch_metrics = {name:
            (torch.sum(torch.stack(val_list))/len(data_loader.dataset)).item()
            for name, val_list in batch_metrics.items()}

    return epoch_metrics

# Test epoch that computes predictions at each time
@torch.no_grad()
def test_epoch(model, data_loader, pred_dist, loss_weighter, config):
    model.train(False)

    const_weighter = utils.parse_loss_weight("const")
    batch_metrics = {
        "wmse": [],
        "mse": [],
    }
    for batch in data_loader:
        batch = batch.to(config["device"]) # Move all graphs to GPU

        # Fixed length
        pred_param_list = [torch.zeros(batch.num_graphs*config["num_nodes"],
            config["max_pred"], 1, device=config["device"])]*config["init_points"]
        pred_dt_list = [torch.zeros(batch.num_graphs, config["max_pred"],
            device=config["device"])]*config["init_points"]
        for cond_length in range(config["init_points"], config["time_steps"]):
            pred, pred_dts = model_forward(model, batch, cond_length,
                    config) # (B*N, max_pred, 1) and (BN, max_pred)

            batch_pred_times = pred_dts.view(-1, config["num_nodes"],
                    config["max_pred"])[:,0,:] # (B, max_pred)
            pred_param_list.append(pred)
            pred_dt_list.append(batch_pred_times)

        full_pred_params = torch.stack(pred_param_list, dim=0).unsqueeze(-1)
        # (N_T, BN, max_pred, d_y=1, d_param=1)

        pred_delta_times = torch.stack(pred_dt_list, dim=0) # (N_T, B, max_pred)

        obs_mask = batch.mask.transpose(0,1) # (N_T, BN)
        batch_wmse = loss.full_future_loss(full_pred_params, batch.y, pred_delta_times,
                obs_mask, pred_dist, loss_weighter, config, metric="mse")
        batch_mse = loss.full_future_loss(full_pred_params, batch.y, pred_delta_times,
                obs_mask, pred_dist, const_weighter, config, metric="mse")

        for val, name in zip((batch_wmse, batch_mse), ("wmse", "mse")):
            batch_metrics[name].append(val.detach()*batch.num_graphs)

    epoch_metrics = {name:
            (torch.sum(torch.stack(val_list))/len(data_loader.dataset)).item()
            for name, val_list in batch_metrics.items()}

    return epoch_metrics

