import torch
import numpy as np

import utils
import loss

def train_epoch(model, data_loader, opt, pred_dist, config, loss_weighter):
    model.train(True)

    batch_losses = []

    for batch in data_loader:
        batch = batch.to(config["device"]) # Move all graphs to GPU

        cur_batch_size = batch.num_graphs
        obs_mask = batch.mask.transpose(0,1) # (N_T, B*N)
        opt.zero_grad()

        full_pred_params, pred_delta_times = model.forward(
                batch) # (N_T, B*N, max_pred, d_y, d_param) and (N_T, B, max_pred)

        batch_loss = loss.full_future_loss(full_pred_params, batch.y, pred_delta_times,
                obs_mask, pred_dist, loss_weighter, config)

        batch_loss.backward()
        opt.step()

        batch_losses.append(batch_loss.detach()*cur_batch_size)

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
        cur_batch_size = batch.num_graphs
        obs_mask = batch.mask.transpose(0,1) # (N_T, B*N)

        full_pred_params, pred_delta_times = model.forward(
                batch) # (N_T, B*N, max_pred, d_y, d_param) and (N_T, B, max_pred)

        batch_wmse = loss.full_future_loss(full_pred_params, batch.y, pred_delta_times,
                obs_mask, pred_dist, loss_weighter, config, metric="mse")
        batch_mse = loss.full_future_loss(full_pred_params, batch.y, pred_delta_times,
                obs_mask, pred_dist, const_weighter, config, metric="mse")

        for val, name in zip((batch_wmse, batch_mse), ("wmse", "mse")):
            batch_metrics[name].append(val.detach()*cur_batch_size)

    epoch_metrics = {name:
            (torch.sum(torch.stack(val_list))/len(data_loader.dataset)).item()
            for name, val_list in batch_metrics.items()}

    return epoch_metrics

