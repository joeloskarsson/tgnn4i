import networkx as nx
import torch_geometric as ptg
import numpy as np
from matplotlib.lines import Line2D
import matplotlib
from tueplots import axes, bundles
import torch
import matplotlib
import matplotlib.pyplot as plt

import loss
import utils

@matplotlib.rc_context({**bundles.aistats2023(family="serif"), **axes.lines()})
def plot_graph(graph, show=False, save_as=None, node_size=30):
    # Perform all computation on cpu (for plotting)
    graph = graph.clone().to(device="cpu")
    n_nodes = graph.num_nodes

    graph_networkx = ptg.utils.to_networkx(graph)
    plot_pos = graph.pos.numpy()

    if graph.x:
        node_color = graph.x.flatten()
    else:
        node_color = np.zeros(n_nodes)

    fig, ax = plt.subplots(figsize=(15,12))
    nodes = nx.draw_networkx_nodes(graph_networkx, pos=plot_pos, node_size=node_size,
            node_color=node_color)
    nx.draw_networkx_edges(graph_networkx, pos=plot_pos, width=0.5)

    if graph.x:
        plt.colorbar(nodes)

    if show:
        plt.show()

    if save_as:
        plt.savefig(f"{save_as}.pdf")

@torch.no_grad()
@matplotlib.rc_context({**bundles.aistats2023(family="serif"), **axes.lines()})
def plot_step_prediction(model, data_loader, n_plots, pred_dist, config,
        batch_prediction=None, data_batch=None):
    if batch_prediction is None:
        model.train(False)
        data_batch = next(iter(data_loader)).to(config["device"])
        all_pred_params, _ = model.forward(data_batch) # (N_T, B*N, max_pred, d_y, d_params)

        target_batch = data_batch.y.transpose(0,1) # Shape (N_T, B*N, d_y)
        t_batch = data_batch.t.transpose(0,1) # Shape (N_T, B)
        mask_batch = data_batch.mask.transpose(0,1) # Shape (N_T, B*N)
    else:
        all_pred_params = batch_prediction # (N_T, B*N, max_pred, d_y, d_params)
        target_batch = data_batch["y"].transpose(0,1) # Shape (N_T, B*N, d_y)
        t_batch = data_batch["t"].transpose(0,1) # Shape (N_T, B)
        mask_batch = data_batch["mask"] #(N_T, B*N)

    next_pred_params = all_pred_params[:-1, :, 0] # (N_T-1, BN, d_y, d_params)

    pred_params = next_pred_params.reshape(config["time_steps"]-1, -1,
            config["num_nodes"], config["y_dim"], config["param_dim"])
    target_batch = target_batch.reshape(
            config["time_steps"], -1, config["num_nodes"], config["y_dim"])
    mask_batch = mask_batch.reshape(config["time_steps"], -1, config["num_nodes"])

    dist = pred_dist(pred_params)
    pred_mean = dist.mean # (N_T-1, B, N, d_y)
    pred_std = dist.stddev # (N_T-1, B, N, d_y)

    # Optionally restrict number of nodes to plot for
    n_plot_nodes = min(config["num_nodes"], config["max_nodes_plot"])
    mean_batch = pred_mean[:,:,:n_plot_nodes] # (N_T-1, B, N_plot_nodes, d_y)
    std_batch = pred_std[:,:,:n_plot_nodes] # (N_T-1, B, N_plot_nodes, d_y)
    target_batch = target_batch[:,:,:n_plot_nodes] # (N_T, B, N_plot_nodes, d_y)
    mask_batch = mask_batch[:,:,:n_plot_nodes] # (N_T, B, N_plot_nodes)

    line_colors = plt.cm.rainbow(np.linspace(0, 1, n_plot_nodes))

    figs = []
    for plot_i in range(n_plots):
        vis_target = target_batch[1:,plot_i].cpu().numpy() # (N_T-1, N, d_y)
        vis_mean = mean_batch[:,plot_i].cpu().numpy() # (N_T-1, N_plot_nodes, d_y)
        vis_std = std_batch[:,plot_i].cpu().numpy() # (N_T-1, N_plot_nodes, d_y)
        vis_mask = mask_batch[1:,plot_i].cpu().numpy() # (N_T-1, N_plot_nodes)
        vis_t = t_batch[1:,plot_i].cpu().numpy() # (N_T-1,)

        fig, axes = plt.subplots(1, config["y_dim"], squeeze=False)
        # Iterate over y-dimensions
        for y_dim, ax in enumerate(axes[0]):
            for node_target, node_pred, node_std, node_mask, col in zip(
                    vis_target[:,:,y_dim].T, vis_mean[:,:,y_dim].T, vis_std[:,:,y_dim].T,
                    vis_mask.T, line_colors):
                # Mask observations for plotted node
                node_plot_mask = (node_mask == 1)
                node_plot_times = vis_t[node_plot_mask]
                node_plot_targets = node_target[node_plot_mask]
                node_plot_pred = node_pred[node_plot_mask]
                node_plot_std = node_std[node_plot_mask]

                # Plot target
                ax.plot(node_plot_times, node_plot_targets, ls="-", marker="o",
                        zorder=2, c=col)

                if config["pred_dist"] == "gauss_fixed":
                    # Plot only prediction
                    ci = 1.0*node_plot_std # \pm one std.
                    ax.plot(node_plot_times, node_plot_pred, ls=":", marker="x",
                            zorder=2, c=col)
                else:
                    # Plot prediction with error bars
                    ci = 1.0*node_plot_std # \pm one std.
                    ax.errorbar(node_plot_times, node_plot_pred, yerr=ci, ls=":", marker="x",
                            zorder=2, c=col, capsize=2)

                # Show warm-up area
                ax.axvspan(0., vis_t[config["init_points"]-1], color="black",
                        alpha=0.05, zorder=1)
                ax.set_xlim(vis_t[0], vis_t[-1])

        axes[0,0].legend([
                Line2D([0], [0], color="blue", ls="-", marker="o", c="blue"),
                Line2D([0], [0], color="blue", ls=":", marker="x", c="blue"),
            ], [
                "target",
                "prediction",
            ])

        figs.append(fig)

    # Return list of figures
    return figs

# Plot some examples after pre-processing
@matplotlib.rc_context({**bundles.aistats2023(family="serif"), **axes.lines()})
def plot_preprocessed(data_save_dict, n_plots=1, max_nodes_plot=10):
    train_ars = data_save_dict["train"]
    for plot_i in range(n_plots):
        y_plot_tensor = train_ars["y"][plot_i].clone() # Shape (N_T, N, d_y)
        t_plot = train_ars["t"][plot_i].numpy() # Shape (N_T)

        y_plot = y_plot_tensor[:, :max_nodes_plot].numpy() # (N_T, N, d_y)
        actual_n_plot = y_plot.shape[1]
        obs_mask_plot = train_ars["mask"][plot_i].to(bool).numpy().T[
                :,:actual_n_plot] # Bool array (N_T, N)

        d_y = y_plot.shape[-1]
        fig, axes = plt.subplots(1, d_y, squeeze=False)
        for ys, mask in zip(y_plot.transpose(1,0,2), obs_mask_plot.T):
            # Plot all y dimensions
            t_masked = t_plot[mask]
            y_masked = ys[mask] # (N_T, d_y)
            for y_dim, ax in enumerate(axes[0]):
                ax.plot(t_masked, y_masked[:,y_dim])

        ax.set_xlabel("t")
        ax.set_ylabel("y")

        plt.show()

@matplotlib.rc_context({**bundles.aistats2023(family="serif"), **axes.lines()})
def scatter_all_predictions(model, data_loader, pred_dist, config):
    model.train(False)
    const_weighter = utils.parse_loss_weight("const")

    errors = [[] for _ in range(config["y_dim"])]
    dts = []

    for batch in data_loader:
        batch = batch.to(config["device"]) # Move all graphs to GPU
        obs_mask = batch.mask.transpose(0,1) # (N_T, B*N)

        full_pred_params, pred_delta_times = model.forward(
                batch) # (N_T, B*N, max_pred, d_y, d_param) and (N_T, B, max_pred)

        _, batch_errors = loss.full_future_loss(full_pred_params, batch.y,
                pred_delta_times, obs_mask, pred_dist, const_weighter,
                config, metric="mse", return_errors=True) # (N_T, B, N, max_pred, d_y)

        # Same mask for all y-dims
        batch_obs_mask = (batch_errors[:,:,:,:,0] > 0.) # (N_T, B, N, max_pred)

        pred_delta_times_rs = pred_delta_times.unsqueeze(2).repeat_interleave(
                config["num_nodes"], 2) # (N_T, B, N, max_pred)

        # Remove predictions from init periods
        batch_errors = batch_errors[config["init_points"]:]
        batch_obs_mask = batch_obs_mask[config["init_points"]:]
        pred_delta_times_rs = pred_delta_times_rs[config["init_points"]:]

        for dim_i in range(config["y_dim"]):
            masked_errors = batch_errors[:,:,:,:,dim_i][batch_obs_mask]
            errors[dim_i].append(masked_errors)

        masked_dts = pred_delta_times_rs[batch_obs_mask]
        dts.append(masked_dts)

        # Keep break to plot only single batch (?)
        break

    all_errors = torch.stack([torch.cat(err_list ) for err_list in errors],
            dim=0) # (d_y, N_err)
    all_dts = torch.cat(dts) # (N_err,)

    # Plot all y dimensions
    fig, axes = plt.subplots(1, config["y_dim"], squeeze=False)
    for dim_errors, ax in zip(all_errors, axes[0]):
        ax.scatter(all_dts.cpu().numpy(), dim_errors.cpu().numpy(),
                c="green", s=1, marker=".")

        ax.set_xlabel(r"$\Delta$t")
        ax.set_ylabel("MSE")

    return fig

@matplotlib.rc_context({**bundles.aistats2023(family="serif"), **axes.lines()})
def plot_all_predictions(model, data_loader, pred_dist, loss_weighter, config):
    model.train(False)
    const_weighter = utils.parse_loss_weight("const")

    errors = [[] for _ in range(config["y_dim"])]
    dts = []

    for batch in data_loader:
        batch = batch.to(config["device"]) # Move all graphs to GPU
        obs_mask = batch.mask.transpose(0,1) # (N_T, B*N)

        full_pred_params, pred_delta_times = model.forward(
                batch) # (N_T, B*N, max_pred, d_y, d_param) and (N_T, B, max_pred)

        _, batch_errors = loss.full_future_loss(full_pred_params, batch.y,
                pred_delta_times, obs_mask, pred_dist, const_weighter,
                config, metric="mse", return_errors=True) # (N_T, B, N, max_pred, d_y)

        # Same mask for all y-dims
        batch_obs_mask = (batch_errors[:,:,:,:,0] > 0.) # (N_T, B, N, max_pred)

        pred_delta_times_rs = pred_delta_times.unsqueeze(2).repeat_interleave(
                config["num_nodes"], 2) # (N_T, B, N, max_pred)

        # Remove predictions from init periods
        batch_errors = batch_errors[config["init_points"]:]
        batch_obs_mask = batch_obs_mask[config["init_points"]:]
        pred_delta_times_rs = pred_delta_times_rs[config["init_points"]:]

        for dim_i in range(config["y_dim"]):
            masked_errors = batch_errors[:,:,:,:,dim_i][batch_obs_mask]
            errors[dim_i].append(masked_errors)

        masked_dts = pred_delta_times_rs[batch_obs_mask]
        dts.append(masked_dts)

        # Keep break to plot only single batch (?)
        #break

    all_errors = torch.stack([torch.cat(err_list ) for err_list in errors],
            dim=0) # (d_y, N_err)
    all_dts = torch.cat(dts) # (N_err,)

    # Round to fix numerical differences
    all_dts_rounded = torch.round(all_dts, decimals=3)
    unique_dts, unique_counts = torch.unique(all_dts_rounded, sorted=True,
            return_counts=True)
    unique_dts_np = unique_dts.cpu().numpy()
    unique_counts_np = unique_counts.cpu().numpy()

    # Plot all y dimensions
    loss_weighting = loss_weighter(unique_dts).cpu().numpy()
    fig, axes = plt.subplots(1, config["y_dim"], squeeze=False)
    for dim_errors, ax in zip(all_errors, axes[0]):
        bin_errors = np.array([torch.mean(dim_errors[all_dts_rounded == cur_dt]).item()
            for cur_dt in unique_dts])
        max_error = bin_errors.max()

        ax.plot(unique_dts_np, bin_errors, c="green", marker=".", label="MSE")
        ax.plot(unique_dts_np, loss_weighting*max_error, c="orange",
                label="Loss Weight") # plot over whole y-axis

        ax.set_xlim(0., unique_dts_np.max())
        ax.set_ylim(0., bin_errors.max())

        ax.set_xlabel(r"$\Delta$t")
        ax.set_ylabel("MSE")

        ax.legend()

    return fig, bin_errors, unique_dts_np, unique_counts_np

