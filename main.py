import torch
import torch_geometric as ptg
import numpy as np
import argparse
import matplotlib
import wandb
import time
import copy
import os
import json
from tueplots import bundles
import matplotlib.pyplot as plt

import utils
import constants
import train
import train_transformer
import pred_dists
import visualization as vis
from models.gru_model import GRUModel
from models.gru_node_model import GRUNodeModel
from models.gru_graph_model import GRUGraphModel
from models.transformer import TransformerForecaster
from models.transformer_joint import TransformerJointForecaster

# Allow for saving plots on remote machine
matplotlib.use('Agg')

MODELS = {
    "grud_joint": GRUModel, # Ignore graph structure, evolve single joint latent state
    "grud_node": GRUNodeModel, # Treat each node independently, independent latent state
    "tgnn4i": GRUGraphModel, # Utilizes graph structure
    "transformer_node": TransformerForecaster,
    "transformer_joint": TransformerJointForecaster,
}

def get_config():
    parser = argparse.ArgumentParser(description='Train Models')
    # If config file should be used
    parser.add_argument("--config", type=str, help="Config file to read run config from")

    # General
    parser.add_argument("--model", type=str, default="tgnn4i",
            help="Which dataset to use")
    parser.add_argument("--dataset", type=str, default="la_node_0.25",
            help="Which dataset to use")
    parser.add_argument("--seed", type=int, default=42,
            help="Seed for random number generator")
    parser.add_argument("--optimizer", type=str, default="adam",
            help="Optimizer to use for training")
    parser.add_argument("--init_points", type=int, default=5,
            help="Number of points to observe before prediction start")
    parser.add_argument("--test", type=int, default=0,
            help="Also evaluate on test set after training is done")
    parser.add_argument("--use_features", type=int, default=1,
            help="If additional input features should be used")
    parser.add_argument("--load", type=str,
            help="Load model parameters from path")

    # Model Architecture
    parser.add_argument("--gru_layers", type=int, default=1,
            help="Layers of GRU units")
    parser.add_argument("--decay_type", type=str, default="dynamic",
            help="Parametrization of GRU decay to use (none/to_const/dynamic)")
    parser.add_argument("--periodic", type=int, default=0,
            help="If latent state dynamics should include periodic component")
    parser.add_argument("--time_input", type=int, default=1,
            help="Concatenate time (delta_t) to the input at each timestep")
    parser.add_argument("--mask_input", type=int, default=1,
            help="Concatenate the observation mask as input")
    parser.add_argument("--hidden_dim", type=int, default=32,
            help="Dimensionality of hidden state in GRU units (latent node state))")
    parser.add_argument("--n_fc", type=int, default=2,
            help="Number of fully connected layers after GRU units")
    parser.add_argument("--pred_gnn", type=int, default=1,
            help="Number of GNN-layers to use in predictive part of model")
    parser.add_argument("--gru_gnn", type=int, default=1,
            help="Number of GNN layers used for GRU-cells")
    parser.add_argument("--gnn_type", type=str, default="graphconv",
            help="Type of GNN-layers to use")
    parser.add_argument("--node_params", type=int, default=1,
            help="Use node-specific parameters for initial state and decay target")
    parser.add_argument("--learn_init_state", type=int, default=1,
            help="If the initial state of GRU-units should be learned (otherwise 0)")

    # Training
    parser.add_argument("--epochs", type=int,
            help="How many epochs to train for", default=10)
    parser.add_argument("--val_interval", type=int, default=1,
            help="Evaluate model every val_interval:th epoch")
    parser.add_argument("--patience", type=int, default=20,
            help="How many evaluations to wait for improvement in val loss")
    parser.add_argument("--pred_dist", type=str, default="gauss_fixed",
            help="Predictive distribution")
    parser.add_argument("--lr", type=float,
            help="Learning rate", default=1e-3)
    parser.add_argument("--l2_reg", type=float,
            help="L2-regularization coefficient", default=0.)
    parser.add_argument("--batch_size", type=int,
            help="Batch size", default=32)
    parser.add_argument("--state_updates", type=str, default="obs",
            help="When the node state should be updated (all/obs/hop)")
    parser.add_argument("--loss_weighting", type=str, default="exp,0.04",
            help="Function to weight loss with, given as: name,param1,...,paramK")
    parser.add_argument("--max_pred", type=int, default=10,
            help="Maximum number of time indices forward to predict")

    # Plotting
    parser.add_argument("--plot_pred", type=int, default=3,
            help="Number of prediction plots to make")
    parser.add_argument("--max_nodes_plot", type=int, default=3,
            help="Maximum number of nodes to plot predictions for")
    parser.add_argument("--save_pdf", type=int, default=0,
            help="If pdf:s should be generated for plots (NOTE: Requires much space)")

    args = parser.parse_args()
    config = vars(args)

    # Read additional config from file
    if args.config:
        assert os.path.exists(args.config), "No config file: {}".format(args.config)
        with open(args.config) as json_file:
            config_from_file = json.load(json_file)

        # Make sure all options in config file also exist in argparse config.
        # Avoids choosing wrong parameters because of typos etc.
        unknown_options = set(config_from_file.keys()).difference(set(config.keys()))
        unknown_error = "\n".join(["Unknown option in config file: {}".format(opt)
            for opt in unknown_options])
        assert (not unknown_options), unknown_error

        config.update(config_from_file)

    # Some asserts
    assert config["model"] in MODELS, f"Unknown model: {config['model']}"
    assert config["optimizer"] in constants.OPTIMIZERS, (
            f"Unknown optimizer: {config['optimizer']}")
    assert config["pred_dist"] in pred_dists.DISTS, (
            f"Unknown predictive distribution: {config['pred_dist']}")
    assert config["gnn_type"] in constants.GNN_LAYERS, (
            f"Unknown gnn_type: {config['gnn_type']}")
    assert config["init_points"] > 0, "Need to have positive number of init points"
    assert (not bool(config["periodic"])) or (config["hidden_dim"] % 2 == 0), (
            "hidden_dim must be even when using periodic latent dynamics")

    if config["plot_pred"] > config["batch_size"]:
        print(f"Warning: Can only make {config['batch_size']} plots")
        config["plot_pred"] = config["batch_size"]

    return config

def main():
    config = get_config()

    # Set all random seeds
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")

        # For reproducability on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    # Load data
    torch.multiprocessing.set_sharing_strategy('file_system') # Fix for num_workers > 0
    train_loader, val_loader, test_loader = utils.load_temporal_graph_data(
            config["dataset"], config["batch_size"],
            compute_hop_mask=(config["state_updates"] == "hop"), L_hop=config["gru_gnn"])

    # Init wandb
    wandb_name = f"{config['dataset']}_{config['model']}_{time.strftime('%H-%M-%S')}"
    wandb.init(project=constants.WANDB_PROJECT, config=config, name=wandb_name)
    # Additional config needed for some model setup (need/should not be logged to wandb)
    config["num_nodes"] = train_loader.dataset[0].num_nodes
    config["time_steps"] = train_loader.dataset[0].t.shape[1]
    config["device"] = device
    config["y_dim"] = train_loader.dataset[0].y.shape[-1]

    config["has_features"] = hasattr(train_loader.dataset[0], "features") and\
        bool(config["use_features"])
    if config["has_features"]:
        config["feature_dim"] = train_loader.dataset[0].features.shape[-1]
    else:
        config["feature_dim"] = 0

    # param_dim is number of parameters in predictive distribution
    pred_dist, config["param_dim"] = pred_dists.DISTS[config["pred_dist"]]

    # Parse loss weighting function
    loss_weight_func = utils.parse_loss_weight(config["loss_weighting"])

    # Create model, optimizer
    model = MODELS[config["model"]](config).to(device)
    if config["load"]:
        model.load_state_dict(torch.load(config["load"], map_location=device))
        print(f"Parameters loaded from: {config['load']}")
    opt = constants.OPTIMIZERS[config["optimizer"]](model.parameters(), lr=config["lr"],
            weight_decay=config["l2_reg"])

    is_transformer = config["model"].startswith("transformer")
    if is_transformer:
        train_epoch = train_transformer.train_epoch
        val_epoch = train_transformer.val_epoch
        test_epoch = train_transformer.test_epoch
    else:
        train_epoch = train.train_epoch
        val_epoch = train.val_epoch
        test_epoch = train.val_epoch # Note: Same function, but test data should be used

    # Train model
    best_val_loss = np.inf
    best_val_metrics = None
    best_val_epoch = -1 # Index of the best epoch
    best_params = None

    model.train()
    for epoch_i in range(1, config["epochs"]+1):
        epoch_train_loss = train_epoch(model, train_loader, opt, pred_dist,
                config, loss_weight_func)

        if (epoch_i % config["val_interval"]== 0):
            # Validate, evaluate
            with torch.no_grad():
                epoch_val_metrics = val_epoch(model, val_loader, pred_dist,
                        loss_weight_func, config)

            log_metrics = {"train_loss": epoch_train_loss}
            log_metrics.update({f"val_{metric}": val for
                metric, val in epoch_val_metrics.items()})

            epoch_val_loss = log_metrics["val_wmse"] # Use wmse as main metric
            epoch_val_mse = log_metrics["val_mse"]

            print(f"Epoch {epoch_i}:\t train_loss: {epoch_train_loss:.6f} "\
                    f"\tval_wmse: {epoch_val_loss:.6f} \tval_mse: {epoch_val_mse:.6f}")

            wandb.log(log_metrics, step=epoch_i, commit=True)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_val_metrics = log_metrics
                best_val_epoch = epoch_i
                best_params = copy.deepcopy(model.state_dict())

            if (epoch_i - best_val_epoch)/config["val_interval"] >= config["patience"]:
                # No improvement, end training
                print("Val loss no longer improving, stopping training.")
                break

    # Save things
    param_save_path = os.path.join(wandb.run.dir, constants.PARAM_FILE_NAME)
    torch.save(best_params, param_save_path)

    # Restore parameters and plot
    model.load_state_dict(best_params)
    if not is_transformer:
        with torch.no_grad(), plt.rc_context(bundles.aistats2023()):
            val_pred_plots = vis.plot_step_prediction(model, val_loader, config["plot_pred"],
                    pred_dist, config)
            for fig_i, fig in enumerate(val_pred_plots):
                if config["save_pdf"]:
                    save_path = os.path.join(wandb.run.dir, f"val_pred_{fig_i}.pdf")
                    fig.savefig(save_path)

                wandb.log({"val_pred": wandb.Image(fig)})

            all_pred_plot, bin_errors, unique_dts, unique_counts = vis.plot_all_predictions(
                    model, val_loader, pred_dist, loss_weight_func, config)
            if config["save_pdf"]:
                save_path = os.path.join(wandb.run.dir, f"val_error.pdf")
                all_pred_plot.savefig(save_path)
            wandb.log({"val_error": wandb.Image(all_pred_plot)})

    # Wandb summary
    del best_val_metrics["train_loss"] # Exclude this one from summary update
    for metric, val in best_val_metrics.items():
        wandb.run.summary[metric] = val
    if config["decay_type"] == "to_const":
        # Decay parameters histogram
        with torch.no_grad():
            for cell_id, gru_cell in enumerate(model.gru_cells):
                wandb.run.summary.update({f"GRU_{cell_id}_decay":
                    wandb.Histogram(gru_cell.decay_weight.cpu())})

    # (Optionally) Evaluate on test set
    if config["test"]:
        with torch.no_grad(), plt.rc_context(bundles.aistats2023()):
            test_metrics = test_epoch(model, test_loader, pred_dist,
                    loss_weight_func, config)
            test_metric_dict = {f"test_{name}": val
                    for name, val in test_metrics.items()}
            wandb.run.summary.update(test_metric_dict)

            print("Test set evaluation:")
            for name, val in test_metric_dict.items():
                print(f"{name}:\t {val}")

            if not is_transformer:
                # Compute test errors at different delta-ts
                all_pred_plot, bin_errors, unique_dts, unique_counts =\
                    vis.plot_all_predictions(
                        model, val_loader, pred_dist, loss_weight_func, config)
                if config["save_pdf"]:
                    save_path = os.path.join(wandb.run.dir, f"test_error.pdf")
                    all_pred_plot.savefig(save_path)
                wandb.log({"test_error": wandb.Image(all_pred_plot)})

                # Save binned errors
                np.save(os.path.join(wandb.run.dir, "test_bin_errors.npy"), bin_errors)
                np.save(os.path.join(wandb.run.dir, "test_bin_dt.npy"), unique_dts)
                np.save(os.path.join(wandb.run.dir, "test_bin_counts.npy"), unique_counts)


if __name__ == "__main__":
    main()

