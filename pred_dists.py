import torch
import torch.distributions as tdists

def gauss_pred_dist(params):
    # params: (..., 2)
    mean = params[...,0]
    std = torch.exp(params[...,1])

    return tdists.Normal(mean, std)

def gauss_fixed_pred_dist(params):
    # Gaussian distribution with fixed variance (equivalent to MSE training)
    # params: (..., 1)
    mean = params[...,0]
    std = torch.ones_like(mean )

    return tdists.Normal(mean, std)

DISTS = {
    "gauss": (gauss_pred_dist,2),
    "gauss_fixed": (gauss_fixed_pred_dist,1),
}

