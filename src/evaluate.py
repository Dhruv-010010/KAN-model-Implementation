"""Metrics used in the paper: (root) mean squared error and a robust median-based variant."""

# %% [evaluate]
import numpy as np
import torch


@torch.no_grad()
def predict(model, x):
    model.eval()
    return model(x)


@torch.no_grad()
def rmse(model, x, y):
    """sqrt(mean of squared errors): the 'test RMSE' on the y-axes of paper Fig. 2.3 / 3.1."""
    return torch.sqrt(torch.mean((predict(model, x) - y) ** 2)).item()


@torch.no_grad()
def median_rmse(model, x, y):
    """sqrt(median of squared errors): the robust variant discussed under Fig. 2.3 (boundary outliers)."""
    return torch.sqrt(torch.median((predict(model, x) - y) ** 2)).item()


def scaling_exponent(xs, errs):
    """Slope alpha of  err ~ xs^(-alpha)  from a least-squares line in log-log space."""
    slope, _ = np.polyfit(np.log10(np.asarray(xs, float)), np.log10(np.asarray(errs, float)), 1)
    return -float(slope)
