"""Synthetic regression datasets from paper Sec. 2.4 / 3.1."""

# %% [data]
import numpy as np
import torch
from scipy.special import j0

# name -> (input_dim, function on a numpy array of shape (n, d), paper reference)
TARGETS = {
    "exp_sin": (2, lambda x: np.exp(np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2), "Sec. 2.4 / Fig. 2.3; Sec. 3.1 (2)"),
    "xy": (2, lambda x: x[:, 0] * x[:, 1], "Sec. 3.1 (3)"),
    "bessel": (1, lambda x: j0(20 * x[:, 0]), "Sec. 3.1 (1)"),
}


def make_dataset(name, n_train=1000, n_val=1000, n_test=10000, seed=0,
                 low=-1.0, high=1.0, dtype=torch.float64):
    """Sample i.i.d. uniform inputs on [low, high]^d; train/val/test use independent random streams.

    No statistic is ever computed from val/test data, so there is no preprocessing leakage
    (inputs are used raw; they already lie in the KAN grid range [-1, 1]).
    """
    d, fn, _ = TARGETS[name]
    out = {}
    for split, n, off in (("train", n_train, 1), ("val", n_val, 2), ("test", n_test, 3)):
        rng = np.random.default_rng([seed, off])
        x = rng.uniform(low, high, size=(n, d))
        out[split] = (torch.tensor(x, dtype=dtype), torch.tensor(fn(x), dtype=dtype).unsqueeze(1))
    return out
