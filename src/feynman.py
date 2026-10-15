"""Symbolic-regression task: Feynman equations (paper Sec. 3.3, Table 2) - data, human-constructed KANs,
equal-parameter MLPs and the sparsify -> prune -> symbolify pipeline."""

# %% [feynman]
import math
import time
import numpy as np
import copy
import torch
import sympy as sp
from .kan import KAN                                             # nb:skip
from .models import matched_mlp                                  # nb:skip
from .train import fit_lbfgs, train_kan_with_extension, train_mlp   # nb:skip
from .evaluate import rmse                                       # nb:skip
from .symbolic import auto_symbolic, fine_tune_affine, symbolic_formula, formula_predict   # nb:skip

# Dimensionless Feynman equations from paper Table 2 (two variables each).
# 'human' = human-constructed KAN shape and 'paper' = (human-KAN, pruned-KAN, unpruned-KAN, MLP) test RMSE, both
# copied from Table 2. Sampling ranges are OUR choice (the paper only states (-1,1) for I.16.6).
EQUATIONS = {
    # the paper's own worked example (Sec. 2.5.2, Fig. 2.4): the shape [2,5,1] -> pruned [2,1,1] -> sin, x^2, exp
    "exp_sin": dict(vars=["x", "y"], ranges=[(-1, 1), (-1, 1)], human=[2, 1, 1],
                    fn=lambda x: np.exp(np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2),
                    text="exp(sin(pi x) + y^2)", paper=None),
    "I.6.2":   dict(vars=["theta", "sigma"], ranges=[(-2, 2), (0.5, 2)], human=[2, 2, 1, 1],
                    fn=lambda x: np.exp(-x[:, 0] ** 2 / (2 * x[:, 1] ** 2)) / np.sqrt(2 * np.pi * x[:, 1] ** 2),
                    text="exp(-theta^2/(2 sigma^2)) / sqrt(2 pi sigma^2)", paper=(7.66e-5, 2.86e-5, 4.60e-5, 1.45e-4)),
    "I.12.11": dict(vars=["a", "theta"], ranges=[(-1, 1), (-2, 2)], human=[2, 2, 2, 1],
                    fn=lambda x: 1 + x[:, 0] * np.sin(x[:, 1]),
                    text="1 + a sin(theta)", paper=(2.07e-3, 1.39e-3, 9.13e-4, 6.71e-4)),
    "I.16.6":  dict(vars=["a", "b"], ranges=[(-1, 1), (-1, 1)], human=[2, 2, 2, 2, 2, 1],
                    fn=lambda x: (x[:, 0] + x[:, 1]) / (1 + x[:, 0] * x[:, 1]),
                    text="(a + b) / (1 + a b)", paper=(1.06e-3, 1.19e-3, 1.53e-3, 6.20e-4)),
    "I.27.6":  dict(vars=["a", "b"], ranges=[(0, 1), (0, 1)], human=[2, 2, 1, 1],
                    fn=lambda x: 1 / (1 + x[:, 0] * x[:, 1]),
                    text="1 / (1 + a b)", paper=(2.22e-4, 1.94e-4, 2.14e-4, 2.46e-4)),
    "I.44.4":  dict(vars=["n", "a"], ranges=[(-1, 1), (0.5, 2)], human=[2, 2, 1],
                    fn=lambda x: x[:, 0] * np.log(x[:, 1]),
                    text="n ln(a)", paper=(2.30e-5, 2.43e-5, 1.10e-4, 3.99e-4)),
    "II.35.18": dict(vars=["n0", "a"], ranges=[(-1, 1), (-1, 1)], human=[2, 1, 1],
                    fn=lambda x: x[:, 0] / (np.exp(x[:, 1]) + np.exp(-x[:, 1])),
                    text="n0 / (exp(a) + exp(-a))", paper=(4.13e-4, 1.58e-4, 7.71e-5, 7.92e-5)),
}


def make_equation_data(name, n_train=2000, n_val=1000, n_test=5000, seed=0, dtype=torch.float64, device="cpu"):
    """i.i.d. uniform samples of the equation's variables; independent streams for train/val/test (no leakage)."""
    eq = EQUATIONS[name]
    out = {}
    for split, n, off in (("train", n_train, 1), ("val", n_val, 2), ("test", n_test, 3)):
        rng = np.random.default_rng([seed, off, 77])
        x = np.stack([rng.uniform(lo, hi, n) for lo, hi in eq["ranges"]], axis=1)
        out[split] = (torch.tensor(x, dtype=dtype, device=device),
                      torch.tensor(eq["fn"](x), dtype=dtype, device=device).unsqueeze(1))
    return out


def run_human_kan(name, data, grids=(3, 5, 10), steps_per_grid=100, k=3, seed=0):
    """Train the human-constructed KAN shape of Table 2 with grid extension."""
    rec, _, model = train_kan_with_extension(EQUATIONS[name]["human"], data, grids=grids, steps_per_grid=steps_per_grid,
                                             k=k, extend=True, seed=seed, grid_update_steps=(0, 10, 20, 30, 40))
    return dict(shape=EQUATIONS[name]["human"], params=model.n_params(), test_rmse=rec[-1]["test_rmse"],
                val_rmse=rec[-1]["val_rmse"]), model


def run_matched_mlp(n_params, data, depths=(2, 3, 4), steps=300, seed=0):
    """Equal-parameter baseline: for each depth pick the width whose parameter count is closest to n_params,
    train with the same optimiser, and keep the depth with the lowest *validation* RMSE."""
    n_in = data["train"][0].shape[1]
    best = None
    for depth in depths:
        torch.manual_seed(seed)
        mlp, gap = matched_mlp(n_params, n_in, depth)
        mlp = mlp.to(device=data["train"][0].device, dtype=data["train"][0].dtype)
        rec, _ = train_mlp(mlp, data, steps=steps)
        row = dict(depth=depth, params=rec["params"], param_gap=gap, val_rmse=rec["val_rmse"], test_rmse=rec["test_rmse"])
        if best is None or row["val_rmse"] < best["val_rmse"]:
            best = row
    return best


def simplify_kan(name, data, width=None, lambdas=(1e-2, 1e-3), seeds=(0, 1), G=5, k=3, theta=1e-2,
                 steps_reg=100, steps_retrain=100, steps_affine=40, log=print):
    """Paper Sec. 2.5: train a wide KAN with sparsity regularisation, prune, retrain, symbolify every live edge,
    fine-tune the affine parameters and read off a closed-form formula. The (lambda, seed) combination is chosen by
    the *validation* RMSE of the final symbolic model. Returns the best result dict (or None)."""
    eq = EQUATIONS[name]
    n_in = len(eq["vars"])
    width = width or [n_in, 5, 1]
    (xt, yt), (xv, yv), (xs, ys) = data["train"], data["val"], data["test"]
    dtype, device = xt.dtype, xt.device
    best = None
    for lamb in lambdas:
        for seed in seeds:
            torch.manual_seed(seed)
            m = KAN(width, G=G, k=k).to(device=device, dtype=dtype)
            m.update_grid(xt)
            fit_lbfgs(m, (xt, yt), (xv, yv), steps=steps_reg, lamb=lamb, select_best=False, grid_update_steps=(0, 10, 20))
            model_reg = copy.deepcopy(m)                          # kept for visualisation (before pruning)
            shape = m.prune(xt, theta=theta)
            fit_lbfgs(m, (xt, yt), (xv, yv), steps=steps_retrain, select_best=False)
            pruned_rmse, pruned_params = rmse(m, xs, ys), m.effective_params()
            edges = auto_symbolic(m, xt)
            fine_tune_affine(m, (xt, yt), steps=steps_affine)
            try:
                expr = symbolic_formula(m, eq["vars"])[0]
                pred = torch.tensor(formula_predict(expr, eq["vars"], xs), dtype=dtype, device=device).reshape(-1, 1)
                sym_test = torch.sqrt(torch.mean((pred - ys) ** 2)).item()
                val_pred = torch.tensor(formula_predict(expr, eq["vars"], xv), dtype=dtype, device=device).reshape(-1, 1)
                sym_val = torch.sqrt(torch.mean((val_pred - yv) ** 2)).item()
            except Exception as e:                                # a live edge was not symbolifiable
                expr, sym_test, sym_val = None, float("nan"), float("nan")
            if not (math.isfinite(sym_val) and math.isfinite(sym_test)):
                expr = None
            log(f"  {name} lambda={lamb:g} seed={seed}: pruned shape {shape}, pruned test RMSE {pruned_rmse:.2e}, "
                f"symbolic test RMSE {sym_test:.2e}")
            row = dict(lamb=lamb, seed=seed, pruned_shape=shape, pruned_params=pruned_params, pruned_test_rmse=pruned_rmse,
                       symbolic_val_rmse=sym_val, symbolic_test_rmse=sym_test, formula=expr, edges=edges, model=m, model_reg=model_reg)
            if expr is not None and math.isfinite(sym_val) and (best is None or sym_val < best["symbolic_val_rmse"]):
                best = row
    return best
