"""LBFGS training with grid extension (paper Sec. 2.4, 3.1)."""

# %% [train]
import copy
import time
import torch
import torch.nn.functional as F
from .kan import KAN                                          # nb:skip
from .evaluate import rmse, median_rmse                       # nb:skip


def fit_lbfgs(model, train, val, steps=200, max_iter=20, history_size=10, lr=1.0, grid_update_steps=(),
              lamb=0.0, select_best=True):
    """Minimise train MSE with LBFGS (strong-Wolfe line search); keep the checkpoint with best *validation* RMSE.

    One 'step' = one optimizer.step(closure) call (= up to `max_iter` LBFGS iterations, torch default 20).
    The paper counts "200 LBFGS steps" per grid without defining a step, so this is our interpretation.
    `grid_update_steps`: step indices at which the KAN grid is re-placed from the current activations
    (paper Sec. 2.2 (3), "update grids on the fly"); the optimizer is rebuilt after each update.
    Returns per-step train/val RMSE curves. Test data is never touched here.
    """
    (xt, yt), (xv, yv) = train, val
    # lamb > 0: add the sparsity regulariser of Eq. 2.20 (KANs only); select_best=False: return the last iterate

    def make_opt():
        return torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=max_iter, history_size=history_size,
                                 line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)

    def closure():
        opt.zero_grad()
        if lamb > 0:
            out, reg = model.forward_reg(xt)
            loss = F.mse_loss(out, yt) + lamb * reg
        else:
            loss = F.mse_loss(model(xt), yt)
        loss.backward()
        return loss

    opt = make_opt()
    best, best_state, curves = float("inf"), copy.deepcopy(model.state_dict()), {"train": [], "val": []}
    for s in range(steps):
        if s in grid_update_steps and hasattr(model, "update_grid"):
            model.update_grid(xt)
            opt = make_opt()
            best = float("inf")                               # function changed slightly: restart model selection
        opt.step(closure)
        tr, va = rmse(model, xt, yt), rmse(model, xv, yv)
        model.train()
        if not (tr == tr):                                    # NaN guard
            break
        curves["train"].append(tr)
        curves["val"].append(va)
        if va < best:
            best, best_state = va, copy.deepcopy(model.state_dict())
    if select_best:
        model.load_state_dict(best_state)
    return curves


def train_kan_with_extension(width, data, grids=(3, 5, 10, 20, 50, 100), steps_per_grid=200, k=3,
                             extend=True, seed=0, **fit_kw):
    """Train a KAN through increasing grid sizes. extend=True: warm-start each finer grid from the previous
    one by least squares (Eq. 2.16). extend=False (ablation): train a freshly initialised KAN at every G."""
    torch.manual_seed(seed)
    (xt, yt), (xv, yv), (xs, ys) = data["train"], data["val"], data["test"]
    dtype = xt.dtype
    records, curves, model = [], {"train": [], "val": []}, None
    for G in grids:
        t0 = time.time()
        if model is None or not extend:
            model = KAN(width, G=G, k=k).to(device=xt.device, dtype=dtype)
            model.update_grid(xt)                             # place knots from the data
        else:
            model.update_grid(xt, G_new=G)                    # grid extension
        c = fit_lbfgs(model, (xt, yt), (xv, yv), steps=steps_per_grid, **fit_kw)
        curves["train"] += c["train"]
        curves["val"] += c["val"]
        records.append(dict(G=G, params=model.n_params(), train_rmse=rmse(model, xt, yt),
                            val_rmse=rmse(model, xv, yv), test_rmse=rmse(model, xs, ys),
                            test_median_rmse=median_rmse(model, xs, ys), seconds=time.time() - t0))
    return records, curves, model


def train_mlp(model, data, steps=1200, **fit_kw):
    """Train an MLP baseline with the same optimiser, data and step budget as the KAN."""
    (xt, yt), (xv, yv), (xs, ys) = data["train"], data["val"], data["test"]
    t0 = time.time()
    c = fit_lbfgs(model, (xt, yt), (xv, yv), steps=steps, **fit_kw)
    rec = dict(params=model.n_params(), train_rmse=rmse(model, xt, yt), val_rmse=rmse(model, xv, yv),
               test_rmse=rmse(model, xs, ys), test_median_rmse=median_rmse(model, xs, ys),
               seconds=time.time() - t0)
    return rec, c
