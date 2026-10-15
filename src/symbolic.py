"""Symbolic simplification of a trained KAN (paper Sec. 2.5.1: sparsification -> pruning -> symbolification)."""

# %% [symfit]
import numpy as np
import sympy as sp
import torch
import torch.nn.functional as F
from .kan import SYM_LIB                                         # nb:skip


@torch.no_grad()
def fit_affine(x, y, name, a_range=(-10.0, 10.0), b_range=(-10.0, 10.0), n_grid=101, n_zoom=3):
    """Fit y ~= c*f(a*x + b) + d (paper Sec. 2.5.1, item 4) by iterative grid search over (a, b);
    for each (a, b) the optimal (c, d) is linear regression, and (a, b) is scored by R^2.
    The grid is zoomed around the best cell n_zoom times. Returns (a, b, c, d, r2) or None if no valid fit."""
    f = SYM_LIB[name][0]
    x, y = x.reshape(-1), y.reshape(-1)
    yc = y - y.mean()
    ss_y = (yc ** 2).sum().clamp_min(1e-30)
    best = (-1.0, None)
    lo_a, hi_a = a_range
    lo_b, hi_b = b_range
    for _ in range(n_zoom + 1):
        A = torch.linspace(lo_a, hi_a, n_grid, dtype=x.dtype, device=x.device)
        Bv = torch.linspace(lo_b, hi_b, n_grid, dtype=x.dtype, device=x.device)
        z = f(A[:, None, None] * x[None, None, :] + Bv[None, :, None])        # (nA, nB, N)
        ok = torch.isfinite(z).all(-1) & (z.abs().amax(-1) < 1e6)
        zc = z - z.mean(-1, keepdim=True)
        zc = torch.where(ok[..., None], zc, torch.zeros_like(zc))
        ss_z = (zc ** 2).sum(-1).clamp_min(1e-30)
        r2 = (zc @ yc) ** 2 / (ss_z * ss_y)                                   # squared correlation
        r2 = torch.where(ok & (ss_z > 1e-20), r2, torch.full_like(r2, -1.0))
        k = int(torch.argmax(r2)); ia, ib = divmod(k, n_grid)
        if float(r2[ia, ib]) > best[0]:
            best = (float(r2[ia, ib]), (float(A[ia]), float(Bv[ib])))
        a0, b0 = float(A[ia]), float(Bv[ib])
        da, db = (hi_a - lo_a) / (n_grid - 1) * 2, (hi_b - lo_b) / (n_grid - 1) * 2
        lo_a, hi_a, lo_b, hi_b = a0 - da, a0 + da, b0 - db, b0 + db            # zoom
    if best[1] is None:
        return None
    a, b = best[1]
    z = f(a * x + b)
    zc = z - z.mean()
    c = float((zc @ yc) / (zc @ zc).clamp_min(1e-30)); d = float(y.mean() - c * z.mean())
    return a, b, c, d, best[0]


# %% [symapply]
@torch.no_grad()
def layer_inputs(model, x):
    """Inputs x_l of every layer for samples x (list of tensors, length L)."""
    xs = []
    for layer in model.layers:
        xs.append(x); x = layer(x)
    return xs


@torch.no_grad()
def suggest_symbolic(model, l, i, j, x, lib=None):
    """Rank library functions for edge (l, i, j) by R^2 of the affine fit (paper's suggest_symbolic)."""
    layer = model.layers[l]
    pre = layer_inputs(model, x)[l]
    post = layer.post_activations(pre)[:, i, j]
    out = []
    for name in (lib or list(SYM_LIB)):
        r = fit_affine(pre[:, i], post, name)
        if r is not None:
            out.append((name, r[4], r[:4]))
    return sorted(out, key=lambda t: -t[1])


@torch.no_grad()
def fix_symbolic(model, l, i, j, name, x):
    """Set edge (l, i, j) to c*f(a*x+b)+d with (a,b,c,d) fitted on samples; returns R^2 (paper's fix_symbolic)."""
    layer = model.layers[l]
    pre = layer_inputs(model, x)[l]
    post = layer.post_activations(pre)[:, i, j]
    r = fit_affine(pre[:, i], post, name)
    if r is None:
        return float("nan")
    a, b, c, d, r2 = r
    layer.sym_edges[(i, j)] = name
    layer.sym_coef[i, j] = torch.tensor([a, b, c, d], dtype=layer.sym_coef.dtype, device=layer.sym_coef.device)
    return r2


@torch.no_grad()
def auto_symbolic(model, x, lib=None, tie=1e-3):
    """Symbolify every live edge, layer by layer. After each layer is fixed, activations are re-propagated
    through the (now partly symbolic) network so later edges are fitted on the inputs they will really see.
    Among candidates within `tie` of the best R^2 the simplest function is chosen. Returns {(l,i,j): (name, r2)}."""
    chosen = {}
    for l, layer in enumerate(model.layers):
        for i in range(layer.n_in):
            for j in range(layer.n_out):
                if layer.mask[i, j] <= 0 or (i, j) in layer.sym_edges:
                    continue
                cands = suggest_symbolic(model, l, i, j, x, lib)
                if not cands:
                    continue
                top = cands[0][1]
                name = min((c for c in cands if c[1] >= top - tie), key=lambda c: SYM_LIB[c[0]][2])[0]
                chosen[(l, i, j)] = (name, fix_symbolic(model, l, i, j, name, x))
    return chosen


def fine_tune_affine(model, train, steps=30):
    """After symbolification the only free parameters are the affine (a,b,c,d) of each symbolic edge
    (paper Sec. 2.5.2, step 4): train them with LBFGS; the loss should fall towards machine precision
    if the symbolic structure is right."""
    xt, yt = train
    params = []
    for layer in model.layers:
        if layer.sym_edges:
            layer.sym_coef.requires_grad_(True); params.append(layer.sym_coef)
    if not params:
        return []
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-32, tolerance_change=1e-32)

    def closure():
        opt.zero_grad(); loss = F.mse_loss(model(xt), yt); loss.backward(); return loss
    hist = []
    for _ in range(steps):
        loss = opt.step(closure); hist.append(float(loss.detach()))
    for p in params:
        p.requires_grad_(False)
    return hist


# %% [symformula]
def symbolic_formula(model, var_names, digits=3):
    """Compose the symbolic edges into a closed-form SymPy expression for each output node (paper Sec. 2.5.2,
    step 5). Raises if a live edge is still a spline."""
    exprs = [sp.Symbol(v) for v in var_names]
    for layer in model.layers:
        nxt = []
        for j in range(layer.n_out):
            total = 0
            for i in range(layer.n_in):
                if layer.mask[i, j] <= 0:
                    continue
                if (i, j) not in layer.sym_edges:
                    raise ValueError(f"edge ({i}->{j}) is not symbolic")
                a, b, c, d = [float(v) for v in layer.sym_coef[i, j]]
                f = SYM_LIB[layer.sym_edges[(i, j)]][1]
                total = total + c * f(a * exprs[i] + b) + d
            nxt.append(total)
        exprs = nxt
    return [sp.N(e, digits) if hasattr(e, "free_symbols") else e for e in exprs]


def formula_predict(expr, var_names, X):
    """Evaluate a SymPy expression on a (N, n_vars) numpy/torch array."""
    f = sp.lambdify([sp.Symbol(v) for v in var_names], expr, modules="numpy")
    X = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    out = f(*[X[:, k] for k in range(X.shape[1])])
    return np.asarray(out, dtype=float) * np.ones(len(X))
