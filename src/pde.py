"""PDE toy task (paper Sec. 3.4): 2-D Poisson equation solved with a physics-informed loss."""

# %% [pde]
import math
import time
import numpy as np
import torch
from .kan import KAN                                             # nb:skip
from .models import matched_mlp                                  # nb:skip


def u_true(z):
    """Exact solution u(x,y) = sin(pi x) sin(pi y^2)."""
    return torch.sin(math.pi * z[:, 0]) * torch.sin(math.pi * z[:, 1] ** 2)


def f_rhs(z):
    """Right-hand side f = u_xx + u_yy = -pi^2 (1 + 4y^2) sin(pi x) sin(pi y^2) + 2 pi sin(pi x) cos(pi y^2)."""
    x, y = z[:, 0], z[:, 1]
    return (-math.pi ** 2 * (1 + 4 * y ** 2) * torch.sin(math.pi * x) * torch.sin(math.pi * y ** 2)
            + 2 * math.pi * torch.sin(math.pi * x) * torch.cos(math.pi * y ** 2))


def sample_domain(n_interior=10000, n_boundary=800, seed=0, dtype=torch.float64, device="cpu"):
    """Uniform interior points in (-1,1)^2 and n_boundary points spread evenly over the four sides."""
    rng = np.random.default_rng([seed, 5])
    zi = rng.uniform(-1, 1, (n_interior, 2))
    per = n_boundary // 4
    t = rng.uniform(-1, 1, (4, per))
    zb = np.concatenate([np.stack([t[0], -np.ones(per)], 1), np.stack([t[1], np.ones(per)], 1),
                         np.stack([-np.ones(per), t[2]], 1), np.stack([np.ones(per), t[3]], 1)])
    return (torch.tensor(zi, dtype=dtype, device=device), torch.tensor(zb, dtype=dtype, device=device))


def pde_loss(model, z_in, z_b, alpha=0.01):
    """alpha * mean |u_xx + u_yy - f|^2 (interior) + mean u^2 (boundary)   (paper Sec. 3.4)."""
    z = z_in.clone().requires_grad_(True)
    u = model(z).squeeze(1)
    g = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    u_xx = torch.autograd.grad(g[:, 0].sum(), z, create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(g[:, 1].sum(), z, create_graph=True)[0][:, 1]
    loss_i = torch.mean((u_xx + u_yy - f_rhs(z_in)) ** 2)
    loss_b = torch.mean(model(z_b).squeeze(1) ** 2)
    return alpha * loss_i + loss_b, loss_i.detach(), loss_b.detach()


@torch.no_grad()
def pde_errors(model, n=101, dtype=torch.float64, device="cpu"):
    """Error against the exact solution on a uniform n x n test grid: MSE and relative L2 norm."""
    g = torch.linspace(-1, 1, n, dtype=dtype, device=device)
    X, Y = torch.meshgrid(g, g, indexing="ij")
    z = torch.stack([X.reshape(-1), Y.reshape(-1)], 1)
    err = model(z).squeeze(1) - u_true(z)
    return dict(mse=float(torch.mean(err ** 2)), rel_l2=float(torch.linalg.norm(err) / torch.linalg.norm(u_true(z))))


def fit_pinn(model, z_in, z_b, steps=100, alpha=0.01, max_iter=20, history_size=10):
    """LBFGS on the PINN loss. No checkpoint selection: the final iterate is reported (the exact solution is
    never used for training or selection, only for the final error table)."""
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter, history_size=history_size,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)

    def closure():
        opt.zero_grad()
        loss, _, _ = pde_loss(model, z_in, z_b, alpha)
        loss.backward()
        return loss
    curve = []
    for _ in range(steps):
        curve.append(float(opt.step(closure).detach()))
    return curve


def run_pde_kan(width=(2, 10, 1), grids=(3, 5, 10, 20), steps_per_grid=100, k=3, ni=10000, nb=800, alpha=0.01,
                seed=0, dtype=torch.float64, device="cpu", log=print):
    """KAN trained with grid extension; returns one row per grid size (error vs exact solution, parameters)."""
    torch.manual_seed(seed)
    z_in, z_b = sample_domain(ni, nb, seed, dtype, device)
    rows, model = [], None
    for G in grids:
        t0 = time.time()
        if model is None:
            model = KAN(list(width), G=G, k=k).to(device=device, dtype=dtype)
            model.update_grid(z_in)
        else:
            model.update_grid(z_in, G_new=G)
        curve = fit_pinn(model, z_in, z_b, steps=steps_per_grid, alpha=alpha)
        err = pde_errors(model, dtype=dtype, device=device)
        rows.append(dict(model=f"KAN{list(width)}", G=G, params=model.n_params(), final_loss=curve[-1],
                         seconds=time.time() - t0, **err))
        log(f"  KAN G={G}: params={model.n_params()} rel L2={err['rel_l2']:.2e} MSE={err['mse']:.2e}")
    return rows, model


def run_pde_mlp(n_params, depth, steps=400, ni=10000, nb=800, alpha=0.01, seed=0, dtype=torch.float64, device="cpu",
                return_model=False):
    """Equal-parameter MLP baseline (SiLU, smooth enough for second derivatives) with the same optimiser and
    the same total number of LBFGS steps as the whole KAN schedule."""
    torch.manual_seed(seed)
    z_in, z_b = sample_domain(ni, nb, seed, dtype, device)
    mlp, gap = matched_mlp(n_params, 2, depth)
    mlp = mlp.to(device=device, dtype=dtype)
    t0 = time.time()
    curve = fit_pinn(mlp, z_in, z_b, steps=steps, alpha=alpha)
    err = pde_errors(mlp, dtype=dtype, device=device)
    row = dict(model=f"MLP(depth {depth})", G=None, params=mlp.n_params(), param_gap=gap, final_loss=curve[-1],
               seconds=time.time() - t0, **err)
    return (row, mlp) if return_model else row
