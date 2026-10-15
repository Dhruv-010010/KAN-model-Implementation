"""Core Kolmogorov-Arnold Network (KAN) implementation (Liu et al., arXiv:2404.19756v5).

Written from scratch in PyTorch: B-spline basis (Cox-de Boor), the KAN layer
phi(x) = w_b*silu(x) + w_s*sum_i c_i B_i(x)   (paper Eq. 2.10-2.12),
layer composition (Eq. 2.5-2.7), grid update / grid extension by least squares (Eq. 2.16).
No third-party KAN package is used.
"""

# %% [imports]
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# %% [bspline]
def extend_grid(grid, k):
    """Pad a (n_in, G+1) grid with k extra knots on each side, spaced by the mean interval.

    The paper (Sec. 2.4) augments {t_0..t_G} to {t_-k..t_{G+k}}; the padding rule is not
    specified, so we use uniform spacing h = (t_G - t_0) / G.
    """
    G = grid.shape[1] - 1
    h = (grid[:, -1:] - grid[:, :1]) / G
    steps = torch.arange(1, k + 1, dtype=grid.dtype, device=grid.device)
    left = grid[:, :1] - h * steps.flip(0)
    right = grid[:, -1:] + h * steps
    return torch.cat([left, grid, right], dim=1)


def b_spline_basis(x, ext_grid, k):
    """Evaluate all B-spline basis functions B_{i,k}(x) with the Cox-de Boor recursion.

    x:        (batch, n_in) inputs
    ext_grid: (n_in, G + 2k + 1) extended knot vector (strictly increasing per row)
    returns:  (batch, n_in, G + k); B_i is non-zero only on [t_i, t_{i+k+1}).

    B_{i,0}(x) = 1 if t_i <= x < t_{i+1} else 0
    B_{i,p}(x) = (x - t_i)/(t_{i+p} - t_i) B_{i,p-1}(x) + (t_{i+p+1} - x)/(t_{i+p+1} - t_{i+1}) B_{i+1,p-1}(x)
    """
    x = x.unsqueeze(-1)                      # (B, n_in, 1)
    t = ext_grid.unsqueeze(0)                # (1, n_in, n_knots)
    B = ((x >= t[..., :-1]) & (x < t[..., 1:])).to(x.dtype)  # degree 0
    for p in range(1, k + 1):
        left = (x - t[..., : -(p + 1)]) / (t[..., p:-1] - t[..., : -(p + 1)]) * B[..., :-1]
        right = (t[..., p + 1:] - x) / (t[..., p + 1:] - t[..., 1:-p]) * B[..., 1:]
        B = left + right
    return B


# %% [fit]
def fit_spline_coefficients(x, y, ext_grid, k):
    """Least-squares B-spline coefficients c such that sum_i c_i B_i(x) ~= y   (paper Eq. 2.16).

    x: (batch, n_in); y: (batch, n_in, n_out); returns (n_in, n_out, G + k).
    Solved on CPU because the system can be under-determined (batch < G + k) and the
    rank-revealing LAPACK driver is not available on CUDA.
    """
    A = b_spline_basis(x, ext_grid, k).permute(1, 0, 2).cpu()    # (n_in, B, G+k)
    Y = y.permute(1, 0, 2).cpu()                                 # (n_in, B, n_out)
    sol = torch.linalg.lstsq(A, Y, driver="gelsd").solution      # (n_in, G+k, n_out); min-norm if rank-deficient
    return sol.permute(0, 2, 1).to(x.device)


# %% [symlib]
# Library of candidate symbolic functions for "symbolification" (paper Sec. 2.5.1, item 4).
# name -> (torch function, sympy function builder, complexity rank used to break ties)
import sympy as sp

SYM_LIB = {
    "x":    (lambda t: t,                    lambda t: t,          1),
    "x^2":  (lambda t: t ** 2,               lambda t: t ** 2,     2),
    "x^3":  (lambda t: t ** 3,               lambda t: t ** 3,     3),
    "x^4":  (lambda t: t ** 4,               lambda t: t ** 4,     4),
    "1/x":  (lambda t: 1.0 / t,              lambda t: 1 / t,      3),
    "sqrt": (lambda t: torch.sqrt(t),        lambda t: sp.sqrt(t), 3),
    "abs":  (lambda t: torch.abs(t),         lambda t: sp.Abs(t),  3),
    "exp":  (lambda t: torch.exp(t),         lambda t: sp.exp(t),  4),
    "log":  (lambda t: torch.log(t),         lambda t: sp.log(t),  4),
    "sin":  (lambda t: torch.sin(t),         lambda t: sp.sin(t),  4),
    "cos":  (lambda t: torch.cos(t),         lambda t: sp.cos(t),  4),
    "tanh": (lambda t: torch.tanh(t),        lambda t: sp.tanh(t), 5),
}


# %% [layer]
class KANLayer(nn.Module):
    """One KAN layer = a matrix of learnable 1-D functions phi_{j,i} (paper Eq. 2.2, 2.5).

    out_j(x) = sum_i phi_{j,i}(x_i),   phi(x) = w_b * silu(x) + w_s * sum_c coef_c B_c(x).
    """

    def __init__(self, n_in, n_out, G=5, k=3, grid_range=(-1.0, 1.0),
                 noise_scale=0.1, grid_eps=0.02):
        super().__init__()
        self.n_in, self.n_out, self.G, self.k, self.grid_eps = n_in, n_out, G, k, grid_eps
        grid = torch.linspace(grid_range[0], grid_range[1], G + 1).repeat(n_in, 1)
        self.register_buffer("grid", extend_grid(grid, k))       # (n_in, G+2k+1): not trained by SGD
        # Paper footnote 2: spline coefficients c_i ~ N(0, sigma^2), sigma = 0.1 -> spline(x) ~ 0
        self.coef = nn.Parameter(torch.randn(n_in, n_out, G + k) * noise_scale)
        # Paper: w_s = 1 at init; w_b ~ Xavier initialisation
        self.w_spline = nn.Parameter(torch.ones(n_in, n_out))
        w_b = torch.empty(n_in, n_out)
        nn.init.xavier_uniform_(w_b)
        self.w_base = nn.Parameter(w_b)
        # --- simplification state (paper Sec. 2.5.1); inactive by default, so plain KAN behaviour is unchanged ---
        self.register_buffer("mask", torch.ones(n_in, n_out))        # 0 = pruned edge
        self.register_buffer("sym_coef", torch.zeros(n_in, n_out, 4))  # affine (a,b,c,d) of symbolic edges
        self.pruned = False
        self.sym_edges = {}                                           # {(i, j): name in SYM_LIB}

    def spline(self, x):
        """Unweighted spline part: sum_c coef_c B_c(x), shape (batch, n_in, n_out)."""
        B = b_spline_basis(x, self.grid, self.k)                  # (batch, n_in, G+k)
        return torch.einsum("bic,ioc->bio", B, self.coef)

    def post_activations(self, x):
        """phi_{j,i}(x_i) for every edge, shape (batch, n_in, n_out) (paper: 'post-activations')."""
        post = self.w_base * F.silu(x).unsqueeze(-1) + self.w_spline * self.spline(x)  # Eq. 2.10
        if self.sym_edges:                                        # edges fixed to c*f(a*x+b)+d
            post = post.clone()
            for (i, j), name in self.sym_edges.items():
                a, b, c, d = self.sym_coef[i, j]
                post[:, i, j] = c * SYM_LIB[name][0](a * x[:, i] + b) + d
        if self.pruned:
            post = post * self.mask
        return post

    def forward(self, x):
        return self.post_activations(x).sum(dim=1)                # Eq. 2.5: node = sum of incoming edges

    def edge_l1(self, x):
        """|phi_{j,i}|_1 = mean over samples of |phi(x)|, shape (n_in, n_out)   (paper Eq. 2.17)."""
        return self.post_activations(x).abs().mean(dim=0)

    @torch.no_grad()
    def edge_function(self, i, j, x1d):
        """The learned 1-D function phi_{j,i} evaluated on x1d (n,), for plotting / interpretation."""
        B = b_spline_basis(x1d.unsqueeze(1), self.grid[i:i + 1], self.k)[:, 0, :]       # (n, G+k)
        return self.w_base[i, j] * F.silu(x1d) + self.w_spline[i, j] * (B @ self.coef[i, j])

    @torch.no_grad()
    def update_grid(self, x, G_new=None, grid_eps=None):
        """Re-place the knots from input samples and refit coefficients so the learned function is kept.

        With G_new > G this is the paper's *grid extension* (Sec. 2.4, Eq. 2.16): the new fine
        spline is the least-squares fit to the old coarse spline over x ~ p(x). With G_new = G it is
        the on-the-fly grid update of Sec. 2.2 (3). The exact knot-placement rule is not given in the
        paper; ours blends data quantiles (adaptive) with a uniform grid: eps*uniform + (1-eps)*quantile.
        """
        G_new = G_new or self.G
        eps = self.grid_eps if grid_eps is None else grid_eps       # 1.0 -> purely uniform knots (used for MNIST)
        y_old = self.spline(x)                                    # function to preserve, (B, n_in, n_out)
        xs, _ = torch.sort(x, dim=0)
        idx = torch.linspace(0, x.shape[0] - 1, G_new + 1, device=x.device).round().long()
        adaptive = xs[idx].T.clone()                              # (n_in, G_new+1) quantile knots
        span = (xs[-1] - xs[0]).clamp_min(1e-6)
        lo, hi = xs[0] - 0.01 * span, xs[-1] + 0.01 * span        # 1% margin: unseen inputs near the edge stay inside
        adaptive[:, 0], adaptive[:, -1] = lo, hi                  # outer knots = margin edges
        u = torch.linspace(0, 1, G_new + 1, dtype=x.dtype, device=x.device)
        uniform = lo[:, None] + (hi - lo)[:, None] * u[None, :]
        grid = eps * uniform + (1 - eps) * adaptive   # strictly increasing if eps > 0
        ext = extend_grid(grid, self.k)
        coef = fit_spline_coefficients(x, y_old, ext, self.k)
        self.G = G_new
        self.grid = ext
        if coef.shape == self.coef.shape:
            self.coef.data.copy_(coef)                                # same G: keep the Parameter object
        else:
            self.coef = nn.Parameter(coef)                            # new G: new Parameter (rebuild optimizer!)


# %% [network]
class KAN(nn.Module):
    """KAN(x) = (Phi_{L-1} o ... o Phi_0)(x), shape given by width = [n_0, ..., n_L] (paper Eq. 2.3, 2.7)."""

    def __init__(self, width, G=5, k=3, grid_range=(-1.0, 1.0), noise_scale=0.1, grid_eps=0.02):
        super().__init__()
        self.width, self.k = list(width), k
        self.layers = nn.ModuleList(
            [KANLayer(width[i], width[i + 1], G=G, k=k, grid_range=grid_range,
                      noise_scale=noise_scale, grid_eps=grid_eps) for i in range(len(width) - 1)])

    @property
    def G(self):
        return self.layers[0].G

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @torch.no_grad()
    def update_grid(self, x, G_new=None):
        """Update/extend the grid of every layer using that layer's own input activations."""
        for layer in self.layers:
            layer.update_grid(x, G_new)
            x = layer(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward_reg(self, x, mu1=1.0, mu2=1.0):
        """Forward pass that also returns the sparsity regulariser of paper Eq. 2.18-2.20:
        sum_l [ mu1*|Phi_l|_1 + mu2*S(Phi_l) ],  S = -sum_ij p_ij log p_ij,  p_ij = |phi_ij|_1 / |Phi_l|_1."""
        reg = x.new_zeros(())
        for layer in self.layers:
            post = layer.post_activations(x)
            l1 = post.abs().mean(dim=0)                           # (n_in, n_out), Eq. 2.17
            L1 = l1.sum()                                         # Eq. 2.18
            p = l1 / (L1 + 1e-12)
            entropy = -(p * torch.log(p + 1e-12)).sum()           # Eq. 2.19
            reg = reg + mu1 * L1 + mu2 * entropy
            x = post.sum(dim=1)
        return x, reg

    @torch.no_grad()
    def prune(self, x, theta=1e-2):
        """Node-level pruning (paper Eq. 2.21): hidden node i of layer l is kept only if both its
        incoming score I = max_k |phi_{l-1,i,k}|_1 and outgoing score O = max_j |phi_{l+1,j,i}|_1 exceed theta.
        Input and output nodes are never pruned. Returns the active shape [n_0, ..., n_L]."""
        scores = []
        h = x
        for layer in self.layers:
            scores.append(layer.edge_l1(h)); h = layer(h)
        for l in range(1, len(self.layers)):                      # hidden node layers
            incoming = scores[l - 1].max(dim=0).values            # per node of layer l: max over its input edges
            outgoing = scores[l].max(dim=1).values                # max over its output edges
            keep = (incoming > theta) & (outgoing > theta)
            self.layers[l - 1].mask[:, ~keep] = 0
            self.layers[l].mask[~keep, :] = 0
            self.layers[l - 1].pruned = self.layers[l].pruned = True
        return self.active_shape()

    def active_shape(self):
        """Number of nodes per layer that still have at least one live edge in and out."""
        shape = [self.layers[0].n_in]
        for l, layer in enumerate(self.layers):
            alive = layer.mask.sum(dim=0) > 0                     # nodes of the next layer with an incoming edge
            if l < len(self.layers) - 1:
                alive = alive & (self.layers[l + 1].mask.sum(dim=1) > 0)
            shape.append(int(alive.sum()))
        return shape

    def effective_params(self):
        """Parameters of live edges only: G+k+2 per spline edge, 4 (a,b,c,d) per symbolic edge."""
        total = 0
        for layer in self.layers:
            for i in range(layer.n_in):
                for j in range(layer.n_out):
                    if layer.mask[i, j] > 0:
                        total += 4 if (i, j) in layer.sym_edges else layer.G + layer.k + 2
        return total
