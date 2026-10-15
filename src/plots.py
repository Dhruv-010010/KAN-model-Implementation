"""Reusable plotting helpers (all figures are drawn from saved result tables)."""

# %% [plots]
import numpy as np
import matplotlib.pyplot as plt


def loglog_reference(ax, xs, y0, alpha, label, **kw):
    """Draw y = y0 * (x/x0)^(-alpha) through (xs[0], y0)."""
    xs = np.asarray(xs, float)
    ax.plot(xs, y0 * (xs / xs[0]) ** (-alpha), "--", lw=1, label=label, **kw)


def plot_curves(curves, boundaries, title):
    """Staircase training curves (train / val RMSE per LBFGS step) with grid-extension boundaries."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.semilogy(curves["train"], label="train RMSE")
    ax.semilogy(curves["val"], label="validation RMSE")
    for b in boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("LBFGS step (dotted lines: grid extension)"); ax.set_ylabel("RMSE"); ax.set_title(title); ax.legend()
    fig.tight_layout()
    return fig


# %% [plotkan]
import torch


@torch.no_grad()
def plot_kan_edges(model, x, var_names=None, title=""):
    """Draw every edge function phi_{j,i} of every layer over the range of its inputs (paper Sec. 2.5.1 'Visualization':
    transparency ~ tanh(3 * |phi|_1), so unimportant edges fade out; pruned edges are blank; symbolic edges are
    titled with their function)."""
    xs, h = [], x
    for layer in model.layers:
        xs.append(h); h = layer(h)
    figs = []
    for l, layer in enumerate(model.layers):
        pre, post = xs[l], layer.post_activations(xs[l])
        fig, axes = plt.subplots(layer.n_in, layer.n_out, figsize=(1.9 * layer.n_out + 0.6, 1.5 * layer.n_in + 0.6), squeeze=False)
        for i in range(layer.n_in):
            order = torch.argsort(pre[:, i])
            for j in range(layer.n_out):
                ax = axes[i][j]; ax.set_xticks([]); ax.set_yticks([])
                if layer.mask[i, j] <= 0:
                    ax.axis("off"); continue
                a = float(torch.tanh(3 * post[:, i, j].abs().mean()))
                ax.plot(pre[order, i].cpu(), post[order, i, j].cpu(), color="tab:blue", alpha=max(a, 0.15))
                if (i, j) in layer.sym_edges:
                    ax.set_title(layer.sym_edges[(i, j)], fontsize=7, color="tab:red")
        name = f"layer {l}" + (f" - {title}" if title else "")
        fig.suptitle(name, fontsize=9); fig.tight_layout(); figs.append(fig)
    return figs
