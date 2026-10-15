"""MNIST: KAN vs equal-parameter MLP classifier (Adam training, GPU friendly)."""

# %% [mnist]
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from .kan import KAN                                             # nb:skip
from .models import MLP, matched_mlp                             # nb:skip


def load_mnist(root="data", n_val=5000, device="cpu", synthetic=False, seed=0):
    """Return dict(train/val/test -> (X, y)); X is flattened to 784 and scaled to [-1, 1] (the KAN grid range).
    Downloads MNIST through torchvision. synthetic=True builds a small MNIST-shaped *fake* dataset (class prototypes
    plus noise) and exists only so the code path can be smoke-tested offline; never report its numbers."""
    if synthetic:
        g = torch.Generator().manual_seed(seed)
        protos = torch.rand(10, 784, generator=g)

        def make(n):
            y = torch.randint(0, 10, (n,), generator=g)
            return (protos[y] + 0.3 * torch.randn(n, 784, generator=g)).clamp(0, 1) * 2 - 1, y
        (xt, yt), (xs, ys) = make(4000), make(1000)
    else:
        from torchvision import datasets
        tr, te = datasets.MNIST(root, train=True, download=True), datasets.MNIST(root, train=False, download=True)
        xt, yt = tr.data.float().div(255).reshape(-1, 784) * 2 - 1, tr.targets
        xs, ys = te.data.float().div(255).reshape(-1, 784) * 2 - 1, te.targets
    n_val = min(n_val, len(xt) // 5)                           # (only matters for the tiny synthetic set)
    perm = torch.randperm(len(xt), generator=torch.Generator().manual_seed(seed))
    vi, ti = perm[:n_val], perm[n_val:]
    sets = {"train": (xt[ti], yt[ti]), "val": (xt[vi], yt[vi]), "test": (xs, ys)}
    return {k: (x.to(device=device, dtype=torch.float32), y.to(device)) for k, (x, y) in sets.items()}


@torch.no_grad()
def update_hidden_grids(model, xb):
    """Re-place the knots of every KAN layer after the first from the current hidden activations (paper Sec. 2.2,
    'update grids on the fly'). The input layer keeps its fixed uniform grid on [-1,1] (pixels are bounded);
    hidden grids are uniform over the observed range (grid_eps=1) because the data quantiles are degenerate."""
    h = model.layers[0](xb)
    for layer in model.layers[1:]:
        layer.update_grid(h, grid_eps=1.0)
        h = layer(h)


@torch.no_grad()
def accuracy(model, X, y, batch=2000):
    model.eval()
    correct = sum((model(X[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item() for i in range(0, len(X), batch))
    return correct / len(X)


def train_classifier(model, data, epochs=10, lr=1e-3, weight_decay=1e-4, batch_size=256, grid_update_epochs=3, seed=0, log=print):
    """AdamW + cross-entropy. The checkpoint with best *validation* accuracy is restored and then evaluated once on
    the test set. For KANs the hidden grids are refreshed at the start of the first `grid_update_epochs` epochs."""
    torch.manual_seed(seed)
    (xt, yt), (xv, yv), (xs, ys) = data["train"], data["val"], data["test"]
    is_kan = isinstance(model, KAN)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if is_kan:
        update_hidden_grids(model, xt[:2000])
    best, best_state, hist, t0 = -1.0, copy.deepcopy(model.state_dict()), [], time.time()
    for ep in range(epochs):
        if is_kan and 0 < ep < grid_update_epochs:
            update_hidden_grids(model, xt[:2000])
        model.train()
        perm = torch.randperm(len(xt), device=xt.device)
        total = 0.0
        for i in range(0, len(xt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        va = accuracy(model, xv, yv)
        hist.append(dict(epoch=ep + 1, train_loss=total / len(xt), val_acc=va, seconds=time.time() - t0))
        log(f"    epoch {ep + 1}/{epochs}: loss {total / len(xt):.4f}  val acc {va:.4f}")
        if va > best:
            best, best_state = va, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return dict(params=sum(p.numel() for p in model.parameters()), val_acc=best, test_acc=accuracy(model, xs, ys),
                seconds=time.time() - t0), hist


def build_pair(kan_width, G, k, depth, n_in=784, n_out=10, device="cpu", dtype=torch.float32):
    """A KAN and an MLP with (almost) the same number of parameters."""
    kan = KAN(list(kan_width), G=G, k=k).to(device=device, dtype=dtype)
    mlp, gap = matched_mlp(kan.n_params(), n_in, depth, n_out)
    return kan, mlp.to(device=device, dtype=dtype), gap
