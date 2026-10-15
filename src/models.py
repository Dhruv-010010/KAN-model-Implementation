"""Baseline models: a plain MLP (paper Eq. 2.9) used for the KAN-vs-MLP comparison."""

# %% [mlp]
import torch.nn as nn


class MLP(nn.Module):
    """MLP(x) = (W_{L-1} o sigma o ... o sigma o W_0)(x); `width` = [n_0, ..., n_L] like the KAN shape."""

    def __init__(self, width, act=nn.SiLU):
        super().__init__()
        layers = []
        for i in range(len(width) - 1):
            layers.append(nn.Linear(width[i], width[i + 1]))
            if i < len(width) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# %% [match]
def mlp_param_count(n_in, width, depth, n_out=1):
    """Parameters of an MLP with `depth` linear layers and constant hidden width."""
    if depth == 1:
        return n_in * n_out + n_out
    return (n_in * width + width) + (depth - 2) * (width * width + width) + (width * n_out + n_out)


def matched_mlp(n_target, n_in, depth, n_out=1, act=nn.SiLU):
    """MLP whose parameter count is as close as possible to n_target (equal-parameter baseline).
    Exact equality is generally impossible for integer widths; the relative gap is returned."""
    best_w = min(range(1, 5000), key=lambda w: abs(mlp_param_count(n_in, w, depth, n_out) - n_target))
    model = MLP([n_in] + [best_w] * (depth - 1) + [n_out], act=act)
    gap = (model.n_params() - n_target) / n_target
    return model, gap
