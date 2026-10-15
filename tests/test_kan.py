"""Unit tests for the from-scratch KAN. Run: pytest -q"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
import torch
from scipy.interpolate import BSpline
from src.kan import KAN, KANLayer, b_spline_basis, extend_grid, fit_spline_coefficients
from src.data import make_dataset
from src.train import fit_lbfgs

DT = torch.float64


def _grid(n_in=2, G=5, k=3):
    return extend_grid(torch.linspace(-1, 1, G + 1, dtype=DT).repeat(n_in, 1), k)


def test_basis_shape_partition_of_unity_nonneg():
    G, k = 5, 3
    x = torch.rand(200, 2, dtype=DT) * 2 - 1
    B = b_spline_basis(x, _grid(2, G, k), k)
    assert B.shape == (200, 2, G + k)
    assert torch.all(B >= -1e-12)
    assert torch.allclose(B.sum(-1), torch.ones(200, 2, dtype=DT), atol=1e-12)  # inside [t0,tG]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_basis_matches_scipy_bspline(k):
    """Defining component: our Cox-de Boor basis equals SciPy's reference B-splines."""
    G = 6
    ext = _grid(1, G, k)
    x = torch.linspace(-0.99, 0.99, 101, dtype=DT).unsqueeze(1)
    ours = b_spline_basis(x, ext, k)[:, 0, :].numpy()
    t = ext[0].numpy()
    for i in range(G + k):
        c = np.zeros(G + k); c[i] = 1
        ref = BSpline(t, c, k, extrapolate=False)(x[:, 0].numpy())
        assert np.allclose(ours[:, i], np.nan_to_num(ref), atol=1e-12)


def test_layer_shapes_params_and_forward_finite():
    layer = KANLayer(3, 4, G=5, k=3).double()
    names = {n for n, _ in layer.named_parameters()}
    assert names == {"coef", "w_spline", "w_base"}
    assert layer.coef.shape == (3, 4, 8) and layer.grid.shape == (3, 12)
    y = layer(torch.rand(10, 3, dtype=DT) * 2 - 1)
    assert y.shape == (10, 4) and torch.isfinite(y).all()


def test_kan_param_count_and_shapes():
    m = KAN([2, 5, 1], G=5, k=3).double()
    assert m(torch.rand(7, 2, dtype=DT)).shape == (7, 1)
    assert m.n_params() == (2 * 5 + 5 * 1) * (5 + 3 + 2)       # edges * (G+k coef + w_b + w_s)


def test_layer_equals_eq_2_10_by_hand():
    layer = KANLayer(2, 1, G=4, k=3).double()
    x = torch.rand(5, 2, dtype=DT) * 2 - 1
    B = b_spline_basis(x, layer.grid, 3)
    manual = 0
    for i in range(2):
        spl = (B[:, i, :] * layer.coef[i, 0]).sum(-1)
        manual = manual + layer.w_base[i, 0] * torch.nn.functional.silu(x[:, i]) + layer.w_spline[i, 0] * spl
    assert torch.allclose(layer(x).squeeze(1), manual, atol=1e-12)


def test_gradients_finite_and_optimizer_step_changes_params():
    m = KAN([2, 3, 1], G=5).double()
    x, y = torch.rand(32, 2, dtype=DT) * 2 - 1, torch.rand(32, 1, dtype=DT)
    loss = torch.nn.functional.mse_loss(m(x), y)
    loss.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    before = [p.detach().clone() for p in m.parameters()]
    torch.optim.SGD(m.parameters(), lr=0.1).step()
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))


def test_lstsq_recovers_spline_exactly():
    ext = _grid(1, 6, 3)
    c = torch.randn(1, 1, 9, dtype=DT)
    x = torch.rand(400, 1, dtype=DT) * 1.98 - 0.99
    y = torch.einsum("bic,ioc->bio", b_spline_basis(x, ext, 3), c)
    assert torch.allclose(fit_spline_coefficients(x, y, ext, 3), c, atol=1e-8)


def test_grid_extension_preserves_function():
    """Eq. 2.16: after refitting on a finer grid the function is (nearly) unchanged."""
    torch.manual_seed(0)
    layer = KANLayer(2, 2, G=5, k=3).double()
    x = torch.rand(2000, 2, dtype=DT) * 2 - 1
    layer.update_grid(x)                                       # same-G update first
    before = layer.spline(x).detach()
    layer.update_grid(x, G_new=20)
    assert layer.G == 20 and layer.coef.shape == (2, 2, 23)
    assert (layer.spline(x) - before).abs().max() < 1e-3


def test_cpu_gpu_consistency_when_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    m = KAN([2, 2, 1], G=5).double(); x = torch.rand(8, 2, dtype=DT)
    assert torch.allclose(m(x), m.cuda()(x.cuda()).cpu(), atol=1e-10)


def test_overfits_tiny_dataset_and_trains_on_target():
    torch.manual_seed(1)
    d = make_dataset("xy", n_train=200, n_val=100, n_test=100, seed=0)
    m = KAN([2, 2, 1], G=5).double()
    m.update_grid(d["train"][0])
    c = fit_lbfgs(m, d["train"], d["val"], steps=30, grid_update_steps=(0, 5))
    assert c["train"][-1] < 0.1 * c["train"][0] and c["train"][-1] < 0.02


def test_edge_function_matches_forward_for_single_input():
    layer = KANLayer(1, 1, G=5, k=3).double()
    x = torch.linspace(-0.9, 0.9, 25, dtype=DT)
    assert torch.allclose(layer.edge_function(0, 0, x), layer(x.unsqueeze(1)).squeeze(1), atol=1e-12)


# ---------------------------------------------------------------- simplification, PDE, MNIST helpers
from src.symbolic import fit_affine, fix_symbolic, auto_symbolic, symbolic_formula, fine_tune_affine
from src.models import matched_mlp, mlp_param_count, MLP


def test_regulariser_matches_definition_and_gradients_flow():
    m = KAN([2, 3, 1], G=5).double()
    x = torch.rand(64, 2, dtype=DT) * 2 - 1
    out, reg = m.forward_reg(x)
    assert torch.allclose(out, m(x))
    manual = 0
    h = x
    for layer in m.layers:                                   # Eq. 2.17-2.20 computed by hand
        l1 = layer.post_activations(h).abs().mean(0)
        p = l1 / l1.sum()
        manual = manual + l1.sum() - (p * torch.log(p)).sum()
        h = layer(h)
    assert torch.allclose(reg, manual, atol=1e-9)
    reg.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_prune_masks_nodes_and_changes_effective_params():
    m = KAN([2, 4, 1], G=5).double()
    x = torch.rand(100, 2, dtype=DT) * 2 - 1
    full = m.effective_params()
    assert full == m.n_params()                              # no pruning yet: all params are live
    with torch.no_grad():
        m.layers[0].w_base.zero_(); m.layers[0].w_spline.zero_()      # make layer-0 edges of node 3 dead
        m.layers[0].w_base[:, :3] = 1.0
    shape = m.prune(x, theta=1e-2)
    assert shape == [2, 3, 1]
    assert m.layers[0].mask[:, 3].sum() == 0 and m.layers[1].mask[3, :].sum() == 0
    assert m.effective_params() < full
    assert torch.isfinite(m(x)).all()


def test_fit_affine_recovers_parameters():
    x = torch.linspace(-1, 1, 500, dtype=DT)
    y = 2.0 * torch.sin(3.0 * x + 0.5) + 1.0
    a, b, c, d, r2 = fit_affine(x, y, "sin")
    assert r2 > 0.9999
    # (a, b, c) are only identifiable up to symmetries such as (a,b,c)->(-a,pi-b,c), so test the reconstructed function
    assert (c * torch.sin(a * x + b) + d - y).abs().max() < 1e-3
    assert abs(abs(a) - 3.0) < 0.05 and abs(d - 1.0) < 0.05
    assert abs(fit_affine(x, x ** 2, "x^2")[4] - 1) < 1e-9


def test_symbolification_end_to_end_on_known_function():
    """[1,1] KAN trained to fit 2*exp(x)+1 -> symbolify with 'exp' -> closed-form expression matches the data."""
    torch.manual_seed(0)
    x = torch.rand(400, 1, dtype=DT) * 2 - 1
    y = 2 * torch.exp(x) + 1
    m = KAN([1, 1], G=8).double(); m.update_grid(x)
    fit_lbfgs(m, (x, y), (x, y), steps=30)
    from src.symbolic import fix_symbolic as fs
    r2 = fs(m, 0, 0, 0, "exp", x)
    assert r2 > 0.999
    fine_tune_affine(m, (x, y), steps=10)
    expr = symbolic_formula(m, ["x"])[0]
    from src.symbolic import formula_predict
    pred = torch.tensor(formula_predict(expr, ["x"], x), dtype=DT).reshape(-1, 1)
    assert torch.sqrt(torch.mean((pred - y) ** 2)) < 1e-3


def test_matched_mlp_param_counts():
    for target, n_in, depth, n_out in [(127040, 784, 2, 10), (300, 2, 3, 1), (60, 2, 4, 1)]:
        mlp, gap = matched_mlp(target, n_in, depth, n_out)
        assert mlp.n_params() == mlp_param_count(n_in, mlp.net[0].out_features, depth, n_out)
        assert abs(gap) < 0.25 and (abs(gap) < 0.02 or target < 100)


def test_pde_residual_of_exact_solution_is_zero():
    """The analytic u(x,y)=sin(pi x) sin(pi y^2) must satisfy u_xx+u_yy=f (checks our f and the autograd derivatives)."""
    from src.pde import u_true, f_rhs

    class Exact(torch.nn.Module):
        def forward(self, z):
            return u_true(z).unsqueeze(1)
    z = (torch.rand(200, 2, dtype=DT) * 2 - 1).requires_grad_(True)
    u = Exact()(z).squeeze(1)
    g = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
    uxx = torch.autograd.grad(g[:, 0].sum(), z, create_graph=True)[0][:, 0]
    uyy = torch.autograd.grad(g[:, 1].sum(), z, create_graph=True)[0][:, 1]
    assert torch.allclose(uxx + uyy, f_rhs(z.detach()), atol=1e-8)


def test_pde_loss_second_derivatives_through_kan_are_finite():
    from src.pde import pde_loss, sample_domain
    m = KAN([2, 4, 1], G=5).double()
    zi, zb = sample_domain(64, 16)
    loss, _, _ = pde_loss(m, zi, zb)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_mnist_pipeline_runs_on_synthetic_data():
    from src.mnist import load_mnist, build_pair, train_classifier
    d = load_mnist(synthetic=True)
    kan, mlp, _ = build_pair([784, 4, 10], 3, 3, 2)
    r, _ = train_classifier(kan, d, epochs=1, log=lambda s: None)
    r2, _ = train_classifier(mlp, d, epochs=1, log=lambda s: None)
    assert 0 <= r["test_acc"] <= 1 and 0 <= r2["test_acc"] <= 1 and r["params"] > 3000
