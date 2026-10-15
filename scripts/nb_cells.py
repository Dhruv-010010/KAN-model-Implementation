# Cell definitions for the notebook (exec'd by build_notebook.py). Markdown text is written for a viva reader.
CFG_LITERAL = json.dumps(cfg, indent=2)
INTERP = json.load(open(f"{root}/scripts/nb_interp.json")) if os.path.exists(f"{root}/scripts/nb_interp.json") else {}
def I(key, default="*(interpretation is written after the results are inspected)*"):
    return INTERP.get(key, default)

cells = []
def md(t): cells.append(MD(t.strip("\n")))
def code(t): cells.append(CODE(t.strip("\n")))

# ---------------------------------------------------------------- 1. intro
md("""
# Kolmogorov–Arnold Networks (KAN): from-scratch implementation and reduced-scale reproduction
**Deep Learning 2026 — Paper Implementation Project**

**Paper:** Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljačić, T. Y. Hou, M. Tegmark, *KAN: Kolmogorov–Arnold Networks*, arXiv:2404.19756v5 [cs.LG], 9 Feb 2025.

**Research problem.** MLPs put fixed activations on nodes and learn linear weights on edges. KANs instead put a **learnable univariate function on every edge** (a B-spline) and only *sum* at the nodes, motivated by the Kolmogorov–Arnold representation theorem. The paper claims better accuracy per parameter and faster scaling laws on small function-fitting tasks.

**What this notebook does**
1. Implements the defining KAN components **from scratch in PyTorch** (Cox–de Boor B-splines, the residual spline edge function, layers, networks, grid update and *grid extension* by least squares). No KAN package is used.
2. Reproduces, at reduced scale, **Fig. 2.3 / Sec. 2.4** (grid extension, staircase loss curves, scaling of test RMSE with grid size G, interpolation threshold) and the $\\exp(\\sin(\\pi x)+y^2)$ panel of **Fig. 3.1** (KAN vs MLP at equal parameter count).
3. Runs two ablations: **grid extension vs training from scratch** and **spline order k**.
4. Analyses the learned edge functions, errors and failure cases.
5. **Assignment extensions (Parts B–D):** sparsification, pruning and **symbolic simplification** on Feynman equations, KAN vs **equal-parameter MLPs** on those symbolic-regression tasks, on a **Poisson PDE** (physics-informed loss) and on **MNIST**.

**Contents.** Part A (sections 1–14): implementation, reproduction, ablation, analysis — executed, outputs included. Part B (16–18): Feynman symbolic regression + simplification. Part C (19): PDE. Part D (20): MNIST. Then conclusions and viva preparation. **Parts B–D ship without outputs: they were smoke-tested at tiny scale but are meant to be run by you (GPU recommended for Part D; set `SMOKE = False`, already the default).**

**Honest scope.** This is a *reduced-scale* reproduction (grids up to G=100 instead of 1000, 100 instead of 200 LBFGS steps per grid, CPU-friendly, a few seeds). It is not an exact replication; every difference is listed in the configuration section.
""")
md("""
### Learning outcomes and limitations
After running this notebook you should be able to explain (i) how a KAN layer differs from an MLP layer, (ii) how B-spline bases are computed, (iii) why grid extension works and what it costs, and (iv) what the measured results do and do not say about the paper's claims.

Limitations that apply throughout: a single target function, a single dataset split, 4 seeds for the main KAN and for every MLP configuration (best seed chosen on validation data in both cases), and optimisation-limited (not capacity-limited) errors for the larger grids.
""")

# ---------------------------------------------------------------- 2. environment
md("## 1. Environment and setup\nOnly standard packages are needed (PyTorch, NumPy, SciPy, pandas, Matplotlib); SymPy and torchvision (for MNIST) are also used; all are pre-installed on Colab. The next cell installs nothing unless something is missing. Expected runtime: **~3 min** with the saved results shipped in `results/`; **~30–40 min on CPU** if the notebook has to recompute them (set `RECOMPUTE = True`).")
code("""
import importlib, subprocess, sys
for pkg in ["torch", "numpy", "scipy", "pandas", "matplotlib", "sympy", "torchvision"]:
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
""")
code("""
import platform, os, json, copy, time
import numpy, scipy, pandas as pd, matplotlib
import matplotlib.pyplot as plt
import torch
print("python", platform.python_version(), "| torch", torch.__version__, "| numpy", numpy.__version__,
      "| scipy", scipy.__version__, "| pandas", pd.__version__, "| matplotlib", matplotlib.__version__)
""")

# ---------------------------------------------------------------- 3. reproducibility
md("## 2. Reproducibility\nWe seed Python, NumPy and PyTorch, and use **float64** everywhere (the target errors go down to ~1e-5, which float32 would blur). CPU results are deterministic for a fixed thread count; on CUDA, reductions can be non-deterministic, so GPU numbers may differ slightly in the last digits. The KAN is tiny, so the CPU is perfectly adequate (a T4 is not needed).")
code(utils["utils"])
code("""
SEED = 0
set_seed(SEED)
device = get_device()
torch.set_default_dtype(torch.float64)
print("device:", device)
""")

# ---------------------------------------------------------------- 4. configuration
md("""
## 3. Configuration (single source of truth)
`CONFIG` is generated from `configs/experiment.yaml`. **Differences from the paper** (all flagged in `docs/paper_analysis.md`):

| Setting | Paper | Here | Why |
|---|---|---|---|
| Grid sizes | 3 … 1000 | 3, 5, 10, 20, 50, 100 | CPU time |
| LBFGS steps per grid | 200 | 100 | CPU time |
| Seeds | not stated | 4 (KAN [2,1,1] and each MLP), 2 (KAN [2,5,1]) | CPU time |
| Test set / val set | not stated | 10 000 / 1 000 (val used for model selection) | needed for honest selection |
| MLP baseline | depth 2–5, activation not stated | depth 2–4, SiLU, widths 5–50 | CPU time |
""")
code(f"CONFIG = {CFG_LITERAL}".replace("true", "True").replace("false", "False").replace("null", "None"))
code("""
OUT = "results"                      # saved metrics (CSV/JSON) live here
FIG = "figures"
CKPT = "checkpoints"
for d in (OUT, FIG, CKPT):
    os.makedirs(d, exist_ok=True)
# True -> re-run every experiment (~30-40 min on CPU); False -> reuse saved CSVs when present
RECOMPUTE = False
# True -> tiny smoke-test configuration (finishes in ~1 min; numbers are NOT meaningful)
SMOKE = False
if SMOKE:
    OUT = "results/smoke"; os.makedirs(OUT, exist_ok=True); RECOMPUTE = True
""")

# ---------------------------------------------------------------- 5. data
md("""
## 4. Data acquisition and inspection
The task is the paper's toy regression problem $f(x,y)=\\exp(\\sin(\\pi x)+y^2)$ (Sec. 2.4, Fig. 2.3; Sec. 3.1 example 2). It is known to be representable *exactly* by a $[2,1,1]$ KAN: first layer learns $\\sin(\\pi x)$ and $y^2$, the sum is $h$, the second layer learns $\\exp(h)$. Data are generated synthetically (no download). The paper does not state the input domain; we use $U[-1,1]^2$.
""")
code(data["data"])
code("""
dset = make_dataset(CONFIG["target"], CONFIG["data"]["n_train"], CONFIG["data"]["n_val"],
                    CONFIG["data"]["n_test"], seed=CONFIG["data"]["data_seed"])
for split, (x, y) in dset.items():
    print(f"{split:5s} x{tuple(x.shape)} y{tuple(y.shape)}  x in [{x.min():.3f},{x.max():.3f}]  "
          f"y mean {y.mean():.3f} std {y.std():.3f} range [{y.min():.3f},{y.max():.3f}]  NaN: {bool(torch.isnan(y).any())}")
""")
code("""
xt, yt = dset["train"]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
sc = ax[0].scatter(xt[:, 0], xt[:, 1], c=yt[:, 0], s=6, cmap="viridis"); plt.colorbar(sc, ax=ax[0])
ax[0].set_title("training inputs coloured by f(x,y)"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
ax[1].hist(yt[:, 0].numpy(), bins=40); ax[1].set_title("target distribution"); ax[1].set_xlabel("f")
gx = torch.linspace(-1, 1, 100); X, Y = torch.meshgrid(gx, gx, indexing="ij")
ax[2].contourf(X, Y, torch.exp(torch.sin(torch.pi * X) + Y ** 2), 30); ax[2].set_title("ground-truth surface")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 6. preprocessing
md("""
## 5. Preprocessing and splits
* Inputs are already in the KAN's initial grid range $[-1,1]$, so **no scaling is applied** and nothing is fitted on data, hence no preprocessing leakage.
* Train / validation / test are drawn from **independent random streams** (`default_rng([seed, split_id])`).
* Training: 1 000 samples (as in the paper). Validation: 1 000 samples, used for *model selection only* (best checkpoint within each grid stage, choice of seed, choice of the scaling-fit range). Test: 10 000 samples, used only for the final numbers and plots.
* LBFGS is a full-batch method, so the "data loader" is simply the three tensors.
""")
code("""
print("train/val/test sizes:", {k: len(v[0]) for k, v in dset.items()})
print("max |x_train - x_val| overlap check (should be > 0, i.e. different samples):",
      float((dset["train"][0][:5] - dset["val"][0][:5]).abs().max()))
""")

# ---------------------------------------------------------------- 7. math
md("""
## 6. Mathematical foundations
**KAN layer.** A layer with $n_{in}$ inputs and $n_{out}$ outputs is a matrix of univariate functions $\\Phi=\\{\\phi_{j,i}\\}$ (Eq. 2.2). Node $j$ of the next layer is simply the sum of incoming edge outputs (Eq. 2.5):
$$x_{l+1,j}=\\sum_{i=1}^{n_l}\\phi_{l,j,i}(x_{l,i}).$$
A KAN of shape $[n_0,\\dots,n_L]$ composes layers: $\\mathrm{KAN}(x)=(\\Phi_{L-1}\\circ\\cdots\\circ\\Phi_0)x$ (Eq. 2.7). An MLP instead alternates *linear* $W$ and *fixed* $\\sigma$ (Eq. 2.9).

**Edge function (Eq. 2.10–2.12).** $$\\phi(x)=w_b\\,\\mathrm{silu}(x)+w_s\\sum_{i}c_iB_i(x),\\qquad \\mathrm{silu}(x)=\\frac{x}{1+e^{-x}}.$$
$B_i$ are B-splines of order $k$ (we use $k=3$) on a grid of $G$ intervals, giving $G+k$ coefficients per edge. Trainable parameters per edge: $G+k$ coefficients $c_i$, plus $w_b$ and $w_s$. Initialisation: $w_s=1$, $c_i\\sim\\mathcal N(0,0.1^2)$ so the spline starts small, $w_b$ Xavier.

**B-spline basis (Cox–de Boor).** With knots $t_0\\le t_1\\le\\dots$:
$$B_{i,0}(x)=\\mathbb 1[t_i\\le x<t_{i+1}],\\qquad B_{i,p}(x)=\\frac{x-t_i}{t_{i+p}-t_i}B_{i,p-1}(x)+\\frac{t_{i+p+1}-x}{t_{i+p+1}-t_{i+1}}B_{i+1,p-1}(x).$$
Each $B_{i,k}$ is non-zero on only $k+1$ intervals, so the spline is *local*, and inside the grid the basis sums to 1.

**Grid extension (Eq. 2.16).** To move from a coarse grid $G_1$ to a fine grid $G_2$ without retraining from scratch, find $c'$ minimising $\\mathbb E_{x\\sim p(x)}\\big|\\sum_j c'_jB'_j(x)-\\sum_i c_iB_i(x)\\big|^2$, solved by linear least squares over the training inputs. The function is preserved; only its resolution grows.

**Theory (Thm 2.1).** For $(k{+}1)$-times differentiable edge functions, the error scales as $G^{-k-1}$, i.e. $\\propto G^{-4}$ for $k=3$ (the paper's "KAN theory" line in Fig. 3.1).

**Parameter count.** $\\approx N^2L(G+k)$ vs $N^2L$ for an MLP: KANs are only competitive because they need a much *smaller* $N$ (here $[2,1,1]$ suffices).
""")

# ---------------------------------------------------------------- 8. building blocks
md("## 7. Implementing the KAN building blocks\n### 7.1 B-spline basis\nThe knot vector is padded with $k$ extra knots on each side (uniform spacing; the paper does not specify the padding rule).")
code(kan["imports"])
code(kan["bspline"])
md("Check: our basis is non-negative, sums to one inside the grid, and has the expected shape. The plot shows the $G+k$ cubic basis functions for $G=5$.")
code("""
G_demo, k_demo = 5, 3
g = extend_grid(torch.linspace(-1, 1, G_demo + 1).repeat(1, 1), k_demo)
xx = torch.linspace(-0.999, 0.999, 400).unsqueeze(1)
Bd = b_spline_basis(xx, g, k_demo)[:, 0, :]
print("basis shape:", tuple(Bd.shape), "(expected (400,", G_demo + k_demo, "))")
print("min value:", float(Bd.min()), "| max |sum-1|:", float((Bd.sum(-1) - 1).abs().max()))
plt.figure(figsize=(6, 3)); plt.plot(xx[:, 0], Bd); plt.title(f"cubic B-spline basis, G={G_demo} (k={k_demo})"); plt.xlabel("x"); plt.show()
""")
md("### 7.2 Least-squares fitting of coefficients (Eq. 2.16)\nGiven samples $(x, y)$ this finds spline coefficients $c$ with $\\sum_i c_iB_i(x)\\approx y$. It is the engine behind both the grid update and grid extension.")
code(kan["fit"])
code("""
# test: a random spline is recovered exactly by least squares
ext = extend_grid(torch.linspace(-1, 1, 7).repeat(1, 1), 3)
c_true = torch.randn(1, 1, 9)
xs_ = torch.rand(400, 1) * 1.98 - 0.99
ys_ = torch.einsum("bic,ioc->bio", b_spline_basis(xs_, ext, 3), c_true)
print("max coefficient error:", float((fit_spline_coefficients(xs_, ys_, ext, 3) - c_true).abs().max()))
""")
md("### 7.2b Library of symbolic functions\nUsed later (Part B) to replace learned edge curves by closed-form functions. It is defined here because `KANLayer` can optionally hold *symbolic edges* and a *pruning mask* (both inactive by default, so a plain KAN behaves exactly as described above).")
code(kan["symlib"])
md("### 7.3 The KAN layer\n`forward` is Eq. 2.10 per edge followed by Eq. 2.5 (sum over incoming edges). `update_grid(x, G_new)` re-places the knots from the layer's own input samples and refits the coefficients so the function is preserved; with `G_new > G` this is the paper's grid extension. **Our design choice (not in the paper):** knots are `0.02·uniform + 0.98·data-quantiles`, with a 1 % margin so unseen inputs near the edge stay inside the grid.")
code(kan["layer"])
code("""
layer = KANLayer(3, 4, G=5, k=3)
x_t = torch.rand(10, 3) * 2 - 1
print("layer output:", tuple(layer(x_t).shape), "| parameters:", {n: tuple(p.shape) for n, p in layer.named_parameters()})
# gradient and numerical checks
layer(x_t).sum().backward()
print("all gradients finite:", all(torch.isfinite(p.grad).all().item() for p in layer.parameters()))
# grid extension must (almost) preserve the learned function
xe = torch.rand(2000, 3) * 2 - 1
layer.update_grid(xe); before = layer.spline(xe).clone()
layer.update_grid(xe, G_new=20)
print("G:", layer.G, "| max |f_fine - f_coarse| after extension:", float((layer.spline(xe) - before).abs().max()))
""")

# ---------------------------------------------------------------- 9. full model
md("## 8. The complete model\nA `KAN` is a list of `KANLayer`s (Eq. 2.7). Its `update_grid` updates each layer from *that layer's own input activations*, which is why the hidden layer's grid follows the range of the sums $\\phi_1(x)+\\phi_2(y)$.")
code(kan["network"])
code("""
kan_model = KAN(CONFIG["kan"]["width"], G=CONFIG["kan"]["grids"][0], k=CONFIG["kan"]["k"])
print(kan_model)
print("parameters:", kan_model.n_params(), "| formula edges*(G+k+2) =", 3 * (CONFIG["kan"]["grids"][0] + CONFIG["kan"]["k"] + 2))
print("output shape:", tuple(kan_model(dset["train"][0][:5]).shape), "| finite:", bool(torch.isfinite(kan_model(dset["train"][0])).all()))
""")
md("The $[2,1,1]$ shape mirrors the exact Kolmogorov–Arnold-style decomposition of the target (paper Sec. 2.4): layer 0 has two edges ($x\\to h$, $y\\to h$), layer 1 has one edge ($h\\to f$). The paper's rough count '15G' for a $[2,5,1]$ KAN corresponds to 15 edges·G; ours additionally counts $k$ extra coefficients and the two scale weights per edge.")

# ---------------------------------------------------------------- 10. baseline
md("## 9. The MLP baseline\nA plain MLP (Eq. 2.9) with SiLU activations. **Fairness:** same data, same LBFGS optimiser, same total step budget (600 steps = 6 grids × 100), model selection on the same validation set, and we compare *test RMSE at equal parameter count* (the paper's Pareto-frontier view). The MLP activation and widths are not given in Sec. 3.1, so depth (2–4) and width (5–50) are swept as a small, untuned grid; the MLP gets no hyper-parameter search beyond this sweep, but, like the KAN, it is trained with 4 seeds and the best seed is chosen on validation RMSE.")
code(models["mlp"])
md("**Equal-parameter baselines.** `matched_mlp` picks the integer hidden width whose parameter count is closest to a target (used in Parts B–D); the relative gap is always reported.")
code(models["match"])
code("""
for depth, w in [(2, 10), (3, 10), (4, 10)]:
    m = MLP([2] + [w] * (depth - 1) + [1])
    print(f"MLP depth {depth}, width {w}: {m.n_params()} parameters")
""")

# ---------------------------------------------------------------- 11. training
md("""
## 10. Training
Loss: mean squared error. Optimiser: **LBFGS** with strong-Wolfe line search (as in the paper). One *step* = one `optimizer.step(closure)` call (up to 20 inner iterations; the paper does not define a step). The best checkpoint **by validation RMSE** is kept within every grid stage, and the test set is never touched inside the training loop. The grid is re-placed from the activations at steps 0, 10, …, 40 of each stage ("update on the fly", Sec. 2.2), then training continues; between stages the grid is *extended* (G = 3 → 5 → 10 → 20 → 50 → 100) by least squares.
""")
code(evaluate["evaluate"])
code(train["train"])
md("### 10.1 Live training of one KAN (first seed in the config)\nThis cell trains the $[2,1,1]$ KAN through all grid stages and takes roughly 2–3 minutes on CPU.")
code("""
K = CONFIG["kan"]
SMOKE_K = CONFIG["smoke"]
grids_live = SMOKE_K["grids"] if SMOKE else K["grids"]
steps_live = SMOKE_K["steps_per_grid"] if SMOKE else K["steps_per_grid"]
live_seed = K["seeds"][0]
set_seed(live_seed)
t0 = time.time()
rec_live, curves_live, kan_live = train_kan_with_extension(
    K["width"], dset, grids=grids_live, steps_per_grid=steps_live, k=K["k"], extend=True,
    seed=live_seed, grid_update_steps=tuple(K["grid_update_steps"]))
print(f"trained in {time.time() - t0:.0f}s")
live_df = pd.DataFrame(rec_live); live_df
""")
code(plots["plots"])
code("""
bounds = [steps_live * i for i in range(1, len(grids_live))]
fig = plot_curves(curves_live, bounds, f"KAN {K['width']} (seed {live_seed}): staircase loss curves")
fig.savefig(f"{FIG}/live_training_curves.png", dpi=150); plt.show()
""")
md(I("live"))

# ---------------------------------------------------------------- 12. baseline training
md("## 11. Baseline training\nOne MLP (depth 3, width 20) trained live under the same conditions as a sanity check; the full MLP sweep is computed in section 12.")
code("""
set_seed(0)
mlp_live = MLP([2, 20, 20, 1])
rec_mlp, curves_mlp = train_mlp(mlp_live, dset, steps=CONFIG["mlp"]["steps"] if not SMOKE else 10)
print({k: (float(f"{v:.3g}") if isinstance(v, float) else v) for k, v in rec_mlp.items()})
""")

# ---------------------------------------------------------------- 13. reproduction
md("""
## 12. Evaluation and reproduction of the paper's result
The multi-seed experiment (4 seeds × 6 grid sizes for $[2,1,1]$, 2 seeds for $[2,5,1]$, and the MLP sweep with 4 seeds) is expensive, so it is run once by `run_main` and **saved as CSV/JSON**; every table and figure below is generated from those files (`RECOMPUTE=True` reruns it). Models are selected on validation data; test RMSE is only reported.
""")
code(exps["experiments"])
code("""
need = [f"{OUT}/kan_records.csv", f"{OUT}/mlp_records.csv"] if not SMOKE else [f"{OUT}/kan_records.csv"]
if RECOMPUTE or not all(os.path.exists(p) for p in need):
    print("computing main experiment ...")
    run_main(CONFIG, OUT, data=dset, smoke=SMOKE)
kan_df = pd.read_csv(f"{OUT}/kan_records.csv")
mlp_df = pd.read_csv(f"{OUT}/mlp_records.csv") if os.path.exists(f"{OUT}/mlp_records.csv") else pd.DataFrame()
print("loaded", len(kan_df), "KAN rows and", len(mlp_df), "MLP rows from", OUT)
""")
md("### 12.1 Per-seed results for the $[2,1,1]$ KAN\nOptimisation of KANs with LBFGS depends strongly on the random initialisation, so **all seeds are shown**. The 'selected' seed is the one with the lowest *final validation* RMSE (the paper does something similar for its Table 2: best over seeds); this is a model-selection step on validation data, not on test data.")
code("""
k211 = kan_df[kan_df.model == "KAN[2,1,1]"]
final = k211[k211.G == k211.G.max()].set_index("seed")[["train_rmse", "val_rmse", "test_rmse", "test_median_rmse"]]
final.columns = ["train RMSE", "val RMSE", "test RMSE", "test sqrt(median sq. err)"]
best_seed = int(final["val RMSE"].idxmin())
print("selected seed (lowest final validation RMSE):", best_seed)
final.style.format("{:.2e}")
""")
code("""
piv = k211.pivot(index="G", columns="seed", values="test_rmse")
piv.columns = [f"seed {c}" for c in piv.columns]
piv.style.format("{:.2e}")
""")
code("""
# same seed, different outcome? compare the live run of section 10.1 with the saved table (scripts ran with 1 CPU thread)
saved = k211[(k211.seed == live_seed) & (k211.G == k211.G.max())].test_rmse.iloc[0]
print(f"seed {live_seed}: live run in this notebook (torch threads = {torch.get_num_threads()}) test RMSE = {rec_live[-1]['test_rmse']:.2e}"
      f" | saved experiment run (1 thread) = {saved:.2e}")
""")
md(I("threads"))
md("### 12.2 Reproducing Fig. 2.3 (bottom-left): test RMSE vs grid size G\nThe paper reports that for the $[2,1,1]$ KAN test RMSE $\\propto G^{-3}$ (mean of squared errors) while $\\sqrt{\\mathrm{median}}$ scales closer to $G^{-4}$, theory predicting $G^{-4}$. We fit the exponent in log–log space over the grids **up to the validation-optimal G** (beyond it the error is optimisation-limited and the curve is flat, so a straight-line fit would be meaningless).")
code("""
def val_optimal_G(df):
    return int(df.loc[df.val_rmse.idxmin(), "G"])

def fit_range_exponent(df, col):
    # slope of log(err) vs log(G) over grids up to the validation-optimal G (at least the 3 smallest grids)
    df = df.sort_values("G").reset_index(drop=True)
    Gstar = val_optimal_G(df)
    n = max(3, int((df.G <= Gstar).sum()))
    sub = df.head(n)
    if len(sub) < 2:
        return float("nan"), Gstar
    return scaling_exponent(sub.G, sub[col]), Gstar

sel = k211[k211.seed == best_seed].sort_values("G")
rows = []
for seed, d in k211.groupby("seed"):
    d = d.sort_values("G")
    a_mean, Gs = fit_range_exponent(d, "test_rmse")
    a_med, _ = fit_range_exponent(d, "test_median_rmse")
    rows.append(dict(seed=seed, G_star_by_val=Gs, alpha_mean=a_mean, alpha_median=a_med))
alpha_df = pd.DataFrame(rows).set_index("seed")
alpha_df.style.format({"alpha_mean": "{:.2f}", "alpha_median": "{:.2f}"})
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for seed, d in k211.groupby("seed"):
    d = d.sort_values("G")
    ax[0].loglog(d.G, d.test_rmse, "o-", alpha=0.35, lw=1, label=f"seed {seed}")
ax[0].loglog(sel.G, sel.test_rmse, "ko-", lw=2.2, label=f"selected (seed {best_seed})")
ax[0].loglog(sel.G, sel.test_median_rmse, "s--", color="tab:red", lw=1.5, label="selected: sqrt(median sq. err)")
loglog_reference(ax[0], sel.G, sel.test_rmse.iloc[0], 3, "$G^{-3}$ (paper, observed)", color="gray")
loglog_reference(ax[0], sel.G, sel.test_rmse.iloc[0], 4, "$G^{-4}$ (theory, k=3)", color="black")
ax[0].set_xlabel("grid intervals G"); ax[0].set_ylabel("test RMSE"); ax[0].set_title("KAN [2,1,1]: test RMSE vs G"); ax[0].legend(fontsize=7)
ax[1].loglog(sel.G, sel.train_rmse, "o-", label="train"); ax[1].loglog(sel.G, sel.val_rmse, "s-", label="val"); ax[1].loglog(sel.G, sel.test_rmse, "^-", label="test")
ax[1].set_xlabel("G"); ax[1].set_title(f"selected seed {best_seed}: train/val/test"); ax[1].legend()
plt.tight_layout(); fig.savefig(f"{FIG}/scaling_vs_G.png", dpi=150); plt.show()
""")
md("### 12.3 Interpolation threshold for the $[2,5,1]$ KAN (Fig. 2.3, top-left)\nThe paper predicts the test-loss minimum (U-shape) near $G\\approx 1000/15\\approx 67$ and observes $G\\sim 50$. Our grid list is coarse near that value (…, 20, 50, 100), so we can only say which grid is validation-optimal.")
code("""
k251 = kan_df[kan_df.model == "KAN[2,5,1]"]
thr_rows = []
for seed, d in k251.groupby("seed"):
    thr_rows.append({"seed": seed, "val-optimal G": val_optimal_G(d), "test-optimal G": int(d.loc[d.test_rmse.idxmin(), "G"]),
                     "params at largest G": int(d[d.G == d.G.max()].params.iloc[0])})
thr = pd.DataFrame(thr_rows).set_index("seed")
print("paper: predicted G = 1000/15 = 67, observed G ~ 50 (n_train = 1000)")
thr
""")
md("### 12.4 Reproducing Fig. 3.1 (panel 2): KAN vs MLP at equal parameter count")
code("""
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for seed, d in k211.groupby("seed"):
    ax.loglog(d.sort_values("G").params, d.sort_values("G").test_rmse, "o-", color="tab:blue", alpha=0.25, lw=1)
ax.loglog(sel.params, sel.test_rmse, "o-", color="tab:blue", lw=2.2, label=f"KAN [2,1,1] (selected seed {best_seed})")
if len(mlp_df):
    mlp_sel = mlp_df.loc[mlp_df.groupby(["model", "width"]).val_rmse.idxmin()]   # best seed per config, chosen on validation RMSE
    for (name, d), c in zip(mlp_sel.groupby("model"), ["tab:orange", "tab:green", "tab:red"]):
        d = d.sort_values("params"); ax.loglog(d.params, d.test_rmse, "s-", color=c, label=name)
loglog_reference(ax, sel.params, sel.test_rmse.iloc[0], 4, "$N^{-4}$ (KAN theory)", color="black")
loglog_reference(ax, sel.params, sel.test_rmse.iloc[0], 2, "$N^{-2}$", color="gray")
ax.set_xlabel("number of parameters N"); ax.set_ylabel("test RMSE"); ax.set_title("f = exp(sin(pi x)+y^2): KAN vs MLP"); ax.legend(fontsize=7)
plt.tight_layout(); fig.savefig(f"{FIG}/kan_vs_mlp.png", dpi=150); plt.show()
""")
code("""
# parameter-matched view: for each MLP, the KAN (selected seed) grid whose parameter count is closest
rows = []
for _, r in (mlp_sel.sort_values("params").iterrows() if len(mlp_df) else []):
    j = (sel.params - r.params).abs().idxmin()
    rows.append({"MLP": f"{r.model}, width {int(r.width)}", "MLP params": int(r.params), "MLP test RMSE": r.test_rmse,
                 "KAN params (G)": f"{int(sel.loc[j, 'params'])} (G={int(sel.loc[j, 'G'])})", "KAN test RMSE": sel.loc[j, "test_rmse"],
                 "KAN median over seeds": k211[k211.G == sel.loc[j, "G"]].test_rmse.median()})
matched = pd.DataFrame(rows); matched.to_csv(f"{OUT}/kan_vs_mlp_matched.csv", index=False)
matched.style.format({"MLP test RMSE": "{:.2e}", "KAN test RMSE": "{:.2e}", "KAN median over seeds": "{:.2e}"})
""")
md("### 12.5 Side-by-side with the paper\nThe paper gives **no numeric RMSE table** for these figures, only exponents and thresholds in the text, so only those are compared. Our numbers are measured; the paper numbers are copied from the text (Sec. 2.4, p.10; Sec. 3.1, p.14).")
code("""
a_sel = alpha_df.loc[best_seed]
best_mlp = mlp_sel.loc[mlp_sel.val_rmse.idxmin()] if len(mlp_df) else None   # MLP selected on validation, like the KAN
comparison = pd.DataFrame([
    ["[2,1,1] test-RMSE exponent vs G (mean sq. err.)", "~3 (Sec. 2.4)", f"{a_sel.alpha_mean:.2f} (G<={int(a_sel.G_star_by_val)}, seed {best_seed})"],
    ["[2,1,1] exponent vs G (sqrt of median sq. err.)", "closer to 4 (Sec. 2.4)", f"{a_sel.alpha_median:.2f}"],
    ["[2,1,1] exponents on the pre-saturation window G<=20 (chosen post hoc)", "~3 / closer to 4", f"{scaling_exponent(sel[sel.G <= 20].G, sel[sel.G <= 20].test_rmse):.2f} / {scaling_exponent(sel[sel.G <= 20].G, sel[sel.G <= 20].test_median_rmse):.2f}"],
    ["Theory exponent for k=3", "4 (Thm 2.1)", "4 (not measured; see order ablation)"],
    ["[2,5,1] interpolation threshold G", "predicted ~67, observed ~50", f"validation-optimal G = {thr['val-optimal G'].tolist()} (seeds {thr.index.tolist()}), grid list ...,20,50,100"],
    ["Best final test RMSE, KAN [2,1,1]", "not tabulated", f"{sel.test_rmse.iloc[-1]:.2e} (G=100) / best over G: {sel.test_rmse.min():.2e}"],
    ["Test RMSE of validation-selected MLP (this sweep)", "MLPs plateau, worse than KAN (Fig. 3.1)", f"{best_mlp.test_rmse:.2e} ({best_mlp.model}, width {int(best_mlp.width)}, {int(best_mlp.params)} params)" if best_mlp is not None else "n/a"],
], columns=["Quantity", "Paper (reported)", "This reproduction (measured)"])
comparison.to_csv(f"{OUT}/paper_comparison.csv", index=False)
comparison
""")
md(I("repro"))

# ---------------------------------------------------------------- 14. ablation
md("""
## 13. Ablation study
**A. Grid extension vs training from scratch (design choice of Sec. 2.4).** *Hypothesis (stated before looking at the result):* warm-starting each finer grid from the previous coarse one by least squares reaches lower error than training a freshly initialised KAN at each G with the same number of LBFGS steps, because it preserves what the coarse model learned. *Changed:* initialisation of each stage. *Held fixed:* data, shape $[2,1,1]$, $k=3$, grids, steps per grid, optimiser, seeds.

**B. Spline order $k\\in\\{1,2,3\\}$ (Thm 2.1).** *Hypothesis:* the error decreases faster with G for higher $k$, with exponent roughly $k+1$ (bounded by optimisation and boundary effects). *Changed:* $k$. *Held fixed:* everything else; grids 3, 5, 10, 20, 50.
""")
code("""
abl_files = [f"{OUT}/ablation_extension.csv", f"{OUT}/ablation_order.csv"]
if RECOMPUTE or not all(os.path.exists(p) for p in abl_files):
    print("computing ablations ...")
    run_ablation(CONFIG, OUT, data=dset, smoke=SMOKE)
abl_ext = pd.read_csv(abl_files[0]); abl_ord = pd.read_csv(abl_files[1])
ext_df = k211[["seed", "G", "params", "train_rmse", "val_rmse", "test_rmse", "test_median_rmse"]].assign(variant="grid_extension")
cmp_ext = pd.concat([ext_df, abl_ext], ignore_index=True)
tab = cmp_ext.groupby(["variant", "G"]).test_rmse.agg(["median", "min", "max"]).unstack(0)
tab.style.format("{:.2e}")
""")
code("""
# robustness view: how many of the seeds reach test RMSE < 1e-2 at each G (descriptive threshold)
ok = cmp_ext.assign(ok=cmp_ext.test_rmse < 1e-2).groupby(["variant", "G"]).ok.sum().unstack(0)
ok.index.name = "G"; ok.columns.name = None
print("number of seeds (out of", cmp_ext.seed.nunique(), ") with test RMSE < 1e-2:"); ok
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for v, c in [("grid_extension", "tab:blue"), ("from_scratch", "tab:red")]:
    d = cmp_ext[cmp_ext.variant == v]
    for s, ds in d.groupby("seed"):
        ax[0].loglog(ds.sort_values("G").G, ds.sort_values("G").test_rmse, "o-", color=c, alpha=0.3, lw=1)
    med = d.groupby("G").test_rmse.median()
    ax[0].loglog(med.index, med.values, "o-", color=c, lw=2.4, label=f"{v} (median over seeds)")
ax[0].set_xlabel("G"); ax[0].set_ylabel("test RMSE"); ax[0].set_title("Ablation A: grid extension vs from scratch"); ax[0].legend(fontsize=8)
final_G = cmp_ext.G.max()
fin = cmp_ext[cmp_ext.G == final_G]
for i, (v, c) in enumerate([("grid_extension", "tab:blue"), ("from_scratch", "tab:red")]):
    vals = fin[fin.variant == v].sort_values("seed").test_rmse.values
    ax[1].semilogy([i] * len(vals), vals, "o", color=c)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["grid extension", "from scratch"]); ax[1].set_title(f"final test RMSE at G={final_G}, one dot per seed")
plt.tight_layout(); fig.savefig(f"{FIG}/ablation_extension.png", dpi=150); plt.show()
fin.groupby("variant").test_rmse.agg(["median", "min", "max"]).style.format("{:.2e}")
""")
md(I("ablA"))
code("""
rows = []
for kk, d in abl_ord.groupby("variant"):
    for seed, ds in d.groupby("seed"):
        ds = ds.sort_values("G")
        a, Gs = fit_range_exponent(ds, "test_rmse")
        rows.append(dict(order=kk, seed=seed, G_star_by_val=Gs, alpha=a, final_val_rmse=ds.val_rmse.iloc[-1],
                         final_test_rmse=ds.test_rmse.iloc[-1], best_test_rmse=ds.test_rmse.min()))
ord_df = pd.DataFrame(rows)
sel_ord = ord_df.loc[ord_df.groupby("order").final_val_rmse.idxmin()].set_index("order")   # per order: seed chosen on validation
sel_ord["theory alpha (k+1)"] = [int(o[2:]) + 1 for o in sel_ord.index]
sel_ord["seeds with test RMSE<1e-2"] = [int((ord_df[ord_df.order == o].final_test_rmse < 1e-2).sum()) for o in sel_ord.index]
sel_ord.to_csv(f"{OUT}/ablation_order_selected.csv")
print("validation-selected seed per spline order (fit range: grids up to the validation-optimal G, at least 3 grids)")
sel_ord.style.format({"alpha": "{:.2f}", "final_val_rmse": "{:.2e}", "final_test_rmse": "{:.2e}", "best_test_rmse": "{:.2e}"})
""")
code("""
ord_df.style.format({"alpha": "{:.2f}", "final_val_rmse": "{:.2e}", "final_test_rmse": "{:.2e}", "best_test_rmse": "{:.2e}"})
""")
code("""
fig, ax = plt.subplots(figsize=(6, 4))
for kk, c in zip(["k=1", "k=2", "k=3"], ["tab:green", "tab:orange", "tab:blue"]):
    d = abl_ord[abl_ord.variant == kk]
    for s, ds in d.groupby("seed"):
        ax.loglog(ds.sort_values("G").G, ds.sort_values("G").test_rmse, "o-", color=c, alpha=0.25, lw=1)
    med = d.groupby("G").test_rmse.median(); ax.loglog(med.index, med.values, "o-", color=c, lw=2.4, label=f"{kk} (median over seeds)")
ax.set_xlabel("G"); ax.set_ylabel("test RMSE"); ax.set_title("Ablation B: spline order"); ax.legend()
plt.tight_layout(); fig.savefig(f"{FIG}/ablation_order.png", dpi=150); plt.show()
""")
md(I("ablB"))

# ---------------------------------------------------------------- 15. analysis
md("""
## 14. Analysis: what did the KAN learn, and where does it fail?
**Learned edge functions** (interpretability, the paper's other claim). If the $[2,1,1]$ KAN found the exact decomposition, the first-layer edges should be (up to an affine map, which the second layer absorbs) $\\sin(\\pi x)$ and $y^2$. We test this with the Pearson correlation between each learned edge and its ground-truth counterpart, evaluated on the *live-trained* model.
""")
code("""
xg = torch.linspace(-1, 1, 400)
l0 = kan_live.layers[0]
phi_x, phi_y = l0.edge_function(0, 0, xg), l0.edge_function(1, 0, xg)
corr_x = float(torch.corrcoef(torch.stack([phi_x, torch.sin(torch.pi * xg)]))[0, 1])
corr_y = float(torch.corrcoef(torch.stack([phi_y, xg ** 2]))[0, 1])
print(f"|corr(phi_x, sin(pi x))| = {abs(corr_x):.4f}   |corr(phi_y, y^2)| = {abs(corr_y):.4f}")
h_range = (float(kan_live.layers[0](dset['test'][0]).min()), float(kan_live.layers[0](dset['test'][0]).max()))
hg = torch.linspace(h_range[0], h_range[1], 400)
phi_out = kan_live.layers[1].edge_function(0, 0, hg)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
ax[0].plot(xg, phi_x, label="learned phi(x)"); a2 = ax[0].twinx(); a2.plot(xg, torch.sin(torch.pi * xg), "r--", label="sin(pi x)"); ax[0].set_title(f"edge x->h  |corr|={abs(corr_x):.3f}")
ax[1].plot(xg, phi_y, label="learned phi(y)"); a3 = ax[1].twinx(); a3.plot(xg, xg ** 2, "r--"); ax[1].set_title(f"edge y->h  |corr|={abs(corr_y):.3f}")
ax[2].plot(hg, phi_out); ax[2].set_title("edge h->f (should look like exp)"); ax[2].set_xlabel("h")
plt.tight_layout(); fig.savefig(f"{FIG}/learned_edges.png", dpi=150); plt.show()
""")
md("**Error analysis.** Where on the input square does the live-trained model make its largest errors? (The paper attributes slow scaling to boundary effects.)")
code("""
xs_t, ys_t = dset["test"]
with torch.no_grad():
    err2 = ((kan_live(xs_t) - ys_t) ** 2).squeeze().numpy()
print(f"test RMSE {err2.mean() ** 0.5:.3e} | sqrt(median) {numpy.median(err2) ** 0.5:.3e} | share of squared error from worst 1% of points: {numpy.sort(err2)[-len(err2) // 100:].sum() / err2.sum():.1%}")
border = (xs_t.abs().max(1).values > 0.9).numpy()
print(f"points within 10% of the boundary: {border.mean():.1%} of data, {err2[border].sum() / err2.sum():.1%} of squared error")
plt.figure(figsize=(4.8, 4)); sc = plt.scatter(xs_t[:, 0], xs_t[:, 1], c=numpy.log10(err2 + 1e-30), s=3, cmap="magma"); plt.colorbar(sc, label="log10 squared error")
plt.xlabel("x"); plt.ylabel("y"); plt.title("test error map (live model)"); plt.tight_layout(); plt.savefig(f"{FIG}/error_map.png", dpi=150); plt.show()
""")
md("""
**Computational cost.** Runtime per grid stage is recorded in `seconds` (`kan_df`). One KAN edge costs $O(k^2)$ per sample for the recursion plus $G+k$ multiply-adds; training cost grows with G mostly through LBFGS line searches on a harder loss landscape (the paper makes the same remark in footnote 4, p.10).
""")
code("""
cost = k211.groupby("G").agg(params=("params", "first"), seconds_median=("seconds", "median"))
cost.style.format({"seconds_median": "{:.1f}"})
""")
md(I("analysis"))

# ---------------------------------------------------------------- 16. save / reload
md("## 15. Saving and reproducing inference\nThe checkpoint stores the weights, the (data-dependent) knot vectors and the configuration; we reload it into a fresh model and verify that predictions are identical.")
code("""
ckpt = {"state_dict": kan_live.state_dict(), "width": K["width"], "G": kan_live.G, "k": K["k"], "config": CONFIG, "seed": live_seed}
torch.save(ckpt, f"{CKPT}/kan_live.pt")
json.dump(CONFIG, open(f"{OUT}/config_used.json", "w"), indent=2)
live_df.to_csv(f"{OUT}/live_stage_metrics.csv", index=False)
loaded = torch.load(f"{CKPT}/kan_live.pt", weights_only=False)
kan_reloaded = KAN(loaded["width"], G=loaded["G"], k=loaded["k"]); kan_reloaded.load_state_dict(loaded["state_dict"])
with torch.no_grad():
    same = torch.allclose(kan_live(dset["test"][0]), kan_reloaded(dset["test"][0]), atol=1e-12)
print("reloaded model reproduces predictions exactly:", same, "| test RMSE:", rmse(kan_reloaded, *dset["test"]))
""")
md("""
**How to rerun.** Open this notebook in Colab (CPU or T4 both work), `Runtime → Run all`. With the `results/` folder present it takes ~3 min; set `RECOMPUTE = True` to regenerate everything (~30–40 min on CPU). The same experiments can be run from the command line with `python scripts/run_experiment.py` and `python scripts/run_ablation.py`.
""")

# ---------------------------------------------------------------- 17. conclusion

# =====================================================================================================
# PART B - symbolic regression (Feynman) + symbolic simplification
# =====================================================================================================
md("""
---
# Part B — Symbolic regression and symbolic simplification (Feynman equations)
**Assignment items covered:** *compare with MLPs of equal parameters on symbolic-regression toy tasks* and *try symbolic simplification*.

**Setup (paper Sec. 3.3, Table 2).** Six dimensionless Feynman equations with two variables each. For every equation we compare
1. a **human-constructed KAN** (the shape listed in Table 2) trained with grid extension,
2. an **MLP with (almost) the same number of parameters** (SiLU, depth 2/3/4 tried, best depth chosen on *validation* RMSE),
3. a **pruned KAN** found automatically (start from a wide KAN, train with sparsity regularisation, prune) and an MLP matched to *its* parameter count,
4. the **symbolic formula** obtained by fitting a library function to every remaining edge (symbolification).

**Honest differences from the paper:** the paper tunes a wide sweep of KAN depths/regularisation strengths and reports the best of many runs; we run **one wide shape ([n,5,1])** with a small (λ, seed) search chosen on validation RMSE, fewer steps, and our own sampling ranges for the variables (the paper only states (−1,1) for I.16.6). The paper's numbers are shown next to ours for orientation, **they are not expected to match**. Parameter counts of the MLP can only match the KAN approximately (integer widths); the relative gap is stored in the results.
""")
code("""
# Tiny float64 models: the CPU is often as fast as a GPU here. Set SMALL_DEVICE = device to run Parts B and C on the GPU.
SMALL_DEVICE = device
if SMOKE:   # tiny configuration just to check that every cell runs (numbers are NOT meaningful)
    CONFIG["feynman"].update(equations=["II.35.18"], data=dict(n_train=300, n_val=200, n_test=500, seed=0))
    CONFIG["feynman"]["human"].update(grids=[3, 5], steps_per_grid=10)
    CONFIG["feynman"]["mlp"].update(depths=[2], steps=20)
    CONFIG["feynman"]["simplify"].update(seeds=[0], steps_reg=15, steps_retrain=15, steps_affine=5)
    CONFIG["feynman"]["demo"].update(seeds=[0])
    CONFIG["pde"].update(grids=[3, 5], steps_per_grid=5, ni=500, nb=100, mlp_depths=[3])
    CONFIG["mnist"].update(epochs=1, models=[dict(kan_width=[784, 4, 10], G=3, k=3, mlp_depths=[2])])
print("small-model device:", SMALL_DEVICE)
""")
md("""
## 16. Simplification techniques (paper Sec. 2.5.1)
**Sparsification.** The L1 norm of an edge function is its average magnitude over the $N_p$ training inputs, and a layer's L1 norm sums its edges (Eq. 2.17–2.18):
$$|\\phi|_1=\\frac1{N_p}\\sum_{s=1}^{N_p}|\\phi(x^{(s)})|,\\qquad |\\Phi|_1=\\sum_{i,j}|\\phi_{i,j}|_1 .$$
L1 alone is not enough, so an entropy term pushes the *distribution of magnitudes* across edges to be peaky (Eq. 2.19), and the training objective is (Eq. 2.20)
$$S(\\Phi)=-\\sum_{i,j}\\frac{|\\phi_{i,j}|_1}{|\\Phi|_1}\\log\\frac{|\\phi_{i,j}|_1}{|\\Phi|_1},\\qquad \\ell_{total}=\\ell_{pred}+\\lambda\\Big(\\mu_1\\sum_l|\\Phi_l|_1+\\mu_2\\sum_lS(\\Phi_l)\\Big),\\ \\mu_1=\\mu_2=1 .$$
**Pruning (Eq. 2.21).** A hidden node survives only if both its incoming score $I=\\max_k|\\phi_{\\text{in},k}|_1$ and outgoing score $O=\\max_j|\\phi_{\\text{out},j}|_1$ exceed $\\theta=10^{-2}$; pruned nodes get all their edges masked to zero.

**Symbolification.** An edge cannot simply be set to $\\sin$ or $\\exp$ because its input and output may be shifted and scaled, so we fit $y\\approx c\\,f(ax+b)+d$ to the edge's pre-/post-activations: a grid search over $(a,b)$ (zoomed three times) where, for each pair, $(c,d)$ is the closed-form linear-regression solution and the pair is scored by $R^2$. After all edges are symbolic, only the affine parameters remain; fine-tuning them with LBFGS should drive the loss towards machine precision when the structure is right, and SymPy composes the edges into a formula.

The regulariser (`KAN.forward_reg`), pruning (`KAN.prune`) and the edge masks/symbolic edges live in the `KAN`/`KANLayer` cells above; the fitting tools are below.
""")
md("### 16.1 Fitting a symbolic function to one edge")
code(symbolic["symfit"])
md("### 16.2 Fixing edges, ranking candidates, automatic symbolification, fine-tuning")
code(symbolic["symapply"])
md("### 16.3 From symbolic edges to a closed-form formula")
code(symbolic["symformula"])
md("**Sanity checks of the machinery** on cases with a known answer: (i) the affine fit recovers a known function, (ii) the regulariser equals its definition, (iii) pruning removes a dead node.")
code("""
xg = torch.linspace(-1, 1, 500, dtype=torch.float64)
yg = 2.0 * torch.sin(3.0 * xg + 0.5) + 1.0
a, b, c, d, r2 = fit_affine(xg, yg, "sin")
print(f"fit of 2*sin(3x+0.5)+1 with 'sin': a={a:.3f} b={b:.3f} c={c:.3f} d={d:.3f}  R^2={r2:.6f}")
print("max reconstruction error:", float((c * torch.sin(a * xg + b) + d - yg).abs().max()),
      "(a, b, c are only identified up to symmetries such as (a,b,c) -> (-a, pi-b, c))")

toy = KAN([2, 4, 1], G=5).double(); xs_toy = torch.rand(100, 2).double() * 2 - 1
out_, reg_ = toy.forward_reg(xs_toy)
manual, h_ = 0, xs_toy
for lyr in toy.layers:
    l1_ = lyr.post_activations(h_).abs().mean(0); p_ = l1_ / l1_.sum()
    manual = manual + l1_.sum() - (p_ * torch.log(p_)).sum(); h_ = lyr(h_)
print("regulariser equals Eq. 2.18-2.20 by hand:", bool(torch.allclose(reg_, manual)))
with torch.no_grad():                      # kill hidden node 3 on purpose
    toy.layers[0].w_base.zero_(); toy.layers[0].w_spline.zero_(); toy.layers[0].w_base[:, :3] = 1.0
print("shape after pruning a dead node:", toy.prune(xs_toy, theta=1e-2), "(expected [2, 3, 1])")
""")

md("""
## 17. Worked example: the paper's own symbolic-regression walk-through (Sec. 2.5.2, Fig. 2.4)
Target $f(x,y)=\\exp(\\sin(\\pi x)+y^2)$. The paper's steps: (1) train a $[2,5,1]$ KAN with sparsification, (2) prune to $[2,1,1]$, (3) set the edges to `sin`, `x^2`, `exp`, (4) train the affine parameters until the loss reaches machine precision, (5) read off the formula with SymPy. Here step 3 is automatic (best $R^2$ among the library, ties broken towards simpler functions) and the best (λ, seed) is chosen on **validation** RMSE of the final symbolic model. Optimisation of KANs is seed-dependent, so some seeds will end up with a messier formula; the selection reports what the best one achieved.
""")
code(feynman["feynman"])
code("""
FC = CONFIG["feynman"]
simp_kw = {k: v for k, v in FC["simplify"].items() if k not in ("lambdas", "seeds")}
demo_data = make_equation_data("exp_sin", **FC["data"], device=SMALL_DEVICE)
t0 = time.time()
demo = simplify_kan("exp_sin", demo_data, lambdas=FC["demo"]["lambdas"], seeds=FC["demo"]["seeds"], **simp_kw)
print(f"done in {time.time() - t0:.0f}s")
""")
code("""
if demo is None:
    print("no (lambda, seed) combination produced a fully symbolic KAN; increase steps or seeds in CONFIG['feynman']")
else:
    print("pruned shape:", demo["pruned_shape"], "| pruned KAN test RMSE: %.2e" % demo["pruned_test_rmse"])
    print("true formula :", EQUATIONS["exp_sin"]["text"])
    print("recovered    :", demo["formula"])
    print("test RMSE of the recovered formula against the true function: %.2e" % demo["symbolic_test_rmse"])
    print({k: (v[0], round(v[1], 4)) for k, v in demo["edges"].items()})   # (layer, in, out) -> (function, R^2)
""")
code(plots["plotkan"])
md("Edge functions **before pruning** (after sparsity-regularised training; fading = small magnitude) and **after symbolification** (red titles = chosen function; pruned edges are blank):")
code("""
if demo is not None:
    for fig in plot_kan_edges(demo["model_reg"], demo_data["train"][0], title="after sparsification, before pruning"): plt.show()
    for fig in plot_kan_edges(demo["model"], demo_data["train"][0], title="after pruning + symbolification"): plt.show()
""")

md("""
## 18. KAN vs equal-parameter MLP on the Feynman equations
For each equation we train the human-constructed KAN and an MLP with the same parameter budget under the same optimiser (LBFGS), then run the automatic simplification pipeline. Training uses full-batch LBFGS with strong-Wolfe line search for both model families; model selection (best checkpoint per grid stage, MLP depth, (λ, seed)) uses validation RMSE only; test RMSE is only reported.
""")
code("""
FC = CONFIG["feynman"]
feyn_rows = []
t_all = time.time()
for name in FC["equations"]:
    eq = EQUATIONS[name]
    data = make_equation_data(name, **FC["data"], device=SMALL_DEVICE)
    h, _ = run_human_kan(name, data, **FC["human"])
    m_h = run_matched_mlp(h["params"], data, depths=FC["mlp"]["depths"], steps=FC["mlp"]["steps"], seed=FC["mlp"]["seed"])
    s = simplify_kan(name, data, lambdas=FC["simplify"]["lambdas"], seeds=FC["simplify"]["seeds"], **simp_kw)
    row = dict(equation=name, formula=eq["text"], human_shape=str(h["shape"]), human_kan_params=h["params"], human_kan_rmse=h["test_rmse"],
               mlp_eq_human_params=m_h["params"], mlp_eq_human_gap=m_h["param_gap"], mlp_eq_human_rmse=m_h["test_rmse"], mlp_depth=m_h["depth"],
               paper_human_kan=eq["paper"][0], paper_pruned_kan=eq["paper"][1], paper_mlp=eq["paper"][3])
    if s is not None:
        m_p = run_matched_mlp(max(s["pruned_params"], 8), data, depths=FC["mlp"]["depths"], steps=FC["mlp"]["steps"], seed=FC["mlp"]["seed"])
        row.update(pruned_shape=str(s["pruned_shape"]), pruned_params=s["pruned_params"], pruned_kan_rmse=s["pruned_test_rmse"],
                   mlp_eq_pruned_params=m_p["params"], mlp_eq_pruned_rmse=m_p["test_rmse"],
                   symbolic_rmse=s["symbolic_test_rmse"], recovered_formula=str(s["formula"]))
    feyn_rows.append(row)
    print(f"{name}: human KAN {h['test_rmse']:.2e} | MLP(equal params) {m_h['test_rmse']:.2e} | symbolic {row.get('symbolic_rmse', float('nan')):.2e}")
print(f"total time {time.time() - t_all:.0f}s")
feyn_df = pd.DataFrame(feyn_rows)
os.makedirs(OUT, exist_ok=True)
feyn_df.to_csv(f"{OUT}/feynman_results.csv", index=False)
feyn_df[["equation", "human_shape", "human_kan_params", "human_kan_rmse", "mlp_eq_human_params", "mlp_eq_human_rmse",
         "paper_human_kan", "paper_mlp"]].style.format({c: "{:.2e}" for c in ["human_kan_rmse", "mlp_eq_human_rmse", "paper_human_kan", "paper_mlp"]})
""")
code("""
cols = [c for c in ["equation", "pruned_shape", "pruned_params", "pruned_kan_rmse", "mlp_eq_pruned_params", "mlp_eq_pruned_rmse", "symbolic_rmse", "paper_pruned_kan"] if c in feyn_df]
feyn_df[cols].style.format({c: "{:.2e}" for c in cols if c.endswith("rmse") or c.startswith("paper")})
""")
code("""
pd.set_option("display.max_colwidth", 120)
feyn_df[[c for c in ["equation", "formula", "recovered_formula"] if c in feyn_df]]
""")
code("""
fig, ax = plt.subplots(figsize=(9, 3.8))
series = [("human_kan_rmse", "human-constructed KAN"), ("mlp_eq_human_rmse", "MLP (equal params)"),
          ("pruned_kan_rmse", "pruned KAN"), ("symbolic_rmse", "symbolic formula")]
w = 0.2
for k_, (col, lab) in enumerate(series):
    if col in feyn_df:
        ax.bar(numpy.arange(len(feyn_df)) + (k_ - 1.5) * w, feyn_df[col].astype(float), w, label=lab)
ax.set_yscale("log"); ax.set_xticks(range(len(feyn_df))); ax.set_xticklabels(feyn_df.equation, rotation=20)
ax.set_ylabel("test RMSE"); ax.set_title("Feynman equations: KAN vs equal-parameter MLP vs symbolic formula"); ax.legend(fontsize=7)
plt.tight_layout(); fig.savefig(f"{FIG}/feynman_comparison.png", dpi=150); plt.show()
""")
md("""
**How to read this part (fill in after you have run it).**
* *Human KAN vs MLP at equal parameters:* the paper's Table 2 reports the MLP as better on several equations and the KAN better on others, so do not expect a uniform win. Compare *your* two columns.
* *Pruned KAN:* check the pruned shape against the human shape and the paper's smallest-shape column.
* *Symbolic RMSE:* values near machine precision (≈1e-10 or lower) mean the exact formula was found; values around 1e-2–1e-3 mean a good approximation with the wrong functional form (e.g. products such as $a\\cdot b$ or $n\\ln a$ need a $[2,2,1]$ structure with squares inside, which an automatic best-R² choice per edge may miss; the paper constructs those by hand).
* Seeds matter: optimisation is chaotic (see the thread-count experiment in Section 12), so rerun with more seeds if a row fails.
""")

# =====================================================================================================
# PART C - PDE
# =====================================================================================================
md("""
---
# Part C — PDE toy task: Poisson equation with a physics-informed loss  (Section 19)
**Assignment item covered:** *compare with MLPs of equal parameters on PDE toy tasks.* (paper Sec. 3.4)

$$u_{xx}+u_{yy}=f\\ \\text{in }\\Omega=[-1,1]^2,\\qquad u=0\\ \\text{on }\\partial\\Omega,$$
with $f=-\\pi^2(1+4y^2)\\sin(\\pi x)\\sin(\\pi y^2)+2\\pi\\sin(\\pi x)\\cos(\\pi y^2)$, whose exact solution is $u=\\sin(\\pi x)\\sin(\\pi y^2)$. The network $u_\\theta$ is trained **only** with the physics-informed loss
$$\\ell=\\alpha\\,\\frac1{n_i}\\sum_{i}|u_{xx}(z_i)+u_{yy}(z_i)-f(z_i)|^2+\\frac1{n_b}\\sum_{b}u_\\theta(z_b)^2,\\qquad n_i=10000,\\ n_b=800,\\ \\alpha=0.01$$
(paper values). The exact solution is used only to report the final error. Second derivatives are obtained by automatic differentiation *through* the B-spline basis (a cubic spline has continuous second derivatives, so $u_{xx},u_{yy}$ exist). Both model families use LBFGS and the same total number of steps; the KAN uses grid extension $G=3\\to5\\to10\\to20$; for **every KAN size an MLP with (almost) equal parameters** (SiLU, depth 3 and 4) is trained. We report the final iterate (no checkpoint selection). The paper notes the ground truth being symbolic favours KANs, and that its KANs are ~10× slower to train.
""")
code(pde["pde"])
code("""
PC = CONFIG["pde"]
pde_rows, kan_pde = run_pde_kan(width=PC["width"], grids=PC["grids"], steps_per_grid=PC["steps_per_grid"], k=PC["k"],
                                ni=PC["ni"], nb=PC["nb"], alpha=PC["alpha"], seed=PC["seed"], device=SMALL_DEVICE)
total_steps = len(PC["grids"]) * PC["steps_per_grid"]
best_mlp_pde, best_mlp_row = None, None
for r in list(pde_rows):
    for depth in PC["mlp_depths"]:
        row, mdl = run_pde_mlp(r["params"], depth, steps=total_steps, ni=PC["ni"], nb=PC["nb"], alpha=PC["alpha"],
                               seed=PC["seed"], device=SMALL_DEVICE, return_model=True)
        pde_rows.append(row)
        print(f"  {row['model']} params={row['params']} (gap {row['param_gap']:+.1%}): rel L2={row['rel_l2']:.2e}")
        if best_mlp_row is None or row["rel_l2"] < best_mlp_row["rel_l2"]:
            best_mlp_pde, best_mlp_row = mdl, row
pde_df = pd.DataFrame(pde_rows)
pde_df.to_csv(f"{OUT}/pde_results.csv", index=False)
pde_df.style.format({"mse": "{:.2e}", "rel_l2": "{:.2e}", "final_loss": "{:.2e}", "seconds": "{:.0f}", "param_gap": "{:+.1%}"}, na_rep="-")
""")
code("""
fig, ax = plt.subplots(figsize=(6, 4))
kd = pde_df[pde_df.model.str.startswith("KAN")]
ax.loglog(kd.params, kd.rel_l2, "o-", lw=2, label="KAN [2,10,1] (grid extension)")
for depth, mk in zip(PC["mlp_depths"], ["s", "^", "v"]):
    md_ = pde_df[pde_df.model == f"MLP(depth {depth})"]
    ax.loglog(md_.params, md_.rel_l2, mk + "--", label=f"MLP depth {depth} (equal params)")
ax.set_xlabel("number of parameters"); ax.set_ylabel("relative L2 error vs exact solution"); ax.set_title("Poisson PINN: KAN vs MLP"); ax.legend(fontsize=8)
plt.tight_layout(); fig.savefig(f"{FIG}/pde_kan_vs_mlp.png", dpi=150); plt.show()
""")
code("""
g1 = torch.linspace(-1, 1, 101, dtype=torch.float64, device=SMALL_DEVICE)
GX, GY = torch.meshgrid(g1, g1, indexing="ij"); zg = torch.stack([GX.reshape(-1), GY.reshape(-1)], 1)
with torch.no_grad():
    u_ex = u_true(zg).reshape(101, 101).cpu()
    u_k = kan_pde(zg).reshape(101, 101).cpu(); u_m = best_mlp_pde(zg).reshape(101, 101).cpu()
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
im = ax[0].contourf(GX.cpu(), GY.cpu(), u_ex, 30); plt.colorbar(im, ax=ax[0]); ax[0].set_title("exact u")
im = ax[1].contourf(GX.cpu(), GY.cpu(), (u_k - u_ex).abs() + 1e-16, 30); plt.colorbar(im, ax=ax[1]); ax[1].set_title("|KAN - exact| (largest G)")
im = ax[2].contourf(GX.cpu(), GY.cpu(), (u_m - u_ex).abs() + 1e-16, 30); plt.colorbar(im, ax=ax[2]); ax[2].set_title(f"|MLP - exact| ({best_mlp_row['model']}, best)")
plt.tight_layout(); fig.savefig(f"{FIG}/pde_error_maps.png", dpi=150); plt.show()
""")
md("""
**How to read this part (fill in after you have run it).** Compare relative L2 error at equal parameter count (the paper claims a 2-layer width-10 KAN is far more accurate and parameter-efficient than a deeper MLP, with a caveat that the symbolic ground truth favours KANs). Check the error maps for where each model fails (boundary vs interior) and the `seconds` column for the training-cost penalty of KANs. If errors stagnate, increase `steps_per_grid` or `ni` in `CONFIG["pde"]`; results with only one seed and one collocation sample are indicative, not statistical.
""")

# =====================================================================================================
# PART D - MNIST
# =====================================================================================================
md("""
---
# Part D — MNIST: KAN vs equal-parameter MLP  (Section 20)
**Assignment item covered:** *compare with MLPs of equal parameters on … MNIST.* (not an experiment of the KAN paper; the paper's own tasks are small AI+Science problems)

**Design.** Images are flattened to 784 values scaled to $[-1,1]$ (the KAN's grid range, no statistics are fitted, so there is no leakage). Train/validation/test = 55 000 / 5 000 / 10 000 (validation split from the official training set; the official test set is used once). The KAN is a 2-layer $[784,h,10]$ network with cubic splines; for each KAN an **MLP with (almost) the same number of parameters** (SiLU, 1 or 2 hidden layers) is built. Both are trained with AdamW + cross-entropy for the same number of epochs, and the checkpoint with the best *validation* accuracy is evaluated on the test set. KAN-specific handling: the first layer keeps its fixed uniform grid on $[-1,1]$; hidden-layer grids are re-placed from the activations at the start of the first epochs (the paper's "update grids on the fly") with uniform knots over the observed range (data quantiles are degenerate for hidden sums of mostly-constant pixels). Training uses `float32` on the GPU. **Reduced scale / caveats:** one seed, no hyper-parameter tuning, no data augmentation, small hidden widths so that the dense B-spline evaluation (cost ∝ $n_{in}\\cdot n_{out}\\cdot(G+k)$ per sample) stays cheap.
""")
code(mnist["mnist"])
code("""
MC = CONFIG["mnist"]
mn = load_mnist(MC["root"], n_val=MC["n_val"], device=device, synthetic=SMOKE, seed=MC["seed"])
if SMOKE:
    print("SMOKE mode: using a small synthetic MNIST-shaped dataset (numbers are meaningless)")
for k_, (x_, y_) in mn.items():
    print(f"{k_:5s} X{tuple(x_.shape)} y{tuple(y_.shape)}  x in [{float(x_.min()):.2f},{float(x_.max()):.2f}]  class counts: {torch.bincount(y_, minlength=10).tolist()}")
fig, ax = plt.subplots(1, 10, figsize=(14, 1.8))
for c_ in range(10):
    img = mn["train"][0][(mn["train"][1] == c_).nonzero()[0, 0]].reshape(28, 28).cpu()
    ax[c_].imshow(img, cmap="gray"); ax[c_].axis("off"); ax[c_].set_title(str(c_))
plt.show()
""")
code("""
mnist_rows, mnist_hist, mnist_models = [], {}, {}
for mc in MC["models"]:
    kan_m = KAN(mc["kan_width"], G=mc["G"], k=mc["k"]).to(device=device, dtype=torch.float32)
    name_k = f"KAN {mc['kan_width']} G={mc['G']}"
    print(name_k, "parameters:", kan_m.n_params())
    res, hist = train_classifier(kan_m, mn, epochs=MC["epochs"], lr=MC["lr"], weight_decay=MC["weight_decay"],
                                 batch_size=MC["batch_size"], grid_update_epochs=MC["grid_update_epochs"], seed=MC["seed"])
    mnist_rows.append(dict(model=name_k, param_gap=0.0, **res)); mnist_hist[name_k] = hist; mnist_models[name_k] = kan_m
    for depth in mc["mlp_depths"]:
        mlp_m, gap = matched_mlp(res["params"], 784, depth, 10)
        mlp_m = mlp_m.to(device=device, dtype=torch.float32)
        name_m = f"MLP depth {depth} (equal params to {name_k})"
        print(name_m, "parameters:", mlp_m.n_params(), f"(gap {gap:+.2%})")
        res2, hist2 = train_classifier(mlp_m, mn, epochs=MC["epochs"], lr=MC["lr"], weight_decay=MC["weight_decay"],
                                       batch_size=MC["batch_size"], seed=MC["seed"])
        mnist_rows.append(dict(model=name_m, param_gap=gap, **res2)); mnist_hist[name_m] = hist2; mnist_models[name_m] = mlp_m
mnist_df = pd.DataFrame(mnist_rows)
mnist_df["seconds_per_epoch"] = mnist_df.seconds / MC["epochs"]
mnist_df.to_csv(f"{OUT}/mnist_results.csv", index=False)
mnist_df.style.format({"val_acc": "{:.4f}", "test_acc": "{:.4f}", "param_gap": "{:+.2%}", "seconds": "{:.0f}", "seconds_per_epoch": "{:.1f}"})
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for name_, h_ in mnist_hist.items():
    ax[0].plot([e["epoch"] for e in h_], [e["val_acc"] for e in h_], "o-" if name_.startswith("KAN") else "s--", label=name_[:48])
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("validation accuracy"); ax[0].set_title("MNIST: validation accuracy"); ax[0].legend(fontsize=6)
ax[1].scatter(mnist_df.params, mnist_df.test_acc, c=["tab:blue" if m.startswith("KAN") else "tab:orange" for m in mnist_df.model])
for _, r_ in mnist_df.iterrows():
    ax[1].annotate(r_.model.split(" (")[0], (r_.params, r_.test_acc), fontsize=6)
ax[1].set_xlabel("parameters"); ax[1].set_ylabel("test accuracy"); ax[1].set_title("test accuracy vs parameters (blue KAN, orange MLP)")
plt.tight_layout(); fig.savefig(f"{FIG}/mnist_kan_vs_mlp.png", dpi=150); plt.show()
""")
code("""
# confusion matrix of the best KAN on the test set
best_name = mnist_df[mnist_df.model.str.startswith("KAN")].sort_values("val_acc").iloc[-1].model
bm = mnist_models[best_name].eval()
with torch.no_grad():
    pred = torch.cat([bm(mn["test"][0][i:i + 2000]).argmax(1) for i in range(0, len(mn["test"][0]), 2000)]).cpu()
conf = numpy.zeros((10, 10), int)
for t_, p_ in zip(mn["test"][1].cpu().tolist(), pred.tolist()):
    conf[t_, p_] += 1
plt.figure(figsize=(4.6, 4)); plt.imshow(conf, cmap="Blues"); plt.colorbar(); plt.xlabel("predicted"); plt.ylabel("true"); plt.title(best_name[:40], fontsize=8)
plt.tight_layout(); plt.savefig(f"{FIG}/mnist_confusion.png", dpi=150); plt.show()
print("per-class accuracy:", [round(conf[i, i] / conf[i].sum(), 3) for i in range(10)])
""")
md("""
**How to read this part (fill in after you have run it).** Compare test accuracy *at equal parameter count*, not at equal width. Expect the MLP to be competitive or better on MNIST: the KAN paper's advantages are for small, smooth, compositional scientific functions, and it does not claim to beat MLPs on image classification. Also compare `seconds_per_epoch` (spline evaluation is much heavier than a matrix multiply). Single seed, no tuning: differences of a few tenths of a percent are not significant.
""")

md(I("conclusion"))
md(I("viva"))
