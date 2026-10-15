# Experiment plan

**Paper reference:** KAN (arXiv:2404.19756v5), Sec. 2.4 + Fig. 2.3 (p.9–10) and Fig. 3.1 panel 2 / Sec. 3.1 (p.14).

**Research question.** Does a KAN with grid extension fit f(x,y)=exp(sin(πx)+y²) increasingly well as G grows, with a power-law
error decay near the paper's G^-3 (G^-4 theory), and does it beat equally sized MLPs?

**Hypotheses (stated before running).** H1: test RMSE falls as a power law in G until the interpolation threshold, exponent between 2 and 4.
H2: a [2,1,1] KAN reaches lower test RMSE than MLPs of the same parameter count. H3 (ablation A): grid extension beats training from scratch at each G.
H4 (ablation B): exponent grows with spline order k (≈ k+1).

**Data.** x~U[-1,1]², 1000 train / 1000 validation / 10 000 test, independent RNG streams, no preprocessing (no leakage possible).
**Models.** KAN [2,1,1] (k=3, 4 seeds), KAN [2,5,1] (2 seeds), MLP SiLU depth 2–4 widths 5–50 (1 seed each).
**Training.** LBFGS (strong Wolfe), full batch, float64, G∈{3,5,10,20,50,100}, 100 steps per grid, grid refresh at steps 0,10,…,40; MLP 600 steps.
Model selection on validation RMSE only. **Metrics.** test RMSE, sqrt(median squared error), fitted log–log exponent, parameter count, seconds.
**Ablations.** A: extend=True vs False. B: k∈{1,2,3}, G∈{3,5,10,20,50}. Same seeds, data and step budget.
**Cost.** ≈2.5 min per [2,1,1] seed, ≈9 min per [2,5,1] seed, ≈1 hour of single-thread CPU in total for everything.
**Differences from the paper.** G≤100 (paper 1000), 100 steps/grid (200), 4 seeds, own choices for unspecified details (see `docs/paper_analysis.md` §4).
**Success criteria.** Tests pass; smoke test runs; all tables/figures generated from saved CSV/JSON; notebook runs top to bottom; claims limited to what the numbers show.
