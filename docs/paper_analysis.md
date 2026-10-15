# Paper analysis — KAN: Kolmogorov–Arnold Networks

**Paper:** Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljačić, T. Y. Hou, M. Tegmark,
"KAN: Kolmogorov–Arnold Networks", arXiv:2404.19756v5 [cs.LG], 9 Feb 2025 (50 pages).
Page numbers below are PDF pages. This is the **original KAN paper** (not a variant), so the original architecture is implemented.

## 1. Problem and motivation
MLPs have fixed activations on nodes and learnable linear weights on edges. The paper proposes KANs: **learnable univariate
activation functions on edges, no linear weights, nodes only sum** (Fig. 0.1, p.2). Motivation: Kolmogorov–Arnold representation
theorem (Eq. 2.1, p.3) in place of the universal approximation theorem; claimed benefits are accuracy (faster neural scaling laws on
small AI+Science tasks) and interpretability.

## 2. Method and key equations
| Item | Equation / statement | Page |
|---|---|---|
| KA representation | f(x)=Σ_q Φ_q(Σ_p φ_{q,p}(x_p)) (2.1) | 3 |
| KAN layer = matrix of 1-D functions Φ={φ_{q,p}} | (2.2) | 5 |
| Shape [n_0,…,n_L]; edge φ_{l,j,i} | (2.3)–(2.4) | 5 |
| Node = sum of incoming post-activations | x_{l+1,j}=Σ_i φ_{l,j,i}(x_{l,i}) (2.5), matrix form (2.6) | 5 |
| Network = composition of layers | KAN(x)=(Φ_{L−1}∘…∘Φ_0)x (2.7) | 5 |
| MLP for comparison | MLP(x)=(W_{L−1}∘σ∘…∘W_0)x (2.9) | 6 |
| Residual activation | φ(x)=w_b·b(x)+w_s·spline(x) (2.10), b=silu (2.11) | 6 |
| Spline | spline(x)=Σ_i c_i B_i(x), trainable c_i (2.12) | 6 |
| Init | w_s=1, spline≈0 via c_i~N(0,σ²), σ=0.1 (footnote 2); w_b Xavier | 6 |
| Grid update on the fly | grids follow input activations (impl. detail 3, footnote 3) | 6 |
| Parameter count | O(N²L(G+k)) vs O(N²L) for an MLP | 6–7 |
| Approximation theory (Thm 2.1) | error ≤ C·G^{−k−1+m}; test RMSE ∝ N^{−4} for k=3 | 7–8 |
| Grid extension | refit finer B-splines to coarse ones by least squares (2.16) | 9–10 |

## 3. Experiments in the paper relevant to us
* **Fig. 2.3 / Sec. 2.4 (p.9–10):** f(x,y)=exp(sin(πx)+y²), 1000 training samples, [2,5,1] and [2,1,1] KANs, G starts at 3 and grows
  every 200 LBFGS steps up to 1000. Reported: staircase loss curves; [2,5,1] interpolation threshold predicted at G≈1000/15≈67,
  observed G≈50; smaller [2,1,1] KAN generalises better; **[2,1,1] test RMSE ∝ G^{−3}** (theory G^{−4}); sqrt(median(sq. err)) scales
  closer to G^{−4}; attributed to boundary effects.
* **Fig. 3.1 / Sec. 3.1 (p.14):** five toy functions, KANs vs MLPs (depth 2–5) trained with LBFGS for 1800 steps,
  G ∈ {3,5,10,20,50,100,200,500,1000}; KANs "almost saturate" α=4, MLPs scale slowly and plateau.
* Other experiments (special functions Table 1, Feynman Table 2 p.16, PDEs, continual learning, interpretability, knot theory) are
  not reproduced.

**Selected for reproduction:** Fig. 2.3 / Sec. 2.4 and the f(x,y)=exp(sin(πx)+y²) panel of Fig. 3.1. It is quantitative, tests the
defining contribution (spline edges + grid extension), is cheap on CPU, and has an MLP comparison.

## 4. Information missing or ambiguous in the paper (our choices, flagged in the code)
1. Input domain and test-set size are not stated (we use U[−1,1]², 10 000 test points; plus a 1000-point validation set).
2. Exact grid-update rule and schedule (we blend data quantiles with a uniform grid, 2 % uniform; refresh at LBFGS steps 0,10,…,40 of each stage).
3. Definition of one "LBFGS step" (we use one `optimizer.step(closure)` = up to 20 inner iterations); we use 100 steps per grid, not 200.
4. Knot padding outside [t_0,t_G] (uniform spacing). B-spline order k=3 is stated; the basis recursion is standard Cox–de Boor.
5. MLP activation and widths in Fig. 3.1 (we use SiLU, depth 2–4, widths 5–50).
6. Number of seeds and which grids Fig. 2.3 used.
7. The paper gives no numeric RMSE table for these figures; comparisons are to exponents/thresholds stated in the text.

## 5. Paper-to-implementation mapping
| Paper component | What the paper specifies | Our implementation | Deviation and reason |
|---|---|---|---|
| Edge function φ | w_b·silu(x)+w_s·Σc_iB_i(x) (2.10–2.12) | `KANLayer.forward` in `src/kan.py` | none |
| B-spline basis | order k=3 on G intervals | `b_spline_basis` (Cox–de Boor), tested against SciPy | none |
| Layer / network | (2.5)–(2.7) | `KANLayer`, `KAN` | none |
| Init | w_s=1, c~N(0,0.1²), w_b Xavier | same | none |
| Grid update | follows input activations | `KANLayer.update_grid` | rule unspecified → quantile/uniform blend |
| Grid extension | LSQ refit (2.16) | `update_grid(x, G_new)` using `torch.linalg.lstsq` | LSQ over the training samples |
| Optimiser | LBFGS | `torch.optim.LBFGS`, strong-Wolfe | 100 steps/grid, not 200 |
| Grid sizes | up to G=1000 | up to G=100 | CPU time |
| Regularisation/pruning/symbolic | Sec. 2.5 | not implemented | not needed for this experiment |
| Baseline | MLP, depth 2–5 | `MLP`, depth 2–4, SiLU | smaller sweep |
