# Kolmogorov–Arnold Networks
## From-scratch implementation and reduced-scale reproduction

**Deep Learning 2026 · Paper implementation project**  
Dhruv · 24CS3018 · DL-107

*KAN: Kolmogorov–Arnold Networks* — Liu et al. (2024)

---

# The idea behind KANs

**MLP:** fixed activation functions at nodes; learned weights on edges.  
**KAN:** learned univariate functions on edges; nodes sum incoming values.

For an edge, our implementation uses

$$\phi(x)=w_b\,\mathrm{SiLU}(x)+w_s\sum_i c_i B_i(x)$$

where $B_i$ are Cox–de Boor B-spline basis functions.

**Core contribution rebuilt:** spline basis, KAN layer/network, adaptive grid update, and least-squares grid extension — in PyTorch, without a KAN library.

---

# What we set out to reproduce

**Paper showcase:** $f(x,y)=\exp(\sin(\pi x)+y^2)$

- KAN shapes [2,1,1] and [2,5,1]; spline order $k=3$
- 1,000 train · 1,000 validation · 10,000 test samples
- Grid sizes $G\in\{3,5,10,20,50,100\}$; 100 LBFGS steps per grid
- Compare with parameter-matched MLPs; test grid extension and spline order by ablation

**Reduced-scale choices:** the paper goes to $G=1000$ and uses 200 steps per grid. We stop at $G=100$ and use 100 steps per grid; these are not exact reproductions.

---

# Reproduction: strong fit, variable across seeds

![Scaling and errors across grid sizes](figures/scaling_vs_G.png)

On the showcased function, the selected [2,1,1] KAN reaches **test RMSE $1.8\times10^{-5}$** at $G=100$ (315 parameters). The best parameter-matched MLP reaches **$1.2\times10^{-3}$** (501 parameters).

**Important qualification:** only one of four KAN seeds reaches this accuracy; median KAN error at $G=100$ is about $3.4\times10^{-3}$. The selected-seed scaling slope is about 3.15 for $G\le20$, but 2.02 over the full range.

---

# What the ablations tell us

![Grid extension ablation](figures/ablation_extension.png)

**Grid extension vs. restarting:** extension makes large grids trainable more reliably, but can carry early failures forward. From-scratch training wins more often at small grids; at $G=100$, extension succeeds on 3/4 seeds and scratch on 0/4.

**Spline order:** higher order lowers the best error in this run. For $k=1,2,3$, best test RMSE at $G=50$ is approximately $9.8\times10^{-4}$, $1.4\times10^{-4}$, and $2.4\times10^{-5}$. Measured slopes remain below the theoretical $k+1$.

---

# Assignment extensions: mixed results

**Feynman symbolic regression:** equal-parameter KANs beat MLPs on 3 of 6 equations, tie on 1, and lose on 2. Automatic simplification did **not** recover the target formulas in these runs.

**Poisson PDE:** KAN error decreases as grid size grows; at the largest size it modestly beats the closest MLP (relative $L^2$: $6.4\times10^{-4}$ vs. $7.4\times10^{-4}$).

**MNIST:** equal-size MLPs outperform these KAN settings (about 97.2–97.5% vs. 92.3–94.1% accuracy) and train about 4–5× faster.

*These extensions use one configuration/seed in small-scale studies; interpret as initial evidence, not a broad benchmark.*

---

# Takeaways and discussion

- Rebuilt and tested the defining KAN components; **20 tests passed, 1 CUDA-only test skipped** on the available machine.
- Grid extension and splines fit a smooth compositional target very well in the best seed.
- Results depend strongly on seed; scaling falls short of the paper’s full-range behavior.
- Beyond the showcase function, KANs do not consistently beat matched MLPs; symbolic recovery remains an open weakness here.

**Questions?**  
Can more seeds or better optimization make the result reliable? When is an interpretable edge function worth extra training cost?

**Code, notebook, results and report:** repository `KAN` · `notebooks/KAN_paper_implementation.ipynb` · `reports/report.pdf`

