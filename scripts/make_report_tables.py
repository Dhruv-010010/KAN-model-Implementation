"""Generate LaTeX tables for report/report.tex from results/*.csv (no numbers are typed by hand).
Usage: python scripts/make_report_tables.py [--results results] [--out report]
Part-A tables are always written. tables_bd.tex (Feynman/PDE/MNIST) is filled from
feynman_results.csv / pde_results.csv / mnist_results.csv when they exist (written by notebook Parts B-D);
otherwise it contains clearly marked 'pending' rows."""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--out", default="report")
a = ap.parse_args(); R, O = a.results, a.out
os.makedirs(O, exist_ok=True)

def esc(s):
    for k, v in {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$"}.items():
        s = str(s).replace(k, v)
    return s
def sci(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return "--"
    m, e = f"{x:.1e}".split("e"); return f"${m}\\!\\times\\!10^{{{int(e)}}}$"
def table(cap, label, head, rows, spec, star=False, pos="t"):
    env = "table*" if star else "table"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (f"\\begin{{{env}}}[{pos}]\n\\caption{{{cap}}}\\label{{{label}}}\n\\centering\\footnotesize\n\\begin{{tabular}}{{{spec}}}\n\\toprule\n"
            f"{head} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")

kan = pd.read_csv(f"{R}/kan_records.csv"); k1 = kan[kan.model == "KAN[2,1,1]"]
best_seed = int(k1[k1.G == 100].sort_values("val_rmse").iloc[0].seed)           # selected on validation, as in the notebook
abl = pd.read_csv(f"{R}/ablation_extension.csv"); mlp = pd.read_csv(f"{R}/mlp_records.csv")
order = pd.read_csv(f"{R}/ablation_order.csv"); osel = pd.read_csv(f"{R}/ablation_order_selected.csv")
matched = pd.read_csv(f"{R}/kan_vs_mlp_matched.csv")
out = []

# Table I: KAN [2,1,1] per grid
rows = []
for G, d in k1.groupby("G"):
    s = d[d.seed == best_seed].iloc[0]
    rows.append([str(G), str(int(s.params)), sci(s.train_rmse), sci(s.val_rmse), sci(s.test_rmse), sci(d.test_rmse.median()), f"{int((d.test_rmse < 1e-2).sum())}/{len(d)}"])
out.append(table(f"KAN [2,1,1], $k{{=}}3$. Selected seed ({best_seed}) chosen on final validation RMSE; median and success count (test RMSE $<10^{{-2}}$) are over {k1.seed.nunique()} seeds.",
                 "tab:grid", "$G$ & params & train & val & test (sel.) & test (median) & seeds ok", rows, "rrrrrrr"))
# Table II: matched MLP
rows = [[esc(r.MLP.replace("MLP(depth ", "depth ").replace("), width ", ", w=")), str(int(r["MLP params"])), sci(r["MLP test RMSE"]),
         esc(r["KAN params (G)"]), sci(r["KAN test RMSE"]), sci(r["KAN median over seeds"])] for _, r in matched.iterrows()]
out.append(table("Equal-parameter comparison (test RMSE). MLP: best of 4 seeds by validation, 600 LBFGS steps; KAN: selected seed, grid with the closest parameter count, and median over seeds.",
                 "tab:mlp", "MLP & params & test & KAN params (G) & test (sel.) & test (median)", rows, "lrrrrr"))
# Table III: ablation A
ext = k1.copy(); ext["variant"] = "extension"; both = pd.concat([ext, abl])
rows = []
for G in sorted(both.G.unique()):
    e, s = both[(both.variant == "extension") & (both.G == G)], both[(both.variant == "from_scratch") & (both.G == G)]
    rows.append([str(G), sci(e.test_rmse.min()), sci(e.test_rmse.median()), f"{int((e.test_rmse < 1e-2).sum())}/{len(e)}",
                 sci(s.test_rmse.min()), sci(s.test_rmse.median()), f"{int((s.test_rmse < 1e-2).sum())}/{len(s)}"])
out.append(table("Ablation A, test RMSE over seeds: grid extension vs re-initialising at every $G$ (same steps, data, seeds).",
                 "tab:ablA", "$G$ & ext.\\ best & ext.\\ median & ok & scratch best & scratch median & ok", rows, "rrrrrrr"))
# Table IV: ablation B
rows = []
for _, r in osel.iterrows():
    d = order[order.variant == r.order]
    rows.append([esc(r.order), f"{r.alpha:.2f}", str(int(r["theory alpha (k+1)"])), str(int(r.G_star_by_val)), sci(r.best_test_rmse), f"{int(r['seeds with test RMSE<1e-2'])}/{d.seed.nunique()}"])
out.append(table("Ablation B: spline order. $\\alpha$ is the log--log slope of test RMSE vs.\\ $G$ for the validation-selected seed up to the validation-optimal $G^\\star$.",
                 "tab:ablB", "order & $\\alpha$ & $k{+}1$ & $G^\\star$ & best test & seeds ok", rows, "lrrrrr"))
open(f"{O}/tables_a.tex", "w").write("\n".join(out))

# ----------------- Parts B-D (filled only when the notebook has produced the CSVs) -----------------
def read(name):
    p = f"{R}/{name}"; return pd.read_csv(p) if os.path.exists(p) else None
fy, pd_, mn = read("feynman_results.csv"), read("pde_results.csv"), read("mnist_results.csv")
PEND = r"\textit{pending}"
t = []
if fy is not None:
    rows = [[esc(r.equation), sci(r.human_kan_rmse), sci(r.mlp_eq_human_rmse), sci(r.get("pruned_kan_rmse", np.nan)), sci(r.get("mlp_eq_pruned_rmse", np.nan)), sci(r.get("symbolic_rmse", np.nan))] for _, r in fy.iterrows()]
else:
    rows = [[PEND] + ["--"] * 5]
t.append(table("Feynman equations, test RMSE: human-constructed KAN, equal-parameter MLP, pruned KAN, MLP with the pruned KAN's parameters, and the extracted symbolic formula.",
               "tab:feyn", "equation & KAN & MLP & pruned KAN & MLP (pruned) & symbolic", rows, "lrrrrr", pos="!htb"))
if pd_ is not None:
    rows = [[esc(r.model), str(int(r.params)), sci(r.rel_l2), f"{r.seconds:.0f}"] for _, r in pd_.iterrows()]
else:
    rows = [[PEND, "--", "--", "--"]]
t.append(table("Poisson PINN: relative $L^2$ error against the exact solution and training time (s).", "tab:pde", "model & params & rel.\\ $L^2$ & s", rows, "lrrr", pos="!htb"))
if mn is not None:
    rows = [[esc(r.model.split(" (")[0]), str(int(r.params)), f"{100 * r.test_acc:.2f}", f"{r.seconds:.0f}"] for _, r in mn.iterrows()]
else:
    rows = [[PEND, "--", "--", "--"]]
t.append(table("MNIST (single seed, no augmentation): test accuracy (\\%) and training time (s) at equal parameter counts.", "tab:mnist", "model & params & test acc. & s", rows, "lrrr", pos="!htb"))
open(f"{O}/tables_bd.tex", "w").write("\n".join(t))
print("best seed", best_seed, "| B-D tables:", "filled" if all(x is not None for x in (fy, pd_, mn)) else f"partly/not filled (feynman {fy is not None}, pde {pd_ is not None}, mnist {mn is not None})")
