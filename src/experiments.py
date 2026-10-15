"""Experiment drivers shared by scripts/ and the notebook (results are saved as CSV/JSON)."""

# %% [experiments]
import os
import time
import pandas as pd
import torch
from .data import make_dataset                                   # nb:skip
from .models import MLP                                          # nb:skip
from .train import train_kan_with_extension, train_mlp           # nb:skip
from .utils import set_seed, save_json                           # nb:skip


def build_data(cfg, smoke=False):
    D = cfg["data"]
    if smoke:
        return make_dataset(cfg["target"], cfg["smoke"]["n_train"], 200, cfg["smoke"]["n_test"], seed=D["data_seed"])
    return make_dataset(cfg["target"], D["n_train"], D["n_val"], D["n_test"], seed=D["data_seed"])


def run_main(cfg, out, data=None, smoke=False, log=print):
    """KAN [2,1,1] and [2,5,1] with grid extension + MLP baselines -> out/kan_records.csv, kan_curves.json, mlp_records.csv"""
    os.makedirs(out, exist_ok=True)
    data = data or build_data(cfg, smoke)
    K = dict(cfg["kan"])
    if smoke:
        K.update(grids=cfg["smoke"]["grids"], steps_per_grid=cfg["smoke"]["steps_per_grid"], seeds=[0])
    rows, curves = [], {}
    for name, width, seeds in (("KAN[2,1,1]", K["width"], K["seeds"]),
                               ("KAN[2,5,1]", cfg["kan_wide"]["width"], cfg["kan_wide"]["seeds"])):
        for seed in seeds[: (1 if smoke else None)]:
            set_seed(seed); t = time.time()
            rec, cur, _ = train_kan_with_extension(width, data, grids=K["grids"], steps_per_grid=K["steps_per_grid"],
                                                   k=K["k"], extend=True, seed=seed,
                                                   grid_update_steps=tuple(K["grid_update_steps"]))
            rows += [dict(model=name, seed=seed, **r) for r in rec]
            curves[f"{name}_seed{seed}"] = cur
            log(f"{name} seed {seed}: {time.time()-t:.0f}s, final test RMSE {rec[-1]['test_rmse']:.3e}")
            pd.DataFrame(rows).to_csv(f"{out}/kan_records.csv", index=False)
            save_json(curves, f"{out}/kan_curves.json")
    if not smoke:
        run_mlp(cfg, out, data=data, log=log)


def run_mlp(cfg, out, data=None, seeds=None, fname="mlp_records.csv", log=print):
    """MLP baselines (depth/width sweep x seeds) trained with the same optimiser, data and step budget."""
    os.makedirs(out, exist_ok=True)
    data = data or build_data(cfg)
    rows = []
    for c in cfg["mlp"]["configs"]:
        for w in c["widths"]:
            for seed in (seeds if seeds is not None else cfg["mlp"]["seeds"]):
                set_seed(seed)
                m = MLP([2] + [w] * (c["depth"] - 1) + [1]).double()
                rec, _ = train_mlp(m, data, steps=cfg["mlp"]["steps"])
                rows.append(dict(model=f"MLP(depth {c['depth']})", depth=c["depth"], width=w, seed=seed, **rec))
                log(f"MLP depth {c['depth']} width {w} seed {seed}: test RMSE {rec['test_rmse']:.3e}")
                pd.DataFrame(rows).to_csv(f"{out}/{fname}", index=False)


def run_ablation(cfg, out, data=None, smoke=False, parts="AB", log=print):
    """A: grid extension vs from-scratch; B: spline order k -> out/ablation_extension.csv, ablation_order.csv"""
    os.makedirs(out, exist_ok=True)
    K, A = cfg["kan"], cfg["ablation"]
    grids, steps, seeds, ograds = K["grids"], K["steps_per_grid"], A["seeds"], A["order_grids"]
    if smoke:
        grids, steps, seeds, ograds = [3, 5], 10, [0], [3, 5]
    data = data or build_data(cfg, smoke)
    gu = tuple(K["grid_update_steps"])

    def run(tag, rows, **kw):
        for seed in seeds:
            set_seed(seed); t = time.time()
            rec, _, _ = train_kan_with_extension(K["width"], data, seed=seed, grid_update_steps=gu, **kw)
            rows.extend(dict(variant=tag, seed=seed, **r) for r in rec)
            log(f"{tag} seed {seed}: {time.time()-t:.0f}s, final test RMSE {rec[-1]['test_rmse']:.3e}")

    if "A" in parts:
        rows = []
        run("from_scratch", rows, grids=grids, steps_per_grid=steps, k=K["k"], extend=False)
        pd.DataFrame(rows).to_csv(f"{out}/ablation_extension.csv", index=False)
    if "B" in parts:
        rows = []
        for k in A["orders"]:
            run(f"k={k}", rows, grids=ograds, steps_per_grid=steps, k=k, extend=True)
            pd.DataFrame(rows).to_csv(f"{out}/ablation_order.csv", index=False)
