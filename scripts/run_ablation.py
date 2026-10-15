"""Ablations: (A) grid extension vs training from scratch at each G; (B) B-spline order k.
Usage: python scripts/run_ablation.py [--smoke]"""
import argparse, os, sys, time
import pandas as pd, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import make_dataset
from src.train import train_kan_with_extension
from src.utils import set_seed

ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--part", default="AB"); a = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = yaml.safe_load(open(os.path.join(root, "configs/experiment.yaml")))
torch.set_num_threads(1)
out = os.path.join(root, "results" + ("/smoke" if a.smoke else "")); os.makedirs(out, exist_ok=True)
D, K, A = cfg["data"], cfg["kan"], cfg["ablation"]
grids, steps, seeds = K["grids"], K["steps_per_grid"], A["seeds"]
if a.smoke:
    D = dict(D, n_train=200, n_val=200, n_test=500); grids, steps, seeds = [3, 5], 10, [0]
data = make_dataset(cfg["target"], D["n_train"], D["n_val"], D["n_test"], seed=D["data_seed"])
gu = tuple(K["grid_update_steps"])
rows = []
def run(tag, **kw):
    for seed in seeds:
        set_seed(seed); t = time.time()
        rec, _, _ = train_kan_with_extension(K["width"], data, seed=seed, grid_update_steps=gu, **kw)
        rows.extend(dict(variant=tag, seed=seed, **r) for r in rec)
        print(tag, "seed", seed, round(time.time() - t), "s final test", rec[-1]["test_rmse"], flush=True)
if "A" in a.part:   # A: extend=False -> fresh KAN at every grid size (extend=True results live in kan_records.csv)
    run("from_scratch", grids=grids, steps_per_grid=steps, k=K["k"], extend=False)
    pd.DataFrame(rows).to_csv(f"{out}/ablation_extension.csv", index=False); rows = []
if "B" in a.part:   # B: spline order
    for k in A["orders"]:
        run(f"k={k}", grids=A["order_grids"] if not a.smoke else grids, steps_per_grid=steps, k=k, extend=True)
        pd.DataFrame(rows).to_csv(f"{out}/ablation_order.csv", index=False)
print("finished")
