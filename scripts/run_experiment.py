"""Main reduced-scale reproduction: KAN [2,1,1] / [2,5,1] with grid extension + MLP baselines.
Usage: python scripts/run_experiment.py [--smoke]"""
import argparse, os, sys, time
import pandas as pd, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import make_dataset
from src.models import MLP
from src.train import train_kan_with_extension, train_mlp
from src.utils import set_seed, save_json

ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = yaml.safe_load(open(os.path.join(root, "configs/experiment.yaml")))
torch.set_num_threads(1)
out = os.path.join(root, "results" + ("/smoke" if a.smoke else "")); os.makedirs(out, exist_ok=True)
D, K = cfg["data"], cfg["kan"]
if a.smoke:
    D = dict(D, n_train=cfg["smoke"]["n_train"], n_test=cfg["smoke"]["n_test"], n_val=200)
    K = dict(K, grids=cfg["smoke"]["grids"], steps_per_grid=cfg["smoke"]["steps_per_grid"], seeds=[0])
data = make_dataset(cfg["target"], D["n_train"], D["n_val"], D["n_test"], seed=D["data_seed"])
rows, curves = [], {}
for name, width, seeds in (("KAN[2,1,1]", K["width"], K["seeds"]), ("KAN[2,5,1]", cfg["kan_wide"]["width"], cfg["kan_wide"]["seeds"])):
    for seed in seeds[: (1 if a.smoke else None)]:
        set_seed(seed); t = time.time()
        rec, cur, _ = train_kan_with_extension(width, data, grids=K["grids"], steps_per_grid=K["steps_per_grid"], k=K["k"],
                                               extend=True, seed=seed, grid_update_steps=tuple(K["grid_update_steps"]))
        rows += [dict(model=name, seed=seed, **r) for r in rec]
        curves[f"{name}_seed{seed}"] = cur
        print(name, "seed", seed, "done in", round(time.time() - t), "s; final test", rec[-1]["test_rmse"], flush=True)
        pd.DataFrame(rows).to_csv(f"{out}/kan_records.csv", index=False)
        save_json(curves, f"{out}/kan_curves.json")
mlp_rows = []
if not a.smoke:
    for c in cfg["mlp"]["configs"]:
        for w in c["widths"]:
            for seed in cfg["mlp"]["seeds"]:
                set_seed(seed); torch.manual_seed(seed)
                m = MLP([2] + [w] * (c["depth"] - 1) + [1]).double()
                rec, _ = train_mlp(m, data, steps=cfg["mlp"]["steps"])
                mlp_rows.append(dict(model=f"MLP(depth {c['depth']})", depth=c["depth"], width=w, seed=seed, **rec))
                print("MLP", c["depth"], w, rec["test_rmse"], flush=True)
                pd.DataFrame(mlp_rows).to_csv(f"{out}/mlp_records.csv", index=False)
print("finished")
