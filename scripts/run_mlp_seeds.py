"""Extra MLP seeds (appends to results/mlp_records.csv). Usage: python scripts/run_mlp_seeds.py 1 2"""
import os, sys, yaml, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.experiments import run_mlp
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = yaml.safe_load(open(f"{root}/configs/experiment.yaml")); torch.set_num_threads(1)
seeds = [int(a) for a in sys.argv[1:]]
run_mlp(cfg, f"{root}/results", seeds=seeds, fname="mlp_part_" + "_".join(map(str, seeds)) + ".csv")
