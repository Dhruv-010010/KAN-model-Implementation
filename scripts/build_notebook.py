"""Assemble notebooks/KAN_paper_implementation.ipynb. Core-code cells are copied verbatim from src/*.py
(split at '# %% [name]' markers) so the notebook and the tested modules cannot diverge."""
import json, os, re, sys, yaml, nbformat
from nbformat.v4 import new_notebook, new_markdown_cell as MD, new_code_cell as CODE

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = yaml.safe_load(open(f"{root}/configs/experiment.yaml"))


def src_cells(module):
    """Return {cell_name: code} for a src module; drop relative imports tagged '# nb:skip' and the docstring."""
    text = open(f"{root}/src/{module}.py").read()
    parts = re.split(r"^# %% \[(\w+)\]\n", text, flags=re.M)
    out = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        lines = [l for l in body.rstrip().split("\n") if "# nb:skip" not in l]
        out[name] = "\n".join(lines).strip("\n")
    return out


kan, models, utils, data, evaluate, train, plots, exps, symbolic, feynman, pde, mnist = (src_cells(m) for m in
    ["kan", "models", "utils", "data", "evaluate", "train", "plots", "experiments", "symbolic", "feynman", "pde", "mnist"])
extra = open(f"{root}/scripts/nb_cells.py").read()
exec(extra)   # defines `cells` (list of nbformat cells) using the objects above
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["accelerator"] = "GPU"
os.makedirs(f"{root}/notebooks", exist_ok=True)
nbformat.write(nb, f"{root}/notebooks/KAN_paper_implementation.ipynb")
print("cells:", len(cells), "code:", sum(c.cell_type == 'code' for c in cells))
