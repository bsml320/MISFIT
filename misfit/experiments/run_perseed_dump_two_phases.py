"""Per-seed canonical misfit/error dump for AT-S15 and AT-S16.

Runs the canonical IOP collect() per seed on ADNI and OASIS-1 for two disjoint seed
sets: phase A = the paper's seeds (42-46) and phase B = fresh seeds (100-104) that
serve as independently repeated later evaluations for the AT-S15 audit endpoint.
Saves per-seed misfit and binary-error arrays plus site ids. GPU 2.
Output: results/perseed_misfit_two_phases.npz
"""

from misfit.paths import resolve_path as _resolve_path
import os
import sys

# CUDA visibility is selected by the caller.
import numpy as np  # noqa: E402

sys.path.insert(0, _resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_clinical_stratified_media import collect  # noqa: E402
from misfit.experiments.run_honest_estimator_table import (  # noqa: E402
    load_adni, load_oasis_ds)

SEEDS_A = [42, 43, 44, 45, 46]
SEEDS_B = [100, 101, 102, 103, 104]
OUT = _resolve_path("<REPO_ROOT>/misfit/results/perseed_misfit_two_phases.npz")

out = {}
for name, loader in [("adni", load_adni), ("oasis", load_oasis_ds)]:
    f, t, s = loader()
    out[f"{name}_site"] = s.numpy()
    out[f"{name}_label"] = t.numpy().astype(int)
    for phase, seeds in (("A", SEEDS_A), ("B", SEEDS_B)):
        mis, err = [], []
        for sd in seeds:
            c = collect(f, t, s, sd)
            mis.append(c["misfit"])
            err.append(c["err"])
            print(f"[{name}/{phase}] seed {sd} done", flush=True)
        out[f"{name}_mis_{phase}"] = np.stack(mis)
        out[f"{name}_err_{phase}"] = np.stack(err)
np.savez(OUT, **out)
print("wrote", OUT, flush=True)
