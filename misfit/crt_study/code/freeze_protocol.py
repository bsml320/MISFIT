"""Write the protocol freeze record: timestamp, sha256 of every code and
protocol file, cohort manifest digests, and the git commit if available.
Scientific stages are gated on the file this writes."""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misfit.crt_study.code import cohorts

STUDY = _resolve_path("<REPO_ROOT>/misfit/crt_study")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": {}, "digests": {}}
for d in ("code", "protocol", "tests"):
    for f in sorted(os.listdir(f"{STUDY}/{d}")):
        p = f"{STUDY}/{d}/{f}"
        if os.path.isfile(p) and not f.startswith("._") and not f.endswith(".pyc") and f != "PROTOCOL_FROZEN.txt":
            rec["files"][f"{d}/{f}"] = sha(p)
for cn in ("oasis3", "adni", "oasis1"):
    spec = cohorts.LOADERS[cn]()
    rec["digests"][cn] = cohorts.manifest_digest(cohorts.split_probes(spec))
try:
    rec["git"] = subprocess.run(
        ["git", "-C", _resolve_path("<REPO_ROOT>/misfit"), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
except Exception:
    rec["git"] = "unavailable"
import torch
rec["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                   "cuda": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version()}
out = f"{STUDY}/protocol/PROTOCOL_FROZEN.txt"
with open(out, "w") as f:
    f.write(json.dumps(rec, indent=2))
print(f"frozen at {rec['ts']}; {len(rec['files'])} files recorded; git {rec['git'][:12]}")
