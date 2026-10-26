"""TASK 7: Figure integrity. Re-hash (sha256, verifier's own hashing) every
file listed in provenance/figure_checksums_before.json and confirm all are
unchanged on disk right now."""

from misfit.paths import resolve_path as _resolve_path
import hashlib
import json
import os

COR = _resolve_path("<REPO_ROOT>/misfit/correction_20261018")
BASE = _resolve_path("<REPO_ROOT>")

man = json.load(open(f"{COR}/provenance/figure_checksums_before.json"))
out = {"task": 7, "n_listed": len(man), "mismatches": [], "missing": []}
n_ok = 0
for rel, want in man.items():
    p = os.path.join(BASE, rel)
    if not os.path.exists(p):
        out["missing"].append(rel)
        continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got == want:
        n_ok += 1
    else:
        out["mismatches"].append({"file": rel, "expected": want, "got": got})
out["n_match"] = n_ok
out["PASS"] = bool(n_ok == len(man) and not out["missing"])

# cross-check against the package's own 'after' manifest
after = json.load(open(f"{COR}/provenance/figure_checksums_after.json"))
out["before_after_manifests_identical"] = bool(man == after)

print(json.dumps(out, indent=2))
json.dump(out, open(f"{COR}/verify/out_task7.json", "w"), indent=2)
