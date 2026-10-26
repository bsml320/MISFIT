"""Route-B SPM25 DARTEL VBM for OASIS-3 (MedIA disease-matched external cohort).

Same recipe as ADNI/AIBL (segment -> DARTEL -> MNI-norm modulated 1.5mm -> 3mm smooth ->
smwc1pT1.nii), but at 2,720 sessions we build the DARTEL template on a stratified SUBSET
(scanner x label) and warp the rest into it -- exact joint DARTEL over 2.7k is days of
single-threaded compute; subset-template + parallel warp is the standard scalable path.

Stages (each resumable):
    pilot [N]   end-to-end on N sessions (segment+template+norm), verify smwc1 geometry
    segment     segment ALL sessions (parallel)                       -> rc1/rc2/c1 per wd
    template    DARTEL template on the stratified subset              -> Template_0..6 + flows
    warp        warp the non-subset sessions into that template       -> flow field per wd
    norm        MNI-normalise+smooth ALL (flowfield,c1)               -> smwc1pT1.nii per wd
    features    assemble smwc1 -> gray-matter mask -> PCA-50          -> oasis3_features.npz

Usage:  python run_route_b_oasis3.py pilot 3
        python run_route_b_oasis3.py segment            # long; run in tmux/background
        python run_route_b_oasis3.py template
        python run_route_b_oasis3.py warp
        python run_route_b_oasis3.py norm
        python run_route_b_oasis3.py features
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vbm_pipeline import (  # noqa: E402
    dartel_create_template, dartel_mni_norm, dartel_warp_to_existing, segment_many, segment_one,
)

LIST = Path(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_t1w_list.csv"))
# WORK / subset seed / feature output are env-overridable so a template-stability replicate can run
# in a separate work tree without clobbering the primary results.
WORK = Path(os.environ.get("OASIS3_WORK", _resolve_path("<REPO_ROOT>/data/oasis3/route_b_work")))
PRIMARY_WORK = Path(_resolve_path("<REPO_ROOT>/data/oasis3/route_b_work"))   # holds the segmentation
FEAT_OUT = Path(os.environ.get("OASIS3_FEAT_OUT", _resolve_path("<REPO_ROOT>/data/oasis3/oasis3_features.npz")))
SUBSET_SEED = int(os.environ.get("OASIS3_SUBSET_SEED", "0"))
SUBSET_PER_CELL = 30          # subjects per (scanner x label) cell for the template (~300 total)
JOBS = 14                     # parallel Segment / warp processes (CPU)


def stage_prep() -> None:
    """Copy the (template-independent) segmentation rc1/rc2/c1 from the primary work tree into
    WORK, so a stability replicate reuses segmentation and only redoes template/warp/norm."""
    import shutil
    df = load()
    n = 0
    for s in df["session"]:
        d = WORK / s
        d.mkdir(parents=True, exist_ok=True)
        for fn in ("rc1pT1.nii", "rc2pT1.nii", "c1pT1.nii"):
            src = PRIMARY_WORK / s / fn
            if src.exists() and not (d / fn).exists():
                shutil.copy(src, d / fn)
        n += 1
        if n % 500 == 0:
            print(f"[prep] {n}/{len(df)} sessions copied", flush=True)
    print(f"[prep] segmentation copied for {n} sessions into {WORK}")


def load() -> pd.DataFrame:
    df = pd.read_csv(LIST)
    df["wd"] = df["session"].map(lambda s: str(WORK / s))
    return df


def wd_of(row) -> Path:
    return WORK / row.session


def prep_t1(t1: str, wd: Path) -> Path:
    """OASIS-3 T1w are .nii.gz; SPM standalone needs an uncompressed .nii. Decompress once
    into the workdir (idempotent) and return the .nii path to feed Segment."""
    import gzip
    import shutil
    wd.mkdir(parents=True, exist_ok=True)
    src = Path(t1)
    if src.suffix == ".gz":
        dst = wd / "pT1_src.nii"
        if not dst.exists():
            with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        return dst
    return src


def stratified_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Sample up to SUBSET_PER_CELL sessions per (scanner model, label) cell, seed SUBSET_SEED."""
    keep = []
    for (_, _), g in df.groupby(["model", "label"]):
        keep.append(g.sample(min(len(g), SUBSET_PER_CELL), random_state=SUBSET_SEED))
    return pd.concat(keep).drop_duplicates("session")


# --------------------------------------------------------------------------- pilot
def pilot(n: int) -> None:
    """End-to-end check on the first n sessions, then inspect one smwc1 header.

    Verifies output geometry, voxel size and descrip string against the
    recovered original recipe before the full cohort is committed.
    """
    df = load().head(n)
    items = [(prep_t1(r.t1w, wd_of(r)), wd_of(r)) for r in df.itertuples()]
    print(f"[pilot] segmenting {n} ...", flush=True)
    seg = segment_many(items, jobs=min(JOBS, n))
    ok = [s for s in seg if s["ok"]]
    print(f"[pilot] segmented ok: {len(ok)}/{n}")
    if len(ok) < 2:
        print("[pilot] too few segmented; aborting.", [s.get("err", "")[:200] for s in seg]); return
    print("[pilot] DARTEL template on the pilot set ...", flush=True)
    log = dartel_create_template([s["rc1"] for s in ok], [s["rc2"] for s in ok], str(WORK / "_pilot_lists"))
    print("  ", log.strip().splitlines()[-1] if log.strip() else "(no log)")
    t6 = Path(ok[0]["rc1"]).parent / "Template_6.nii"
    subj = [(str(Path(s["rc1"]).parent / "u_rc1pT1_Template.nii"), s["c1"]) for s in ok]
    print("[pilot] MNI-normalise + smooth ...", flush=True)
    log = dartel_mni_norm(str(t6), subj)
    print("  ", log.strip().splitlines()[-1] if log.strip() else "(no log)")
    # verify one smwc1
    import nibabel as nib
    sm = Path(ok[0]["c1"]).parent / "smwc1pT1.nii"
    if sm.exists():
        img = nib.load(str(sm)); d = img.get_fdata()
        vox = img.header.get_zooms()[:3]
        print(f"\n[pilot] smwc1: shape={img.shape} vox={tuple(round(v,2) for v in vox)} "
              f"descrip={img.header['descrip'].tobytes().decode(errors='ignore').strip(chr(0))!r} "
              f"mean(>0)={d[d>0].mean():.3f}")
        print("[pilot] EXPECT shape=(121,145,121) vox=(1.5,1.5,1.5) descrip~'Smoothed (3x3x3) Dartel normed' "
              "mean~0.20 -> if so, recipe is valid on OASIS-3; run 'segment' next.")
    else:
        print("[pilot] smwc1 NOT produced -- check logs before scaling up.")


# --------------------------------------------------------------------------- full stages
def stage_segment() -> None:
    """Segment every session without an rc1 yet (resumable, JOBS parallel)."""
    df = load()
    todo = []
    for r in df.itertuples():
        w = wd_of(r)
        if not (w / "rc1pT1.nii").exists():
            todo.append((prep_t1(r.t1w, w), w))
    ntot = len(df)
    print(f"[segment] {len(todo)}/{ntot} to segment (rest already done), jobs={JOBS}", flush=True)
    seg = segment_many(todo, jobs=JOBS)
    fail = [s for s in seg if not s["ok"]]
    done = sum((WORK / s / "rc1pT1.nii").exists() for s in df["session"])
    print(f"[segment] complete this pass: {len(seg)-len(fail)} ok, {len(fail)} failed; "
          f"total segmented {done}/{ntot}")


def stage_template() -> None:
    """Build the DARTEL template on the stratified subset; record the subset sessions."""
    df = load()
    sub = stratified_subset(df)
    sub = sub[sub["session"].map(lambda s: (WORK / s / "rc1pT1.nii").exists())]
    print(f"[template] building DARTEL template on {len(sub)} stratified subset subjects "
          f"(cells: model x label) ...", flush=True)
    rc1s = [str(WORK / s / "rc1pT1.nii") for s in sub["session"]]
    rc2s = [str(WORK / s / "rc2pT1.nii") for s in sub["session"]]
    (WORK / "_meta").mkdir(parents=True, exist_ok=True)
    (WORK / "_meta" / "subset_sessions.txt").write_text("\n".join(sub["session"]) + "\n")
    log = dartel_create_template(rc1s, rc2s, str(WORK / "_template_lists"))
    print("  ", log.strip().splitlines()[-1] if log.strip() else "(no log)")
    print(f"[template] Template_0..6 written next to {rc1s[0]}")


def stage_warp() -> None:
    """Warp non-subset sessions into the subset template (resumable, chunked, parallel)."""
    import glob
    from concurrent.futures import ThreadPoolExecutor, as_completed
    df = load()
    subset = set((WORK / "_meta" / "subset_sessions.txt").read_text().split())
    tdir = str(Path(glob.glob(str(WORK / "*" / "Template_6.nii"))[0]).parent)
    rest = df[~df["session"].isin(subset)]
    rest = rest[rest["session"].map(lambda s: (WORK / s / "rc1pT1.nii").exists()
                                    and not (WORK / s / "u_rc1pT1.nii").exists())]
    sess = list(rest["session"])
    CH = 8
    chunks = [sess[i:i + CH] for i in range(0, len(sess), CH)]
    print(f"[warp] warping {len(sess)} non-subset sessions into template @ {tdir} "
          f"in {len(chunks)} chunks x{CH}, {JOBS} parallel ...", flush=True)

    def do_chunk(grp):
        rc1s = [str(WORK / s / "rc1pT1.nii") for s in grp]
        rc2s = [str(WORK / s / "rc2pT1.nii") for s in grp]
        return "WARP1_DONE" in dartel_warp_to_existing(rc1s, rc2s, tdir)

    done = fail = 0
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        futs = {ex.submit(do_chunk, g): g for g in chunks}
        for i, fut in enumerate(as_completed(futs), 1):
            ok = False
            try:
                ok = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  chunk error: {e}", flush=True)
            done += ok
            fail += not ok
            if i % 10 == 0:
                nf = sum((WORK / s / "u_rc1pT1.nii").exists() for s in sess)
                print(f"  ...{i}/{len(chunks)} chunks ({done} ok, {fail} failed); "
                      f"flow fields written: {nf}/{len(sess)}", flush=True)
    nf = sum((WORK / s / "u_rc1pT1.nii").exists() for s in sess)
    print(f"[warp] complete: {nf}/{len(sess)} non-subset flow fields written "
          f"({fail} chunks reported failure -- re-run 'warp' to retry).")


def _flowfield(s) -> Path | None:
    """Subset subjects (create-template) -> u_rc1pT1_Template.nii; non-subset (warp1) -> u_rc1pT1.nii."""
    a = WORK / s / "u_rc1pT1_Template.nii"
    b = WORK / s / "u_rc1pT1.nii"
    return a if a.exists() else (b if b.exists() else None)


def stage_norm() -> None:
    """MNI-normalise and smooth every session with a flow field but no smwc1 yet."""
    df = load()
    subj = []
    for s in df["session"]:
        ff = _flowfield(s); c1 = WORK / s / "c1pT1.nii"
        sm = WORK / s / "smwc1pT1.nii"
        if ff is not None and c1.exists() and not sm.exists():
            subj.append((str(ff), str(c1)))
    import glob
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t6 = glob.glob(str(WORK / "*" / "Template_6.nii"))[0]
    CH = 20
    chunks = [subj[i:i + CH] for i in range(0, len(subj), CH)]
    print(f"[norm] normalising+smoothing {len(subj)} sessions in {len(chunks)} chunks x{CH}, "
          f"{JOBS} parallel -> smwc1pT1.nii ...", flush=True)

    def do_chunk(grp):
        return "NORM_DONE" in dartel_mni_norm(t6, grp)

    done = fail = 0
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        futs = {ex.submit(do_chunk, g): g for g in chunks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                ok = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  chunk error: {e}", flush=True); ok = False
            done += ok; fail += not ok
            if i % 10 == 0:
                ns = sum((WORK / s / "smwc1pT1.nii").exists() for s in df["session"])
                print(f"  ...{i}/{len(chunks)} chunks; smwc1 written: {ns}/{len(df)}", flush=True)
    ns = sum((WORK / s / "smwc1pT1.nii").exists() for s in df["session"])
    print(f"[norm] complete: {ns}/{len(df)} smwc1 written ({fail} chunks failed -- re-run 'norm').")


def stage_features() -> None:
    """Assemble finished smwc1 volumes into the feature npz.

    Cohort mask (mean GM > 0.1), per-voxel standardization, PCA-50 via SVD;
    writes FEAT_OUT with features, session metadata and the voxel count, plus
    a sibling _meta.csv.
    """
    import nibabel as nib
    df = load()
    sms, keep = [], []
    for r in df.itertuples():
        sm = WORK / r.session / "smwc1pT1.nii"
        if sm.exists():
            sms.append(str(sm)); keep.append(r)
    print(f"[features] loading {len(sms)} smwc1 volumes ...", flush=True)
    X = np.stack([nib.load(p).get_fdata(dtype=np.float32).ravel() for p in sms])
    mask = X.mean(0) > 0.1
    Xm = X[:, mask]
    Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-6)
    # PCA-50 via SVD
    U, S, _ = np.linalg.svd(Xm - Xm.mean(0), full_matrices=False)
    feats = (U[:, :50] * S[:50]).astype(np.float32)
    meta = pd.DataFrame(keep)[["session", "subject", "model", "field", "label", "sex", "age"]]
    out = FEAT_OUT
    np.savez(out, features=feats, sessions=meta["session"].values,
             subjects=meta["subject"].values, scanner=meta["model"].values,
             field=meta["field"].values, label=meta["label"].values,
             sex=meta["sex"].values, age=meta["age"].values, n_voxels=int(mask.sum()))
    meta.to_csv(str(FEAT_OUT).replace(".npz", "_meta.csv"), index=False)
    print(f"[features] wrote {out}: {feats.shape[0]} sessions x 50 PCA dims "
          f"({int(mask.sum())} GM voxels). labels {meta['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    WORK.mkdir(parents=True, exist_ok=True)
    if cmd == "prep":
        stage_prep()
    elif cmd == "pilot":
        pilot(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif cmd == "segment":
        stage_segment()
    elif cmd == "template":
        stage_template()
    elif cmd == "warp":
        stage_warp()
    elif cmd == "norm":
        stage_norm()
    elif cmd == "features":
        stage_features()
    else:
        print(f"unknown stage {cmd!r}")
