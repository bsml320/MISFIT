"""SPM25 DARTEL VBM pipeline (route B): reprocess ADNI + AIBL through ONE consistent
recipe so the reliability model can be refit and applied across cohorts without the
lost-original-template problem (see experiments/spm_repro/FINDINGS.md).

Recovered original recipe (from NIfTI descrip 'Smoothed (3x3x3) Dartel normed'):
  Segment (native c1 + DARTEL-import rc1/rc2)
    -> DARTEL create Template (joint over the training cohort)
    -> Normalise to MNI, modulated (preserve=1), 1.5 mm
    -> smooth 3 mm FWHM      => smwc1pT1.nii  (the model feature volume)

All stages run through the SPM25 standalone (no MATLAB licence). Batch code is passed to
`spm25 eval` as one line (the MCR's run()/external-script path is unavailable).
"""
from __future__ import annotations

from misfit.paths import resolve_path as _resolve_path

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SPM = _resolve_path("<TOOLS>/bin/spm25")
TPM = _resolve_path("<TOOLS>/spm_standalone/spm_standalone/spm25_mcr/spm25/tpm/TPM.nii")
FWHM = 3
VOX = 1.5

# SPM12/25 default DARTEL schedule: (its, rparam[3], K, slam)
_DARTEL_PARAM = [
    (3, (4, 2, 1e-6), 0, 16),
    (3, (2, 1, 1e-6), 0, 8),
    (3, (1, 0.5, 1e-6), 1, 4),
    (3, (0.5, 0.25, 1e-6), 2, 2),
    (3, (0.25, 0.125, 1e-6), 4, 1),
    (3, (0.25, 0.125, 1e-6), 6, 0.5),
]


def _spm_eval(code: str, timeout: int) -> str:
    """Run one MATLAB batch string through the SPM25 standalone; return stdout+stderr."""
    r = subprocess.run([SPM, "eval", code], capture_output=True, text=True, timeout=timeout)
    return r.stdout + "\n" + r.stderr


# ----------------------------------------------------------------- Stage 1: Segment
def _seg_import_code(pT1: Path) -> str:
    """Build the SPM batch string for Segment with DARTEL import of GM/WM."""
    ng = [1, 1, 2, 3, 4, 2]
    L = [f"matlabbatch{{1}}.spm.spatial.preproc.channel.vols={{'{pT1},1'}};",
         "matlabbatch{1}.spm.spatial.preproc.channel.biasreg=0.001;",
         "matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm=60;",
         "matlabbatch{1}.spm.spatial.preproc.channel.write=[0 0];"]
    for k in range(1, 7):
        nat = "[1 1]" if k in (1, 2) else "[0 0]"   # GM & WM: native c + DARTEL-import rc
        L += [f"matlabbatch{{1}}.spm.spatial.preproc.tissue({k}).tpm={{'{TPM},{k}'}};",
              f"matlabbatch{{1}}.spm.spatial.preproc.tissue({k}).ngaus={ng[k-1]};",
              f"matlabbatch{{1}}.spm.spatial.preproc.tissue({k}).native={nat};",
              f"matlabbatch{{1}}.spm.spatial.preproc.tissue({k}).warped=[0 0];"]
    L += ["matlabbatch{1}.spm.spatial.preproc.warp.mrf=1;",
          "matlabbatch{1}.spm.spatial.preproc.warp.cleanup=1;",
          "matlabbatch{1}.spm.spatial.preproc.warp.reg=[0 0.001 0.5 0.05 0.2];",
          "matlabbatch{1}.spm.spatial.preproc.warp.affreg='mni';",
          "matlabbatch{1}.spm.spatial.preproc.warp.fwhm=0;",
          "matlabbatch{1}.spm.spatial.preproc.warp.samp=3;",
          "matlabbatch{1}.spm.spatial.preproc.warp.write=[0 0];",
          "matlabbatch{1}.spm.spatial.preproc.warp.vox=NaN;",
          "matlabbatch{1}.spm.spatial.preproc.warp.bb=[NaN NaN NaN; NaN NaN NaN];",
          "spm('defaults','PET'); spm_jobman('initcfg'); spm_jobman('run',matlabbatch); fprintf('SEG_DONE\\n');"]
    return " ".join(L)


def segment_one(t1: Path, wd: Path, timeout: int = 1200) -> dict:
    """Segment one T1 -> native c1 + DARTEL-import rc1/rc2 in `wd`. Idempotent."""
    wd.mkdir(parents=True, exist_ok=True)
    rc1 = wd / "rc1pT1.nii"
    if not rc1.exists():
        shutil.copy(t1, wd / "pT1.nii")
        out = _spm_eval(_seg_import_code((wd / "pT1.nii").resolve()), timeout)
        if "SEG_DONE" not in out or not rc1.exists():
            return {"ok": False, "wd": str(wd), "err": out[-500:]}
    return {"ok": True, "wd": str(wd), "c1": str(wd / "c1pT1.nii"),
            "rc1": str(rc1), "rc2": str(wd / "rc2pT1.nii")}


def segment_many(items: list[tuple[Path, Path]], jobs: int = 4) -> list[dict]:
    """items: [(t1_path, workdir), ...]. Runs SPM Segment in parallel (jobs processes)."""
    res = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(segment_one, t1, wd) for t1, wd in items]
        for f in as_completed(futs):
            res.append(f.result())
    return res


# ------------------------------------------------------- Stage 2: DARTEL create Template
def _dartel_template_code(rc1_txt: str, rc2_txt: str, name: str = "Template") -> str:
    # read path lists from files (avoids the ~128KB single-argv limit at cohort scale)
    rd = "c=textscan(fid,'%s','Delimiter','\\n'); fclose(fid);"
    L = [f"fid=fopen('{rc1_txt}'); {rd} rc1=c{{1}}; rc1=rc1(~cellfun('isempty',rc1));",
         f"fid=fopen('{rc2_txt}'); {rd} rc2=c{{1}}; rc2=rc2(~cellfun('isempty',rc2));",
         "fprintf('LOADED rc1=%d rc2=%d\\n', numel(rc1), numel(rc2));",
         "matlabbatch{1}.spm.tools.dartel.warp.images={rc1, rc2};",
         f"matlabbatch{{1}}.spm.tools.dartel.warp.settings.template='{name}';",
         "matlabbatch{1}.spm.tools.dartel.warp.settings.rform=0;"]
    for i, (its, rp, K, slam) in enumerate(_DARTEL_PARAM, 1):
        L += [f"matlabbatch{{1}}.spm.tools.dartel.warp.settings.param({i}).its={its};",
              f"matlabbatch{{1}}.spm.tools.dartel.warp.settings.param({i}).rparam=[{rp[0]} {rp[1]} {rp[2]}];",
              f"matlabbatch{{1}}.spm.tools.dartel.warp.settings.param({i}).K={K};",
              f"matlabbatch{{1}}.spm.tools.dartel.warp.settings.param({i}).slam={slam};"]
    L += ["matlabbatch{1}.spm.tools.dartel.warp.settings.optim.lmreg=0.01;",
          "matlabbatch{1}.spm.tools.dartel.warp.settings.optim.cyc=3;",
          "matlabbatch{1}.spm.tools.dartel.warp.settings.optim.its=3;",
          "spm('defaults','PET'); spm_jobman('initcfg'); spm_jobman('run',matlabbatch); fprintf('DARTEL_DONE\\n');"]
    return " ".join(L)


def dartel_create_template(rc1s: list[str], rc2s: list[str], list_dir: str,
                           timeout: int = 172800) -> str:
    """Joint DARTEL over the cohort (paths passed via list files). Writes Template_0..6.nii
    (dir of first rc1) and per-subject flow fields u_rc1pT1_Template.nii next to each rc1."""
    ld = Path(list_dir)
    ld.mkdir(parents=True, exist_ok=True)
    r1, r2 = ld / "rc1_list.txt", ld / "rc2_list.txt"
    r1.write_text("\n".join(rc1s) + "\n")
    r2.write_text("\n".join(rc2s) + "\n")
    return _spm_eval(_dartel_template_code(str(r1), str(r2)), timeout)


# --------------------------------------------------- Stage 3: Normalise to MNI + smooth
def _mni_norm_code(template6: str, subj: list[tuple[str, str]]) -> str:
    """Build the SPM batch string for DARTEL Normalise-to-MNI over (flowfield, c1) pairs."""
    L = [f"matlabbatch{{1}}.spm.tools.dartel.mni_norm.template={{'{template6}'}};"]
    for i, (ff, c1) in enumerate(subj, 1):
        L += [f"matlabbatch{{1}}.spm.tools.dartel.mni_norm.data.subj({i}).flowfield={{'{ff}'}};",
              f"matlabbatch{{1}}.spm.tools.dartel.mni_norm.data.subj({i}).images={{'{c1}'}};"]
    L += [f"matlabbatch{{1}}.spm.tools.dartel.mni_norm.vox=[{VOX} {VOX} {VOX}];",
          "matlabbatch{1}.spm.tools.dartel.mni_norm.bb=[NaN NaN NaN; NaN NaN NaN];",
          "matlabbatch{1}.spm.tools.dartel.mni_norm.preserve=1;",           # modulated
          f"matlabbatch{{1}}.spm.tools.dartel.mni_norm.fwhm=[{FWHM} {FWHM} {FWHM}];",
          "spm('defaults','PET'); spm_jobman('initcfg'); spm_jobman('run',matlabbatch); fprintf('NORM_DONE\\n');"]
    return " ".join(L)


def dartel_mni_norm(template6: str, subj: list[tuple[str, str]], timeout: int = 7200,
                    chunk: int = 40) -> str:
    """Warp native c1 to MNI via each flow field, modulated, 3 mm smooth -> smwc1pT1.nii
    (next to each c1). Chunked to keep each eval argument small at cohort scale.
    subj: [(flowfield, c1), ...]."""
    done = []
    for i in range(0, len(subj), chunk):
        grp = subj[i:i + chunk]
        log = _spm_eval(_mni_norm_code(template6, grp), timeout)
        if "NORM_DONE" not in log:
            return f"NORM_CHUNK_FAILED at subject {i}:\n" + log[-800:]
        done.append(f"norm chunk {i}-{i + len(grp)}: ok")
    return "\n".join(done) + "\nNORM_DONE"


# ------------------------------ Apply to a NEW cohort (AIBL) using the EXISTING ADNI template
def _warp_existing_code(rc1s: list[str], rc2s: list[str], templates1to6: list[str]) -> str:
    """DARTEL 'Run (existing Templates)' -> flow fields warping new subjects into the
    space defined by an already-built Template_1..6 (the ADNI template)."""
    def cell(fs):
        return "{" + ";".join(f"'{f},1'" for f in fs) + "}"
    L = [f"matlabbatch{{1}}.spm.tools.dartel.warp1.images={{{cell(rc1s)},{cell(rc2s)}}};",
         "matlabbatch{1}.spm.tools.dartel.warp1.settings.rform=0;"]
    for i, (its, rp, K, _slam) in enumerate(_DARTEL_PARAM, 1):
        L += [f"matlabbatch{{1}}.spm.tools.dartel.warp1.settings.param({i}).its={its};",
              f"matlabbatch{{1}}.spm.tools.dartel.warp1.settings.param({i}).rparam=[{rp[0]} {rp[1]} {rp[2]}];",
              f"matlabbatch{{1}}.spm.tools.dartel.warp1.settings.param({i}).K={K};",
              f"matlabbatch{{1}}.spm.tools.dartel.warp1.settings.param({i}).template={{'{templates1to6[i-1]}'}};"]
    L += ["matlabbatch{1}.spm.tools.dartel.warp1.settings.optim.lmreg=0.01;",
          "matlabbatch{1}.spm.tools.dartel.warp1.settings.optim.cyc=3;",
          "matlabbatch{1}.spm.tools.dartel.warp1.settings.optim.its=3;",
          "spm('defaults','PET'); spm_jobman('initcfg'); spm_jobman('run',matlabbatch); fprintf('WARP1_DONE\\n');"]
    return " ".join(L)


def dartel_warp_to_existing(rc1s, rc2s, template_dir: str, timeout: int = 14400) -> str:
    """Warp new-cohort rc1/rc2 into the existing ADNI DARTEL space (Template_1..6 in
    template_dir), producing per-subject flow fields. Returns log."""
    import glob as _g
    t = [sorted(_g.glob(f"{template_dir}/Template_{i}.nii"))[0] for i in range(1, 7)]
    return _spm_eval(_warp_existing_code(rc1s, rc2s, t), timeout)
