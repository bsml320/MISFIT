"""Audit exports for biology_composite.pdf and the fancy-figs (dashboard etc.):
per-subject CSV + a stats JSON mapping every plotted number to its source file/field,
including the misfit-vs-CDR vs certificate-vs-CDR reconciliation.
Outputs: results/provenance/biology_composite_source.csv, biology_composite_stats.json,
         dashboard_source_map.json
Inputs: results/per_subject_misfit_iop.npz, results/per_subject_misfit.npz,
        results/image_signature.json, the nilearn OASIS-1 VBM tables, and the
        legacy clinical_insights.json. Run as a plain script with no arguments.
"""

from misfit.paths import resolve_path as _resolve_path
import json, re
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp, spearmanr, rankdata

RES = Path(_resolve_path("<REPO_ROOT>/misfit/results"))
OUT = RES / "provenance"; OUT.mkdir(exist_ok=True)

ci = np.load(RES/"per_subject_misfit_iop.npz")      # canonical IOP misfit (collect(), 5 seeds)
dx = np.load(RES/"per_subject_misfit.npz")           # image-pipeline misfit (examples/signature)
sig = json.load(open(RES/"image_signature.json"))

# OASIS CDR in loader order (same derivation as loader/examples scripts)
from nilearn.datasets import fetch_oasis_vbm
o = fetch_oasis_vbm()
cdr = o.ext_vars["cdr"].values.astype(float); age = o.ext_vars["age"].values.astype(float)
valid = ~np.isnan(cdr); age_v, cdr_v = age[valid], cdr[valid]
q = np.quantile(age_v, np.linspace(0,1,5)); env = np.digitize(age_v, q[1:-1])
counts = {e:int((env==e).sum()) for e in sorted(set(env))}
keep = np.isin(env,[e for e,c in counts.items() if c>=10]); cdr_o = cdr_v[keep]

# ---- per-subject CSV
import csv
with open(OUT/"biology_composite_source.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["cohort","idx","misfit_iop_canonical","misfit_imgpipe","label","err_frac_iop","cdr","pctile_iop"])
    ma=ci["adni_misfit"]; pa=100*rankdata(ma)/len(ma)
    for i in range(len(ma)):
        w.writerow(["adni",i,f"{ma[i]:.6g}",f"{dx['adni_misfit'][i]:.6g}",int(ci["adni_label"][i]),
                    f"{ci['adni_err'][i]:.3f}","",f"{pa[i]:.2f}"])
    mo=ci["oasis_misfit"]; po=100*rankdata(mo)/len(mo)
    for i in range(len(mo)):
        w.writerow(["oasis1",i,f"{mo[i]:.6g}",f"{dx['oasis_misfit'][i]:.6g}",int(ci["oasis_label"][i]),
                    f"{ci['oasis_err'][i]:.3f}",cdr_o[i],f"{po[i]:.2f}"])

# ---- stats JSON (every number annotated in the composite + reconciliation)
la=ci["adni_label"]; cn,imp=ma[la==0],ma[la==1]
ks=ks_2samp(cn,imp)
stage=np.where(cdr_o==0,0,np.where(cdr_o==0.5,1,2))
def sp(a,b): r=spearmanr(a,b); return {"rho":float(r.statistic),"p":float(r.pvalue)}
legacy=json.load(open(_resolve_path("<REPO_ROOT>/misfit/legacy_inputs/clinical_insights/clinical_insights.json")))
stats={
 "panel_a_adni":{"source":"per_subject_misfit_iop.npz (collect(), 5 seeds, canonical harness)",
   "cn_mean":float(cn.mean()),"impaired_mean":float(imp.mean()),
   "mean_ratio":float(cn.mean()/imp.mean()),"ks_D":float(ks.statistic),"ks_p":float(ks.pvalue),
   "note":"Legacy generator used a different loss scale; use the canonical recomputed statistics in this record."},
 "panel_a_oasis":{"source":"per_subject_misfit_iop.npz + nilearn CDR (loader-order derivation)",
   "spearman_stage_misfit_canonical":sp(stage,mo),
   "spearman_stage_misfit_imgpipe":sp(stage,dx["oasis_misfit"]),
   "spearman_continuousCDR_misfit_canonical":sp(cdr_o,mo),
   "stage_n":[int((stage==k).sum()) for k in (0,1,2)],
   "demented_over_healthy_mean_ratio":float(mo[la_o].mean()/mo[~la_o].mean()) if (la_o:=(ci["oasis_label"]==1)).any() else None},
 "cdr_reconciliation":{
   "composite_reports":"Spearman(CDR stage, structural MISFIT) = see panel_a_oasis",
   "legacy_certificate_stage_association":"cert_vs_cdr in misfit/legacy_inputs/clinical_insights/clinical_insights.json = Spearman(continuous CDR, PAC-BAYES CERTIFICATE), legacy TPAMI-era pipeline",
   "cert_vs_cdr_legacy":legacy["oasis"]["cert_vs_cdr"],
   "conclusion":"Distinct quantities: compare structural MISFIT and the legacy certificate separately."},
 "panel_b":{"source":"per_subject_misfit.npz picks (identical to supplementary misfit_examples.pdf)",
   "note":"illustrative single cases"},
 "panel_c":{"source":"adni_misfit_signature.nii.gz + image_signature.json",
   "adni_split_half":sig["adni"]["stability_split_half"],"adni_within_class":sig["adni"]["overall_within_agreement"],
   "oasis_split_half":sig["oasis"]["stability_split_half"],"oasis_within_class":sig["oasis"]["overall_within_agreement"],
   "caveat":"maps computed on the standard (full) cohorts; z-scored for display"}}
json.dump(stats,open(OUT/"biology_composite_stats.json","w"),indent=2)

# ---- dashboard source map (confirms the audit's mixed-lineage finding)
dash={
 "panel_a_glass_brain":{"file":"data/oasis3/_fancy/misfit_gm_spearman.nii.gz",
   "computed_on":"ALL 2720 OASIS-3 sessions (longitudinal, subject overlap present)"},
 "panel_b_forest":{"sources":{"adni/oasis1":"clinical_stratified.json + honest_estimator_table.json",
   "oasis3":"oasis3/oasis3_validation_unique.json (leakage-free)","abide":"abide_validation.json","ukbb":"ukbb/ukbb_validation.json"}},
 "panel_c_per_scanner":{"source":"oasis3/oasis3_per_scanner.json (leakage-free)"},
 "panel_d_selective":{"source":"data/oasis3/_fancy/per_subject.npz",
   "computed_on":"ALL 2720 sessions -> +16 pp at 50% coverage",
   "audit_finding_confirmed":"differs from participant-level leakage-free +18.9 pp (oasis3_validation_unique.json); dashboard mixes lineages and needs redesign before any use"},
 "status":"dashboard + clinical_utility figs are exploratory deliverables, NEVER wired into the manuscript or submission tarball"}
json.dump(dash,open(OUT/"dashboard_source_map.json","w"),indent=2)
print("wrote", [p.name for p in OUT.iterdir()])
