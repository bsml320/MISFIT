"""Participant-separated OASIS-3 anatomical signature (PI audit request).

Voxelwise Spearman between raw smwc1 gray-matter density and the canonical per-subject
misfit, on the LEAKAGE-FREE cohort (N=1360, one earliest session per participant; no
repeated participants, no cross-scanner overlap). Misfit = collect() 5-seed average on the
leakage-free features (same harness as oasis3_validation_unique). Gates: split-half
stability + within-class (CN/impaired) agreement, mirroring the ADNI/OASIS-1 signature
methodology. GPU 2.
Outputs: results/oasis3/oasis3_signature_participant.nii.gz + .json
         paper/figures/oasis3_signature_participant.pdf
"""

from misfit.paths import resolve_path as _resolve_path
import os, sys, json, re
# CUDA visibility is selected by the caller.
import numpy as np, torch, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import rankdata
sys.path.insert(0,_resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_clinical_stratified_media import collect

WORK=_resolve_path("<REPO_ROOT>/data/oasis3/route_b_work")
RES=_resolve_path("<REPO_ROOT>/misfit/results/oasis3")
FIG=_resolve_path("<REPO_ROOT>/misfit/paper/figures")

d=np.load(_resolve_path("<REPO_ROOT>/data/oasis3/oasis3_features.npz"),allow_pickle=True)
subj=d["subjects"].astype(str); sess=d["sessions"].astype(str)
day=np.array([int(re.search(r"_d(\d+)$",s).group(1)) for s in sess])
# one earliest session per participant (dedup by the _d<day> session suffix)
keep=np.array(sorted(np.where(subj==u)[0][np.argmin(day[subj==u])] for u in np.unique(subj)))
sess=sess[keep]; lab=d["label"].astype(int)[keep]; scan=d["scanner"].astype(str)[keep]
uniq,cnt=np.unique(scan,return_counts=True); order=uniq[np.argsort(-cnt)]
code={s:i for i,s in enumerate(order)}
site=torch.tensor([code[x] for x in scan])
f=torch.tensor(d["features"][keep],dtype=torch.float32); t=torch.tensor(lab)

print(f"leakage-free N={len(sess)}; computing canonical misfit (5 seeds)...",flush=True)
mis=np.zeros(len(sess))
for k in range(5):
    mis+=collect(f,t,site,42+k)["misfit"]
mis/=5.0

print("loading volumes...",flush=True)
ref=nib.load(f"{WORK}/{sess[0]}/smwc1pT1.nii"); shp=ref.shape
def one(s): return nib.load(f"{WORK}/{s}/smwc1pT1.nii").get_fdata(dtype=np.float32).ravel()
with ThreadPoolExecutor(max_workers=16) as ex:
    X=np.stack(list(ex.map(one,sess)))
mask=X.mean(0)>0.1; Xm=X[:,mask]; del X  # GM mask: mean density > 0.1 across subjects
print(f"masked voxels={mask.sum()}; ranking...",flush=True)

def vox_spearman(idx):
    """Voxelwise Spearman of GM density vs misfit over subjects idx.

    Computed as the mean product of z-scored ranks: double argsort (ordinal
    ranks) over subjects per voxel, rankdata for the misfit vector.
    """
    g=np.argsort(np.argsort(Xm[idx],axis=0),axis=0).astype(np.float32)
    g-=g.mean(0); g/=(g.std(0)+1e-9)
    mr=rankdata(mis[idx]).astype(np.float32); mr=(mr-mr.mean())/mr.std()
    return (g*mr[:,None]).mean(0)

allmap=vox_spearman(np.arange(len(mis)))
# gates: split-half stability (random halves) and within-class agreement (CN/impaired mean map)
rng=np.random.RandomState(0); perm=rng.permutation(len(mis))
h1,h2=vox_spearman(perm[:len(mis)//2]),vox_spearman(perm[len(mis)//2:])
split_half=float(np.corrcoef(h1,h2)[0,1])
cls=np.stack([vox_spearman(np.where(lab==c)[0]) for c in (0,1)]).mean(0)
within_agree=float(np.corrcoef(allmap,cls)[0,1])
frac_neg=float((allmap<0).mean())
print(f"split-half r={split_half:.3f}  within-class agreement={within_agree:.3f}  "
      f"frac negative={frac_neg:.3f}  range [{allmap.min():.3f},{allmap.max():.3f}]",flush=True)

vol=np.zeros(int(np.prod(shp)),np.float32); vol[mask]=allmap
img=nib.Nifti1Image(vol.reshape(shp),ref.affine,ref.header)
nib.save(img,f"{RES}/oasis3_signature_participant.nii.gz")
json.dump({"n":int(len(mis)),"cohort":"leakage-free one-session-per-participant",
  "n_voxels":int(mask.sum()),"split_half":split_half,"within_class_agreement":within_agree,
  "frac_negative":frac_neg,"misfit_source":"collect() 5-seed average on leakage-free features"},
  open(f"{RES}/oasis3_signature_participant.json","w"),indent=2)

z=np.zeros_like(vol); z[mask]=(allmap-allmap.mean())/(allmap.std()+1e-12)
from nilearn import plotting
fig=plt.figure(figsize=(7.4,2.9))
plotting.plot_stat_map(nib.Nifti1Image(z.reshape(shp),ref.affine),figure=fig,colorbar=True,
    display_mode="z",cut_coords=6,vmax=2.6,threshold=0.8,cmap="RdBu_r")
fig.suptitle("OASIS-3 (participant-level, $N{=}1360$): standardized gray-matter association with misfit",
    fontsize=10,y=1.0)
fig.savefig(f"{FIG}/oasis3_signature_participant.pdf",bbox_inches="tight",dpi=300)
print("wrote oasis3_signature_participant.{nii.gz,json,pdf}",flush=True)
