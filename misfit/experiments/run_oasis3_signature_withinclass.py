"""Within-class-pooled OASIS-3 anatomical signature (main-eligibility check).

The overall participant-level map passes its split-half gate but shows only
partial within-class agreement, i.e. it carries a between-class (CN-vs-impaired)
component. Here the map is computed WITHIN each class (voxelwise Spearman of GM
vs canonical misfit, separately among CN and among impaired participants) and
pooled (unweighted mean), removing the between-class contrast by construction.
Gates: class-stratified split-half stability of the pooled map, CN-vs-impaired
map agreement, fraction negative.
Leakage-free cohort (one earliest session per participant). GPU 2.
Outputs: results/oasis3/oasis3_signature_withinclass.{nii.gz,json}
         results/oasis3/oasis3_misfit_leakagefree_5seed.npy (canonical misfit cache)
"""

from misfit.paths import resolve_path as _resolve_path
import os, sys, json, re
# CUDA visibility is selected by the caller.
import numpy as np, torch, nibabel as nib
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import rankdata
sys.path.insert(0,_resolve_path("<REPO_ROOT>"))
from misfit.experiments.run_clinical_stratified_media import collect

WORK=_resolve_path("<REPO_ROOT>/data/oasis3/route_b_work")
RES=_resolve_path("<REPO_ROOT>/misfit/results/oasis3")

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

print(f"N={len(sess)}; canonical misfit (5 seeds)...",flush=True)
mis=np.zeros(len(sess))
for k in range(5):
    mis+=collect(f,t,site,42+k)["misfit"]
mis/=5.0
np.save(f"{RES}/oasis3_misfit_leakagefree_5seed.npy",mis)

print("loading volumes...",flush=True)
ref=nib.load(f"{WORK}/{sess[0]}/smwc1pT1.nii"); shp=ref.shape
def one(s): return nib.load(f"{WORK}/{s}/smwc1pT1.nii").get_fdata(dtype=np.float32).ravel()
with ThreadPoolExecutor(max_workers=16) as ex:
    X=np.stack(list(ex.map(one,sess)))
mask=X.mean(0)>0.1; Xm=X[:,mask]; del X
print(f"masked={mask.sum()}",flush=True)

def vox_spearman(idx):
    """Voxelwise Spearman of GM density vs misfit over subjects idx.

    Computed as the mean product of z-scored ranks: double argsort (ordinal
    ranks) over subjects per voxel, rankdata for the misfit vector.
    """
    g=np.argsort(np.argsort(Xm[idx],axis=0),axis=0).astype(np.float32)
    g-=g.mean(0); g/=(g.std(0)+1e-9)
    mr=rankdata(mis[idx]).astype(np.float32); mr=(mr-mr.mean())/mr.std()
    return (g*mr[:,None]).mean(0)

def wc_map(idx):
    """Unweighted mean of the CN-only and impaired-only voxelwise maps."""
    return np.stack([vox_spearman(idx[lab[idx]==c]) for c in (0,1)]).mean(0)

allidx=np.arange(len(mis))
wc=wc_map(allidx)
cn_map=vox_spearman(allidx[lab==0]); imp_map=vox_spearman(allidx[lab==1])
class_agree=float(np.corrcoef(cn_map,imp_map)[0,1])
# class-stratified split-half of the pooled within-class map
rng=np.random.RandomState(0)
h1=[]; h2=[]
for c in (0,1):
    ci=allidx[lab==c]; p=rng.permutation(len(ci))
    h1.append(ci[p[:len(ci)//2]]); h2.append(ci[p[len(ci)//2:]])
h1=np.concatenate(h1); h2=np.concatenate(h2)
m1,m2=wc_map(h1),wc_map(h2)
split_half=float(np.corrcoef(m1,m2)[0,1])
frac_neg=float((wc<0).mean())
overall_map=vox_spearman(allidx)
agree_overall=float(np.corrcoef(wc,overall_map)[0,1])
print(f"WITHIN-CLASS map: split-half r={split_half:.3f}  CN-vs-impaired map agreement={class_agree:.3f}  "
      f"frac_neg={frac_neg:.3f}  corr(with overall)={agree_overall:.3f}",flush=True)

vol=np.zeros(int(np.prod(shp)),np.float32); vol[mask]=wc
nib.save(nib.Nifti1Image(vol.reshape(shp),ref.affine,ref.header),f"{RES}/oasis3_signature_withinclass.nii.gz")
json.dump({"n":int(len(mis)),"pooling":"unweighted mean of CN-only and impaired-only voxelwise Spearman maps",
  "split_half_class_stratified":split_half,"cn_impaired_map_agreement":class_agree,
  "frac_negative":frac_neg,"corr_with_overall_map":agree_overall,
  "n_cn":int((lab==0).sum()),"n_impaired":int((lab==1).sum())},
  open(f"{RES}/oasis3_signature_withinclass.json","w"),indent=2)
print("wrote oasis3_signature_withinclass.{nii.gz,json}",flush=True)
