# Prespecification: controlled upweighting study, OASIS-3 CNN (frozen 2026-10-11, before execution)

PI-directed controlled experiment: does MISFIT-based selection of difficult source
cases, used to upweight (not remove) them under a fixed review budget, deliver a
model-development benefit beyond ordinary difficulty scores? This file is committed
before any training run; all design choices and the decision rule below are frozen.
New fits are required; the saved cnn18 score arrays are not reused for scoring here
because they were computed with fits whose training folds include each outer
evaluation scanner.

## Cohort, model, training settings

Identical to the committed backbone harness (`run_oasis3_cnn_backbone.py`): leakage-free
OASIS-3 cohort (N=1360, one earliest T1w session per participant), twofold-downsampled
smwc1 volumes (60x72x60), CNN3D (4 Conv-GroupNorm-ReLU-MaxPool blocks, 8-64 ch, GAP
head), Adam lr 1e-4, weight decay 1e-4, batch 16, 50 epochs, leave-one-scanner-out
outer folds, small-fold guard (held-out n < 5 skips Avanto, n=2), so 5 outer evaluation
scanners: TrioTim, Biograph_mMR, Vision, MAGNETOM_Vida, Sonata.

**One stated deviation, required by the design**: the backbone harness normalizes
volumes with the global mean/std over all 1360 subjects, which fits preprocessing on
the held scanner. Here the intensity normalization (scalar mean/std) is computed per
outer fold from the source pool only and applied unchanged to the held scanner. The
outer evaluation scanner therefore stays entirely outside preprocessing fitting,
scoring, selection and calibration. No threshold calibration is performed anywhere;
all evaluations use the default argmax operating point.

## Scoring stage (per outer fold; source pool only; scoring seed 41)

- **misfit**: inner leave-one-scanner-out over the source pool's 5 remaining scanners
  (5 inner fits; no inner test evaluation, training CE only; no small-fold guard needed
  since inner held-out subjects are never evaluated). Per-subject score = training CE
  at epoch 50 averaged over the 4 inner fits in which the subject is in training.
- **ordinary**: training CE at epoch 50 from one designated fit on the full source pool.
- **aum**: negative mean margin (true-logit minus best-other-logit) over epochs 1-50 of
  the designated fit; higher = harder.
- **el2n**: ||softmax - onehot||_2 at epoch 20 of the designated fit.
- All scoring fits use torch.manual_seed(41), permutation RNG np.RandomState(41*100 +
  inner_held) for inner fits and np.RandomState(41*100 + 99) for the designated fit.

## Budget, selection, upweighting

- Review/training budget: top 20% of the source pool by score (k = round(0.20 *
  n_source)), per fold, per score.
- **stage_random** control: a random source subset matched to the CDR-stage composition
  (0 / 0.5 / >=1 counts) of the misfit-selected set, sampled uniformly within stage,
  redrawn per final seed with np.RandomState(9000 + 10*outer_held + s), s = 0,1,2.
  This isolates whether misfit picks better subjects than its stage profile alone.
- Upweighting: selected cases get weight 3, others 1; weights renormalized to mean 1
  over the source pool; per-batch loss = mean(w_b * CE_b). With w = 1 everywhere this
  reduces exactly to the backbone loss.

## Arms and final fits (matched settings, common fresh seeds)

Six arms: uniform (no upweighting), misfit, ordinary, aum, el2n, stage_random.
Three fresh common seeds 52, 53, 54 (never used in this study line). For a given
(fold, seed), every arm uses the same torch.manual_seed(seed) and the same batch
permutation stream np.RandomState(seed*100 + outer_held), so initialization and batch
order are identical across arms; arms differ only in the per-sample loss weights.
Final fits: 6 arms x 3 seeds x 5 folds = 90; scoring fits: 5 x (5 inner + 1
designated) = 30; total 120 fits. Held-scanner predictions and softmax probabilities
recorded at epoch 50. GPU 2.

## Primary endpoint (frozen decision rule)

Held-scanner CDR-0.5 sensitivity: among tested impaired participants with CDR = 0.5
(n = 277, pooled over the 5 held scanners and 3 seeds, 831 trials), the fraction
predicted impaired at argmax. The specificity safeguard uses tested CDR = 0
participants (n = 974).

**Benefit is claimed only if both hold:**
1. The paired subject-level bootstrap 95% CI (B = 2000, RandomState(7), resampling the
   277 CDR-0.5 subjects with their seed-trials) of the difference in pooled CDR-0.5
   sensitivity, misfit arm minus the strongest comparator arm, excludes zero. The
   strongest comparator is fixed as the arm with the highest observed pooled CDR-0.5
   sensitivity among {ordinary, aum, el2n, stage_random} before bootstrapping.
2. Specificity safeguard: the misfit arm's pooled specificity is not lower than that
   strongest comparator's by more than 0.02 (point estimates).

Improvement over the uniform arm alone is explicitly not sufficient.

## Secondary endpoints (reported regardless of direction)

Per arm: pooled specificity, CDR >= 1 sensitivity, overall accuracy, AUROC on the
tested pool, per-scanner CDR-0.5 sensitivity; all pairwise sensitivity differences vs
misfit with the same bootstrap; misfit-vs-comparator selected-set Jaccard overlaps;
same contrasts vs the uniform arm.

## Reporting

All endpoints reported, honest nulls stated as nulls. Caveats carried: CDR also
defines the label; retrospective cohort; 15-of-18-style small-fold guard (Avanto never
evaluated); cuDNN conv3d backward nondeterminism can prevent bit-exact reproduction. Sonata contributes no CDR-0.5 impaired tested subjects; the primary
endpoint pool spans 4 held scanners. CDR stages from the candidate-audited participant
records (`results/provenance/oasis3_cdr_participant_records.csv`, audited 2026-10-05).
Outputs: `results/oasis3/oasis3_cnn_upweight_arrays.npz` (per-subject scores,
selections, per-arm predictions/probabilities/tested masks),
`results/oasis3/oasis3_cnn_upweight_study.json` (endpoints).
