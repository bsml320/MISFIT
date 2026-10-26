# CRT study: deviations and amendments log

Every entry is timestamped and states what changed, why, what data had been
seen, and what it could have influenced.

- 2026-10-18 (pre-freeze, engineering): firewall manifests made per-cohort
  (FIREWALL_<cohort>_rep<k>.json) so cohorts can run on separate GPUs without a
  shared-file race. No scientific outcome existed.

- 2026-10-18 (post-freeze, pre-confirmation, development data only): the
  null-calibration simulation v1 coupled each simulated participant's residual
  effect with that same participant's dev-fitted M3-minus-M2 prediction gap,
  which carries the dev-estimated signal into the nominal null, making a
  power reading inappropriate as a type-I error estimate. v2 draws the gap independently of the
  participant effect (permutation null, marginal preserved). The original simulation outputs are
  archived separately and are not bundled with this source release. sim/ is outside the frozen code/ gate;
  no frozen quantity, fit, calibrator or confirmation outcome was touched.

## Documented design deviations from established cohort definitions (pre-freeze)

- ADNI participant-level deduplication (directive-required). The established
  MedIA cohort is scan-level (N=731 scans, 547 participants, 156 participants
  in >= 2 protocol environments, same participant can appear in training and
  held-out folds). This study deduplicates to one earliest-phase scan per
  participant. The established cohort's non-deduplication is itself reported as
  an audit finding to the author (it affects the interpretation of the
  published ADNI within-environment correlations, not any file of record here).
- OASIS-1: volumes (mwrc1 VBM maps) are used directly instead of the
  established 100-parcel features, because the directive forbids substituting
  aggregate features for MRI volumes; cohort membership, label and
  age-quartile environments follow the established loader exactly.
- Normalization: global scalar volume z-score constants are computed once per
  cohort and frozen (the established backbone computed them on the fly over the
  full cohort; numerically identical constants, now recorded).
