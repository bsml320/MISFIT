# CRT study: frozen protocol (curvature rank-transfer)

Created 2026-10-18, before any development or confirmation outcome existed.
Engineering pilots (timing, numerical identities) precede this freeze and use
reserved literal seeds (>= 990000, outside the purpose-namespaced stream by
convention); their outputs are never reused scientifically.

## 1. Hypothesis and endpoint (frozen)

Question: can source-model gradients and curvature predict changes in
participant-level fitting difficulty under environment exclusion, beyond
ordinary loss, AUM, EL2N and a matched gradient-only predictor?

Primary endpoint: within-environment loss-gap prediction MSE on confirmation
probes, E_gk(M) = (2/(n-1)) sum_i (u_i - ubar)^2 with u = q - qhat(M), averaged
EQUALLY over primary environments g and confirmation repetitions k. Decisive
comparison: M3 (difficulty + gradient + curvature) versus M2 (difficulty +
gradient). Delta_M = E(M) - E(M3), positive favors M3.

Primary statistical support: the two-sided 95% participant-block bootstrap
interval for Delta_M2 has a positive lower bound. Stronger conjunction claim:
positive lower bounds against ALL of M0, M1, M2 (a prespecified conjunction,
never a selection). Practical planning benchmark: 5% relative MSE reduction
versus M2. The four reporting categories are derived MECHANICALLY from the
intervals: supported and practically noticeable (lower bound > 0 AND relative
reduction >= 5% AND the crossed bootstrap agrees in sign); supported but small
(lower bound > 0 but either relative < 5% or the crossed bootstrap disagrees in
sign); inconclusive (interval spans 0); favors a control (upper bound < 0).
Pre-committed: an unsuccessful primary is reported as such; no secondary
replaces it.

## 2. Cohorts (three, each with its own frozen manifest digest)

- OASIS-3 (primary cohort of the directive): N=1360, one earliest session per
  participant, established clinical alignment (closest CDR to session, median
  gap 83 days; CDR defines the evaluation stratum, never a predictor). Label:
  CDR > 0. Environments: 6 scanner models; eligible exclusions = the 5
  established fits (TrioTim, Biograph_mMR, Vision, MAGNETOM_Vida, Sonata);
  primary target environments = the 4 with >= 8 CDR-0.5 probes. Probe stratum:
  CDR 0.5 (160/60/33/24 = 277). Volumes: ADNI-template smwc1pT1
  (data/oasis3/adnitpl_work), fixed external-template preprocessing.
- ADNI (replication cohort): established MedIA protocol-environment cohort with
  one DIRECTIVE-REQUIRED deviation: participant-level deduplication to one
  earliest-phase scan per PTID (phase order GO < 2 < 3, then protocol name;
  the established cohort is scan-level, N=731 over 547 participants, and that
  non-deduplication is documented as a finding of this audit). Label: CN=0 vs
  {AD, MCI, EMCI, LMCI}=1; SMC/Patient excluded (established rule). Environments:
  dataset_id protocol groups with >= 30 members post-dedup. Probe stratum: MCI
  spectrum {EMCI, MCI, LMCI} (ADNI MCI entry criteria correspond to CDR 0.5);
  AD scans stay in training but are never probes. Primary cells: >= 8 probes.
- OASIS-1 (replication cohort): established loader cohort (nilearn release, CDR
  present, N=226), label CDR > 0, environments = age quartiles over included
  subjects (established rule). Probe stratum: CDR 0.5. Primary cells: >= 8
  probes. Volumes: nilearn mwrc1 VBM gray-matter maps (fixed preprocessing).

Probe split (all cohorts): deterministic, environment-stratified, n_dev =
round(n/3) per primary cell, remainder confirmation; seed 20261018 via
purpose-namespaced sha256 streams. OASIS-3 realizes exactly the directive's
53/20/11/8 development and 107/40/22/16 confirmation counts. Split manifests are
private (DUA); digests are recorded and verified at every stage.

The probe split does NOT establish participant-held-out validation of the CNN
pipeline: confirmation participants appear in source training (known-label
retrospective audit) and in the global normalization constants. Target fits
exclude the probe's environment, so no probe is in its own target training
(asserted at assembly; violation raises).

## 3. Model and training recipe (frozen; identical for source and target banks)

Documented backbone CNN3D (4 conv blocks 8-64ch + GroupNorm + ReLU + MaxPool,
GAP, linear head; P = 73,282 parameters), volumes avg-pooled 2x, global scalar
z-score with constants computed once per cohort over the full cohort and frozen
in protocol/cfg_<cohort>.json (two-scalar leakage disclosed). 50 epochs, batch
16, Adam lr 1e-4, COUPLED weight decay 1e-4 (= gradient of (lam/2)||theta||^2,
lam = 1e-4; not decoupled AdamW, so it is a true objective penalty). No
augmentation, no dropout, no early stopping; final checkpoint = epoch 50.
GroupNorm has no batch statistics: the deterministic objective used for
derivatives equals the evaluation objective exactly. cuDNN conv3d training
nondeterminism exists (order 1e-3); all pairing is by hash, never bit identity.

Traces (source fits): per-epoch full-cohort margins (AUM = mean margin over
epochs 1-50, training members only; orientation recorded as aum_neg = -AUM so
larger = harder), EL2N at epoch 1, CE at checkpoints {10, 25, 50}.

## 4. Repetitions, banks, firewall (frozen)

3 development repetitions (0-2) and 5 confirmation repetitions (3-7). Each
repetition has its own source bank (one fit per eligible exclusion) and its own
separate target bank (one fresh fit per primary environment), all with distinct
purpose-namespaced seeds; no fit is shared across repetitions or roles. A
leave-one-out fit NEVER serves as both source and target: source-bank fits
produce features only; target outcomes q_i come only from the separate target
bank of the same repetition.

Firewall: for every confirmation repetition, source fits, transport features
and frozen-calibrator predictions are saved and sha256-hashed into
conf/FIREWALL_rep<k>.json BEFORE any confirmation target fit is trained; the
target driver refuses to run without the manifest; the locked evaluation
verifies every hash and refuses on any change, and additionally verifies that
re-predicting from the frozen calibrators reproduces the hashed predictions.

## 5. Mathematical definitions (frozen; see code/transport.py)

R_{-S}(theta) = mean CE over the cohort minus environment S + (lam/2)||theta||^2,
lam = 1e-4. For probes of environment g, containing refits C_g = eligible
exclusions minus g. L_i = mean final CE over C_g. q_i = final CE under the
target fit excluding g. Delta_{e,g} = grad[R_{-g} - R_{-e}] at theta_{-e} (the
ridge cancels; both objectives use the known labeled cohort; no target
parameters or outcomes enter). Curvature operator C_e = GGN of R_{-e} at
theta_{-e} + lam I + rho I (damped generalized Gauss-Newton, an approximation,
not the exact Hessian of the theorem). Binary CE makes the GGN exactly
rank-|train|: solves use the Woodbury identity through the |train| x |train|
kernel in float64 (torch.linalg.solve; exact, matrix-free in parameters;
residuals and condition numbers logged; engineering test verifies equality with
the dense autograd Hessian on ridge logistic, where GGN = Hessian).

Features: b_i = -(1/|C_g|) sum_e grad l_i(theta_{-e}) . Delta_{e,g};
a_i = -(1/|C_g|) sum_e grad l_i(theta_{-e}) . C_e^{-1} Delta_{e,g}.
Two approximations are named explicitly: (i) theta_{-e} is a 50-epoch Adam
endpoint, not a critical point, so the Newton reading of a drops the
grad R_{-e}(theta_{-e}) term (stationarity diagnostics report its norm); (ii)
the curvature used is the GGN of R_{-e}, not of R_{-g} (an O((|e|+|g|)/n)
operator difference). Delta and the GGN contain the probe's own known-label
self-influence terms (the probe is a training member of every containing
refit); this is source-side, known-label information, disclosed here.
Damping rho = c * tr(G_e)/P, c selected on DEVELOPMENT data only from the
prespecified grid {0.03, 0.1, 0.3, 1.0, 3.0} by best M3 blocked-CV score; one
c frozen per cohort for confirmation. Raw theoretical prediction (secondary
diagnostic only): rhat_raw_ij = d_ij + a_i - a_j, evaluated as the pair MSE of
u_raw = q - (L + a); its success or failure never validates or replaces the
calibrated comparison.

## 6. Calibration (frozen family; development-fitted)

B = {L, first-eligible-refit final CE, mean AUM(-), mean EL2N, refit CE std,
age} as natural cubic splines (5 knots at development quantiles
.1/.3/.5/.7/.9 -> 4 columns each) + linear sex. M0: qhat = L. M1: L + f(B).
M2: L + f(B, spline(b)). M3: L + f(B, spline(b), spline(a)). b and a receive
exactly the same basis so M3 vs M2 is never a flexibility artifact. f = ridge,
outcome and columns centered within (environment, repetition) blocks (block
intercepts cannot masquerade as rank-transfer skill; engineering test verifies
invariance to block shifts). Ridge lambda per model from {1e-3..1e2} by
participant-blocked 5-fold GroupKFold on development rows only, scored by the
within-block centered pair-MSE. All knots, standardization, lambdas, c and
coefficients are frozen to dev/frozen_calibrators_<cohort>.json before any
confirmation outcome exists.

## 7. Uncertainty (frozen)

Participant-block bootstrap, B = 2000, seed 20261018: resample participants
within environment, keeping each participant's full method x repetition block;
percentile CIs on Delta_M. THE PRIMARY CI IS CONDITIONAL ON THE REALIZED SOURCE
AND TARGET MODEL BANKS: participant resampling cannot capture bank-level
variation, so bank-level uncertainty is probed only by the repetition-level
spread (5 values; robustness, never independent replication by itself) and the
crossed participant x repetition bootstrap (seed 777; 5 blocks, few-block
caveat, anti-conservative risk in both directions). The crossed CI is always
printed adjacent to the primary CI; if their signs disagree, "supported and
practically noticeable" is downgraded to "supported but small" (frozen rule).
Degenerate (non-finite) draws are counted and their fraction reported
prominently; intervals are computed over the valid draws, and any nonzero
degenerate fraction is flagged next to the interval (with min cell size 16 and
every participant present in all repetitions, a degenerate draw is effectively
unreachable).
Ties: |gap| <= 1e-6 (an exploratory sensitivity at the training-noise scale
1e-2 is also reported); tie pairs are excluded from sign-based inversion counts
and their fraction reported. Calibration CV fold assignment uses frozen seed 7.

Null-calibration gate (required before confirmation opens): from the
development tables, estimate per-environment residual scale and the
participant-vs-repetition variance decomposition of M2 residuals; simulate
confirmation-shaped data (OASIS-3 cell sizes 107/40/22/16, 5 repetitions) by
resampling development residual components (tails preserved), inject relative
effects f in {0, 0.05, 0.10, 0.15}, and run the production bootstrap machinery
on >= 500 simulations per f with engineering seeds (>= 990000). Acceptance: at
f = 0 the frequency of a positive lower bound is <= 3.5%. If it exceeds 3.5%,
the frozen criterion still stands but every "supported" statement must carry
the measured type-I rate alongside. Measured power at each f is reported with
the results. The whole exercise is logged as pre-confirmation engineering.

## 8. Secondary endpoints (frozen list)

Raw uncalibrated transport pair-MSE; rank preservation (Spearman, tie-aware
Kendall tau-b); inversion prevalence and inversion average precision (score =
predicted movement against source order; rare-inversion caveat stated; the
number of contributing cells and total inversion pairs reported; AP score ties
are measure-zero for continuous predictions and noted); per-environment and
per-repetition tables; leave-one-environment-out influence; source stationarity
(||Delta_{e,g}|| and ||grad R_{-e}(theta_{-e})|| norms); solver residuals and
condition numbers; feature correlations INCLUDING corr(a, b) and the frozen
damping c_star reported next to the primary result (if c_star lands on the grid
boundary 3.0, where C^{-1} approaches a rescaled identity and a approaches a
rescaled b, the curvature interpretation is reported as tempered, frozen rule);
calibration-coefficient stability; compute time and memory. Exploratory
other-stage or added-feature analyses (e.g. checkpoint-10/25 CE in an extended
baseline) are labeled post hoc and can never replace a failed primary.

## 9. Interpretation boundaries (frozen)

A positive result supports additional predictive information in the
curvature-derived feature for within-stage, within-environment loss-gap changes
in fresh CNN refits. It does NOT establish: superiority of the original MISFIT
score; an environment-averaging advantage; a rank-preservation guarantee for
CNNs; clinical benefit; generalization to new participants or populations; the
unit-coefficient raw formula; a valid remainder bound; or stationary-path
assumptions. Internal confirmation within existing cohorts, not external
validation. "Fresh CNN refits" describes how q is defined; generalization
across refits beyond the realized banks is probed, not established (the primary
CI is bank-conditional, section 7). Per-cohort endpoints are reported
separately; the OASIS-3 endpoint is the directive's primary; ADNI and OASIS-1
are prespecified replications; no pooling across cohorts; the primary
conclusion comes from OASIS-3 alone, and replication statements require the
same criterion met in that cohort. OASIS-1 is PRE-LABELED UNDERPOWERED by
design (3 small cells; detectable relative effects only well above the
practical benchmark); its expected reading is inconclusive and that reading
carries no evidential weight against the OASIS-3 result in either direction.

## 10. Blinding, deviations, bug policy

Confirmation results do not determine damping, features, calibration, stopping,
eligibility, seeds or the endpoint (all frozen above). The freeze record
(PROTOCOL_FROZEN.txt) carries sha256 hashes of every code, protocol and test
file plus the cohort split digests; every scientific stage re-verifies the
code hashes and refuses on drift (override only via a logged, documented
deviation). Every deviation is logged in protocol/DEVIATIONS.md with a
timestamp. If a bug is found after
confirmation is opened, the original output is preserved, the correction
documented, and any corrected reanalysis labeled as such; the untouched
confirmation run remains in the record. All prespecified methods and completed
runs are retained regardless of direction; no reruns in search of a favorable
seed. This is a frozen local analysis plan, not a public preregistration.
