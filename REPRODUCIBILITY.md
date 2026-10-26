# Synthetic example and optional manuscript reproduction

The default entry point is the self-contained synthetic example. It needs no original research data, saved study predictions, reporting references, credentials or GPU.

## Synthetic example

After installation, run:

~~~bash
python -m misfit.demo --output results/synthetic_demo
~~~

The command generates three environments of toy feature vectors and binary labels, runs the existing MISFIT collector with a short CPU training budget, and writes `synthetic_scores.csv` and `summary.json`. The CSV contains the generated inputs, fitting scores and held-out predictions. The summary records the settings, accuracy and failure-detection AUROCs, including within-label comparisons. An AUROC is null if a group contains only correct or only incorrect predictions.

Use a fresh or empty output directory for each run. `--seed`, `--epochs` and `--samples-per-environment` control this example; run `python -m misfit.demo --help` for details. Generated outputs are local and ignored by Git.

The example calls the released collector and fitting implementation; it only shortens the training budget. Its synthetic feature shifts and label noise illustrate the workflow and do not reproduce the paper's data-generating process or results.

## About the optional study workflows

The remaining sections are for users who choose to run cohort experiments or reproduce the manuscript's exact numerical results. Their authorized-data and historical-input requirements do not apply to the synthetic example or the HTML demo. Original replay files are not required to install, demonstrate or publish this code.

## Installation and paths

Install the project in editable mode from the repository root:

~~~bash
python -m pip install -e ".[test]"
~~~

Run drivers as modules from that root, for example `python -m misfit.experiments.run_oasis3_validation`. Editable installation also allows the original script-style commands.

The release's `<REPO_ROOT>` tokens are resolved by `misfit/paths.py`; they are no longer literal directory names. No source-file search-and-replace is required.

| Variable | Use |
| --- | --- |
| `MISFIT_DATA_DIR` | Redirects paths originally under `<REPO_ROOT>/data`; defaults to this repository's `data/` directory. |
| `OASIS3_NPZ` | OASIS-3 feature file for drivers that expose this variable. |
| `ADNI_SMRI_DATA_ROOT` | ADNI directory containing `nii/` and `labels/combined.csv`; set explicitly when data are outside the repository. |
| `MISFIT_DATA_SHARE` | Root for the optional UK Biobank loader; required only for that route. |
| `MISFIT_TOOLS` | Directory containing the SPM25 installation expected by Route B. |
| `MISFIT_NILEARN_DATA` | OASIS-1 cache used by CRT; falls back to `NILEARN_DATA` and then `~/nilearn_data`. |
| `MISFIT_REPORTING_REFERENCES` | External JSON containing historical aggregate inputs for optional reporting scripts; see below. |
| `CUDA_VISIBLE_DEVICES` | Caller-selected CUDA device visibility. Use an empty value for CPU execution where supported. |

Loader-specific options still apply: for example `misfit.data.abide_loader` accepts its own data directory, and the OASIS-1 loader uses nilearn's cache. `MISFIT_DATA_DIR` does not override every loader API. Outputs retain the original per-driver locations. The ignore rules exclude cohort files, caches, checkpoints and generated results.

Only load trusted `.npz` and pickle-based files: some historical loaders use `allow_pickle=True`.

## Current routes and prerequisites

### OASIS-3 feature-based validation

`misfit/experiments/run_oasis3_validation.py` reads a locally authorized feature archive with `features`, `label`, `scanner`, `subjects` and `sessions` arrays. Session names encode visit days. Its `load(unique=True)` path selects one earliest session per participant; pass `unique` to run the unique-participant analysis (the default is all sessions). Use the unique-participant results for the manuscript's leakage-free cohort.

~~~bash
python -m misfit.experiments.run_oasis3_validation unique
~~~

The clinical drivers also require the aligned session/CDR table. Do not substitute the all-session results for the unique-participant analysis.

### Corrected ADNI

The current route is `misfit/correction_20261018/code/rerun_adni_purged.py v2`. It purges every acquisition belonging to held-out participants from the training pool, then fits PCA and normalization on that source pool.

~~~bash
python -m misfit.correction_20261018.code.rerun_adni_purged v2
~~~

This requires the authorized preprocessed ADNI volumes and label table, or a verified local cache. The loader checks its historical feature binding before applying the corrected fold construction. The `v0` route is the historical, unpurged control; it must not be treated as current ADNI evidence. `v1` isolates purging while retaining historical preprocessing. After fitting, `python -m misfit.correction_20261018.code.corrected_summaries v2` produces the extended summaries. It additionally requires the participant linkage file `correction_20261018/audit/adni_linkage_local.csv`, with the verified participant, sex and age fields from the membership audit; the primary fit driver does not generate that file. The audit implementation is `audit_adni_membership.py` and additionally binds against the archived historical per-seed record, so it is a separate provenance step. Deferred corrected controls are under `deferred_reruns/code/`.

### Recovered CNN studies

These production fits use CUDA and preprocessed gray-matter volumes under `$MISFIT_DATA_DIR/oasis3/route_b_work/`. The backbone loader uses `$MISFIT_DATA_DIR/oasis3/oasis3_features.npz`, independently of the feature-validation driver's `OASIS3_NPZ` override.

First supply the authorized clinical exports under `misfit/results/provenance/`, or create them from the feature archive and aligned `oasis3_mr_labels.csv`:

~~~bash
python -m misfit.experiments.export_oasis3_cdr_records
~~~

The exporter runs the canonical feature-based collector and writes the participant CSV, matching clinical NPZ and metadata. Comparison with a previous mean-score NPY is optional and recorded as null when that file is absent.

Then run the image-level score-comparison or fixed-budget upweighting study:

~~~bash
python -m misfit.experiments.run_oasis3_cnn18_study all
python -m misfit.experiments.run_oasis3_cnn_upweight_study all
~~~

The score-comparison analysis reads both `oasis3_cdr_participant_arrays.npz` and `oasis3_cdr_participant_records.csv`; the upweighting study uses the records table. The first command produces CNN score/prediction arrays and an aggregate study JSON under `misfit/results/oasis3/`. These new fits are distinct from replaying the historically hash-bound arrays with the manuscript-analysis CLI.

Prespecifications are retained under `provenance/protocols/`. The nominal “18-fit” design executes only eligible scanner folds, as specified in that protocol.

### Current manuscript analyses

The [clinical commands](#recompute-the-clinical-and-conditional-analyses) cover the adjusted CDR contrast and same-/different-initialization CNN comparisons. The [rank-transfer command](#run-model-fitting-and-rank-transfer-validation) runs the auxiliary smooth logistic experiment. The [manuscript-to-code map](#manuscript-to-code-map) identifies the supporting sources.

### Auxiliary CNN curvature rank transfer

The CRT study uses independent source/target CNN banks, development-only calibration and participant-block uncertainty estimates. Its source modules are under [misfit/crt_study/code](misfit/crt_study/code/). The [frozen protocol](misfit/crt_study/protocol/frozen_protocol.md) and [deviations](misfit/crt_study/protocol/DEVIATIONS.md) define the historical design; CRT itself selects one scan per participant.

Production fits require CUDA and authorized preprocessed volumes:

| Cohort | Required inputs |
| --- | --- |
| OASIS-3 | `data/oasis3/adnitpl_work/<session>/smwc1pT1.nii` and `tcs_feasibility_manifest_local.csv` under `misfit/results/provenance/` |
| ADNI | `data/adni/smri/nii/` and `data/adni/smri/labels/combined.csv` |
| OASIS-1 | The nilearn OASIS-1 VBM cache and `oasis_cross-sectional.csv` |

The documented data/cache overrides apply. The three `cfg_*.json` files retain the historical normalization and training settings. After verifying inputs and configuration, create a local freeze and run:

~~~bash
python -m misfit.crt_study.code.freeze_protocol
python -m misfit.crt_study.code.run_dev banks oasis3 adni oasis1
python -m misfit.crt_study.code.run_dev freeze oasis3 adni oasis1
python -m misfit.crt_study.sim.null_calibration
python -m misfit.crt_study.code.run_conf sources oasis3 adni oasis1
python -m misfit.crt_study.code.run_conf targets oasis3 adni oasis1
python -m misfit.crt_study.code.run_conf evaluate oasis3 adni oasis1
~~~

Evaluate the null-calibration gate according to the protocol before opening confirmation. The [original freeze](provenance/crt_original_freeze.json) is archival evidence; it cannot substitute for the new source hashes. New runs create new provenance and must not be labeled as the historical execution.

For corrected inversion-AP and tie diagnostics, [crt_recompute.py](misfit/correction_20261018/code/crt_recompute.py) uses the original participant-block bootstrap and requires saved confirmation-row CSVs and `combined_summary.json`. The original rank-metric implementation remains available for historical reproduction. Interpret each comparison with its prespecified endpoint and cohort-specific uncertainty.

Run CPU engineering checks with `python -m pytest misfit/crt_study/tests`. Two real-cohort tests require configured inputs and explicit `-m slow`; the GPU pilot and smoke workflow are separate.

### Other analyses and spatial maps

The supplied drivers cover OASIS-1, ABIDE, clinical stratification, training-dynamics comparators and historical controls. Some fetch public atlases/templates through nilearn; those downloads are not part of the synthetic checks.

The orthogonal-view renderer consumes saved aggregate maps and binding files. They are not bundled here. The included HTML viewer is independently usable because it contains the finished aggregate video and poster, without individual scans.

### Package scope

The public package is `misfit`. It includes the paper analyses and the model, training, loader and comparator dependencies they need. Earlier project trees, their duplicate experiment copies and unrelated benchmark/certificate runners are excluded. The ADNI loader is available at `misfit.data.adni_loader`; corrected participant-purged fitting remains the current manuscript route.

### Inputs not distributed with the code

Authorized cohort volumes, labels, aligned clinical records, participant manifests and original fit/result arrays are still needed for real-data execution. Keep these inputs outside public version control.

The optional anonymized Route-B AIBL script also needs the original ADNI template directory resolved locally; `ADNI-PTID-01` is a placeholder, not a runnable template path. Derived-map permissions remain governed by their source dataset and template terms.

## Source-only distribution and reporting inputs

This repository ships no research results, raw cohort data, checkpoints or numerical result summaries. The HTML explorer's aggregate video and poster are the sole demo exception. Synthetic test inputs and algorithm/preprocessing settings are retained because they are needed to exercise and run the code. Generated outputs are ignored by Git.

Most analysis and figure scripts read the outputs produced by earlier pipeline stages. A few historical reporting scripts additionally use aggregate reference rows. Set `MISFIT_REPORTING_REFERENCES` to an external JSON file for these scripts; the forest/dashboard plotters require their aggregate rows and raise an explicit missing-input error when they are absent. Historical enrichment, operator and reliability analyses can run without reference values: they retain their computed outputs and omit unavailable comparisons. If an external file is explicitly configured but missing or malformed, it is reported as an input error. The default lookup is `misfit/results/reporting_references.json`, which is ignored by Git.

The JSON object may contain the following keys, as needed by the requested script:

| Key | Value schema |
| --- | --- |
| `cross_cohort_forest` | List of rows: label, subtitle, correlation, correlation interval half-width or null, AUROC, AUROC interval half-width, cognitive-cohort flag, highlight flag. |
| `dashboard_rows` | List of rows: label, correlation, correlation interval half-width, AUROC, AUROC interval half-width, color string, marker string. The pooled row uses the label `OASIS-3`. |
| `misfit_anchor` | Object mapping cohort keys `adni` and `oasis` to historical reference correlations. |
| `snt_zeroshot` | Object mapping the same cohort keys to historical zero-shot reference correlations. |

These historical references support legacy comparisons and plots; they are not substitutes for corrected cohort outputs. Keep the JSON and all generated reports outside the published source.

## Reproduction sequence

| Evidence family | Run order | Outputs or prerequisites |
| --- | --- | --- |
| OASIS-3 feature-based evidence | Authorized features → `run_oasis3_validation unique` | Aggregate validation JSON and selective-prediction figure. |
| Clinical-stage contrasts | Aligned labels/features → `export_oasis3_cdr_records` → `analyze_cdr_stage`, or use the authorized original records | Participant records remain private; the analysis writes aggregate estimates and intervals. |
| CNN conditional comparisons | Authorized original CNN/clinical arrays → `analyze_cnn_cdr_scanner` / `analyze_cnn_crossseed` | Hash-bound replay. Newly trained arrays require a separately verified binding. |
| Corrected ADNI comparisons | Authorized volumes/labels → `rerun_adni_purged v2` → extended summaries with the verified linkage file | Current participant-purged evidence. Historical unpurged outputs serve only provenance/control routes. |
| Smooth rank transfer | Features + aligned study records → `rank_validation.real_validation.run_validation` | Aggregate summary, settings, tables and private numerical arrays; output directory must be fresh. |
| Anatomical panels | Per-participant scores + authorized images/templates → map calculation → renderer | Map inputs and publication assembly assets must be supplied separately. |

The optional exact-replay section below gives copyable commands for the principal manuscript routes. [The manuscript-to-code map](#manuscript-to-code-map) maps individual figures and tables to the supporting scripts. Exact replay additionally requires study-specific data preparation, original replay inputs and publication assembly assets.

## Optional exact manuscript replay

### Prepare the inputs

The commands below run from the repository root after installation. Set these paths to your authorized files and a local output directory; the paths shown are examples to replace:

~~~bash
export MISFIT_DATA_DIR="/absolute/path/to/cohort-data"
export ADNI_SMRI_DATA_ROOT="$MISFIT_DATA_DIR/adni/smri"
export OASIS3_NPZ="$MISFIT_DATA_DIR/oasis3/oasis3_features.npz"
export MISFIT_ANALYSIS_INPUTS="/absolute/path/to/authorized-analysis-inputs"
export MISFIT_ANALYSIS_OUT="/absolute/path/to/local-analysis-output"
~~~

| Input | Used for |
| --- | --- |
| OASIS-3 features with participant/session identifiers, labels and scanners | Feature-based validation; the rank-transfer analysis also checks clinical metadata. |
| Preprocessed ADNI volumes under `nii/` and `labels/combined.csv` | Participant-purged ADNI fitting and comparisons. |
| `oasis3_cdr_participant_records.csv` | Adjusted CDR-stage analysis, including paired model-error records. |
| `oasis3_cnn18_arrays.npz` and `oasis3_cdr_participant_arrays.npz` | Saved CNN scores/predictions and aligned clinical records for the conditional comparisons. |

Cohort data and saved study arrays are **not included**. The [path guide](#installation-and-paths) and [workflow prerequisites](#current-routes-and-prerequisites) explain where each driver reads them. The HTML demo works independently of these inputs.

### Recompute the clinical and conditional analyses

With the authorized study inputs in `MISFIT_ANALYSIS_INPUTS`, run:

~~~bash
python -m misfit.manuscript_analysis.analyze_cdr_stage \
  --input "$MISFIT_ANALYSIS_INPUTS/oasis3_cdr_participant_records.csv" \
  --output "$MISFIT_ANALYSIS_OUT/cdr_stage.json"

python -m misfit.manuscript_analysis.analyze_cnn_cdr_scanner \
  --cnn-path "$MISFIT_ANALYSIS_INPUTS/oasis3_cnn18_arrays.npz" \
  --clinical-npz "$MISFIT_ANALYSIS_INPUTS/oasis3_cdr_participant_arrays.npz" \
  --out-dir "$MISFIT_ANALYSIS_OUT/cnn_same_initialization"

python -m misfit.manuscript_analysis.analyze_cnn_crossseed \
  --cnn-path "$MISFIT_ANALYSIS_INPUTS/oasis3_cnn18_arrays.npz" \
  --clinical-npz "$MISFIT_ANALYSIS_INPUTS/oasis3_cdr_participant_arrays.npz" \
  --out-dir "$MISFIT_ANALYSIS_OUT/cnn_different_initialization"
~~~

The clinical reader accepts the original 20-column records or the later 30-column release with verified predictions and tested flags. The CDR analysis uses 2,000 scanner-stratified participant-bootstrap draws, keeps each participant's five model records together and fixes percentile ranks before resampling. Optional `--reference-dir` checks separately supplied reference summaries after estimation; it is not required.

The CNN commands write `RESULTS.json` and `ANALYSIS_REPORT.txt` using 5,000 attempted participant-block bootstrap draws. Different-initialization analysis retains all six ordered pairings, six score constructions, fixed eligibility and equal cell weights. Saved AUM is already the negative epoch-averaged margin and must not be negated again; saved EL2N is from epoch 20. These commands do not retrain CNNs or export participant records.

**The CNN replay loaders verify the original study hashes and participant alignment.** Newly trained arrays will not necessarily have those hashes. The CLI is for replaying the manuscript's bound inputs; applying its statistical functions to a new run requires a separately verified input binding. Do not disable the checks or substitute arrays from a different analysis. The direct clinical NPZ must match the historical archive member; `--release-tar` can alternatively read that member from the original October 5 archive.

### Run model fitting and rank-transfer validation

For the principal feature-based fits:

~~~bash
python -m misfit.experiments.run_oasis3_validation unique
python -m misfit.correction_20261018.code.rerun_adni_purged v2
~~~

`unique` selects one earliest OASIS-3 session per participant. Its summary is written to `misfit/results/oasis3/oasis3_validation_unique.json`. `v2` selects participant-purged ADNI folds with source-fitted preprocessing; it writes estimator, within-fold and training-dynamics summaries under `misfit/correction_20261018/results/`, with private arrays under `reruns/`. Historical `v0` outputs are not the corrected manuscript evidence.

The smooth logistic rank-transfer experiment uses the feature matrix and aligned study records:

~~~bash
python -m misfit.rank_validation.real_validation.run_validation \
  --authorized-aligned-input \
  --features "$OASIS3_NPZ" \
  --cnn-path "$MISFIT_ANALYSIS_INPUTS/oasis3_cnn18_arrays.npz" \
  --clinical-npz "$MISFIT_ANALYSIS_INPUTS/oasis3_cdr_participant_arrays.npz" \
  --output "$MISFIT_ANALYSIS_OUT/rank_validation"
~~~

The auditor checks earliest-session selection and participant/session, label, scanner, age and CDR alignment. Object-string metadata use a non-executing opcode parser; numerical arrays use `allow_pickle=False`.

This fixed-representation logistic validation uses 50-component features, a common augmented-feature scale, penalties 0.1 and 1 on all coefficients including the intercept, nested 17/33/65-point Simpson grids and the original scanner-path eligibility rule. It checks conditional transport under smooth-model assumptions; it is separate from the CNN curvature study.

The command writes aggregate `summary.json`, settings and tables, plus private numerical arrays. Use a fresh output directory and keep outputs outside public version control. Synthetic checks are available through `python -m pytest misfit/rank_validation` and `python -m misfit.rank_validation.numerics.report_checks`.

Generated results remain local and are excluded by Git. The training drivers do not require bundled historical benchmark numbers. Optional legacy comparisons can read external references; plots that combine aggregate studies require their corresponding external inputs. See [reporting inputs](REPRODUCIBILITY.md#source-only-distribution-and-reporting-inputs).

## Manuscript-to-code map

Paths below are relative to the repository. These scripts cover the computations; reproducing the final assembled manuscript panels also requires the original figure layout assets.

| Manuscript result | Included implementation | Inputs and current status |
| --- | --- | --- |
| Methods and Fig. 1: paired fitting/error construction | [Canonical collection](misfit/experiments/run_clinical_stratified_media.py), [training and scoring](misfit/experiments/run_honest_estimator_table.py), [corrected ADNI harness](misfit/correction_20261018/code/purged_harness.py) | Synthetic scoring, inclusion/exclusion and participant-purge checks pass. Fig. 1 is a conceptual illustration. |
| Fig. 2 and Table 3: diagnosis-by-environment associations | [ADNI fit driver](misfit/correction_20261018/code/rerun_adni_purged.py), [corrected summaries](misfit/correction_20261018/code/corrected_summaries.py), [joint-strata export](misfit/correction_20261018/integration/code/export_joint_strata.py); [OASIS-1 per-seed records](misfit/experiments/run_perseed_dump_two_phases.py) and [stratum analysis](misfit/experiments/run_class_env_strata_s16.py) | Corrected ADNI requires `v2` fits; do not use the historical ADNI block in the OASIS-1 export route as current evidence. Authorized volumes or verified saved arrays are required. |
| Fig. 3, Abstract and Supplementary Table S9: adjusted CDR-stage contrast and missed impairment | [CDR-stage analysis](misfit/manuscript_analysis/analyze_cdr_stage.py) | Recovered and replayed on the existing records. Adjusted contrast and stage medians match the manuscript at reported precision. |
| Fig. 4, Abstract and Supplementary Tables S10–S12: within-stage/scanner CNN results | [CNN fits and score histories](misfit/experiments/run_oasis3_cnn18_study.py), [same-initialization analysis](misfit/manuscript_analysis/analyze_cnn_cdr_scanner.py), [different-initialization analysis](misfit/manuscript_analysis/analyze_cnn_crossseed.py) | Both statistical analyses were replayed with the original hash-bound arrays and 5,000 bootstrap attempts. New CNN training was not performed. |
| Table 4: access-matched loss/AUM/EL2N/forgetting | [Plain-MLP training dynamics](misfit/experiments/run_training_dynamics.py), [corrected ADNI driver](misfit/correction_20261018/code/rerun_adni_purged.py) and [paired summaries](misfit/correction_20261018/code/corrected_summaries.py) | Definitions and current ADNI path are present; a synthetic fit verifies all four score outputs using the paper implementation. |
| Table 5: low-loss training and CNN backbone | [ADNI duration sweep](misfit/correction_20261018/deferred_reruns/code/rerun_interpolation_purged.py), [OASIS-3 duration sweep](misfit/experiments/run_interpolation_ablation.py), [CNN backbone control](misfit/experiments/run_oasis3_cnn_backbone.py) | Training code is included. The backbone control is a distinct run from the CNN clinical-stage study. |
| Fig. 5: diagnostic distributions, selected cases and anatomy | [Corrected ADNI integration](misfit/correction_20261018/integration/code/integration_stats.py), [corrected figure calculations](misfit/correction_20261018/integration/code/regen_figures_corrected.py), [OASIS-1 anatomical analysis](misfit/experiments/run_image_signature_media.py) | Requires saved scores, authorized images and aggregate map inputs. The historical ADNI function in the OASIS-1 script is superseded by the corrected implementation. |
| Fig. 6: three-plane cognitive-cohort anatomy | [OASIS-3 map](misfit/experiments/run_oasis3_signature_participant.py), [corrected ADNI map](misfit/correction_20261018/integration/code/adni_signature_corrected.py), [OASIS-1 reconstruction](misfit/ortho_views/reconstruct_oasis1_map.py), [orthogonal renderer](misfit/ortho_views/make_ortho_views.py) | Map and rendering code are present. The final video/poster remain available in the HTML explorer; individual images and map inputs are not distributed. |
| Conditional rank-transfer calculation and Supplementary Section S10 | [Synthetic transport checks](misfit/rank_validation/numerics/rank_transport.py), [MRI-feature validation](misfit/rank_validation/real_validation/run_validation.py) | Recovered, tested and replayed on existing MRI-derived features. This smooth logistic experiment is distinct from the auxiliary CNN CRT study. |
| Supporting preprocessing, refit-construction and ABIDE analyses | [Corrected ADNI controls](misfit/correction_20261018/deferred_reruns/code/), [OASIS source-only features](misfit/experiments/run_sourceonly_features.py), [refit constructions](misfit/experiments/run_crossrefit_increment.py), [ABIDE validation](misfit/experiments/run_abide_validation.py) | Implementations are included; full cohort training was not replayed during this audit. |

Tables 1 and 2 describe information access and cohort design; they are not additional model-comparison experiments.

## Implementation notes

The primary source archive and recovered study files are recorded in [source_manifest.sha256](provenance/source_manifest.sha256), [recovered_sources.json](provenance/recovered_sources.json) and [manuscript_analysis_sources.json](provenance/manuscript_analysis_sources.json). The [package-layout manifest](provenance/package_layout.json) records the transition to the single `misfit` package. Original archive/freeze hashes are historical records; current file hashes identify the portable code.

The canonical `_train_fire` routine retains the historical inactive spectral-initialization branch: its incomplete `MRIData` construction raises `TypeError` inside a broad `try/except`. Operative scoring uses seeded random rank-one programs, frozen programs and fixed random subject loadings while fitting the encoder and prediction head. MISFIT uses per-subject loss. Tests protect this behavior; making the inactive branch execute would change the implementation underlying the saved results.

The paper package preserves its scientific calculations and original CNN input-integrity checks. The current ADNI route uses participant purging and source-fitted preprocessing; retained unpurged routes are historical controls. Model/data settings, input schemas and estimator definitions remain in the corresponding source modules.

## Validation boundary

Local validation covers syntax, editable installation, key import paths, deterministic CPU training on synthetic data, the retained inactive initializer, participant purging, source-only PCA, portable path resolution, CRT gradient/curvature/calibration/bootstrap identities, restored-module integration and viewer tests. No restricted cohort was copied into the repository. The manuscript coverage audit replayed the clinical analyses from saved scores/predictions and the small auxiliary logistic experiment on existing MRI features; it did not retrain the canonical or CNN cohort models.

Dependency versions tested locally are recorded in `requirements-validation.txt`. This is a software-check environment, not the historical environment used to generate manuscript results. CUDA and numerical library differences may affect near-tied ranks and bitwise reproducibility.

## Preparing a public release later

The repository is local and has no remote configured. Publication does not require distributing original replay inputs. Add the final citation/repository URL when available; derived-media permissions remain governed by the source terms. Keep participant identifiers, raw data, all research outputs and local environment files out of version control.
