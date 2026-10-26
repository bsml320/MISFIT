"""Run the MISFIT collector on generated data without study inputs or downloads."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def _run(args):
    # Scientific dependencies are loaded only after the CLI selects CPU execution.
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from misfit.experiments import run_honest_estimator_table as harness
    from misfit.experiments.run_clinical_stratified_media import collect

    rng = np.random.default_rng(args.seed)
    n_envs, dim = 3, 8
    n = n_envs * args.samples_per_environment
    env = np.repeat(np.arange(n_envs), args.samples_per_environment)
    latent = rng.normal(size=(n, dim))
    noise = rng.normal(size=n) * rng.uniform(0.4, 1.4, size=n)
    labels = (latent[:, 0] + 0.6 * latent[:, 1] + noise > 0).astype(np.int64)
    features = latent.copy()
    features[:, 2:] += (env[:, None] - 1) * 0.6

    previous_epochs, previous_patience = harness.N_EPOCHS, harness.PATIENCE
    previous_threads = torch.get_num_threads()
    try:
        harness.N_EPOCHS = args.epochs
        harness.PATIENCE = args.epochs + 1
        torch.set_num_threads(1)
        records = collect(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(env, dtype=torch.long),
            args.seed,
        )
    finally:
        harness.N_EPOCHS, harness.PATIENCE = previous_epochs, previous_patience
        torch.set_num_threads(previous_threads)

    tested = records["tested"]
    if not tested.all() or not np.isfinite(records["misfit"]).all():
        raise RuntimeError("Synthetic collection did not produce a valid score for every sample.")

    def auroc(mask):
        errors = records["err"][mask]
        if len(np.unique(errors)) < 2:
            return None
        return float(roc_auc_score(errors, records["misfit"][mask]))

    summary = {
        "synthetic": True,
        "description": "Generated-data usage example; these are not manuscript or clinical results.",
        "collector": "misfit.experiments.run_clinical_stratified_media.collect",
        "score": "Mean training cross-entropy over refits that include the sample.",
        "evaluation": "Prediction error from the refit that holds out the sample's environment.",
        "settings": {
            "seed": args.seed, "epochs": args.epochs, "device": "cpu",
            "environments": n_envs, "features": dim,
            "samples_per_environment": args.samples_per_environment,
            "training_refits_per_sample": n_envs - 1,
            "initializations": 1,
        },
        "n_samples": n,
        "n_tested": int(tested.sum()),
        "held_out_accuracy": float(1 - records["err"][tested].mean()),
        "failure_detection_auroc": auroc(tested),
        "within_label_failure_detection_auroc": {
            str(label): auroc(tested & (labels == label)) for label in (0, 1)
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "synthetic_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "environment", "label",
                         *[f"feature_{i}" for i in range(dim)],
                         "misfit", "prediction", "held_out_error"])
        for i in range(n):
            writer.writerow([f"synthetic_{i:04d}", int(env[i]), int(labels[i]),
                             *features[i].tolist(), float(records["misfit"][i]),
                             int(records["pred"][i]), int(records["err"][i])])
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Synthetic example: {n} samples, {n_envs} environments, CPU.")
    print(f"Held-out accuracy: {summary['held_out_accuracy']:.3f}")
    value = summary["failure_detection_auroc"]
    print("Failure-detection AUROC: " + (f"{value:.3f}" if value is not None else "undefined (one error class)"))
    print("These generated-data metrics demonstrate usage, not manuscript findings.")
    print(f"Wrote {args.output / 'synthetic_scores.csv'}")
    print(f"Wrote {args.output / 'summary.json'}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/synthetic_demo"),
                        help="Fresh or empty directory for generated synthetic outputs")
    parser.add_argument("--seed", type=int, default=7, help="Synthetic-data and fitting seed")
    parser.add_argument("--epochs", type=int, default=20, help="Short training budget per refit")
    parser.add_argument("--samples-per-environment", type=int, default=60,
                        help="Generated samples in each of three environments (at least 20)")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.seed > 2**32 - 1:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.samples_per_environment < 20:
        parser.error("--samples-per-environment must be at least 20")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Choose a new or empty output directory; existing runs are preserved.")
    # This standalone CLI sets visibility before importing torch or the collector.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _run(args)


if __name__ == "__main__":
    main()
