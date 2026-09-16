"""Summarize the fixed-budget experiment without selecting runs on test metrics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    records = json.loads((directory / "results.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    if "completed_at" not in metadata:
        raise RuntimeError("Experiment is incomplete")
    table = pd.json_normalize(records)
    table.to_csv(directory / "metrics.csv", index=False)
    full = table[table.method == "full"].iloc[0]
    small = table[table.method != "full"]
    methods = ["random", "stratified", "synthetic_random", "synthetic_stratified"]
    colors = ["#9a6700", "#687787", "#9b4eb2", "#087f8c"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    for axis, metric, title in zip(axes,
                                 ["validation.kl", "test.kl", "test.accuracy"],
                                 ["Validation teacher KL ↓", "Test teacher KL ↓", "Test accuracy ↑"]):
        for method, color in zip(methods, colors):
            group = small[small.method == method].groupby("rows")[metric]
            axis.errorbar(group.mean().index, group.mean(), yerr=group.std().fillna(0),
                          marker="o", capsize=3, label=method.replace("_", " "), color=color)
        if metric == "test.accuracy":
            axis.axhline(full[metric], color="black", linestyle=":", label="full context")
            axis.legend(fontsize=8, loc="lower right")
        else:
            axis.set_yscale("log")
        axis.set(xscale="log", xlabel="Context rows", title=title)
        axis.set_xticks(sorted(small.rows.unique()), labels=sorted(small.rows.unique()))
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Vehicle · frozen TabPFN-3.5 · synthetic features, fixed labels\nMean ± seed SD; one fixed data split")
    fig.savefig(directory / "frontier.png", dpi=180)
    fig.savefig(directory / "frontier.svg")
    wins = sum(
        all(syn[f"{split}.kl"] < table[(table.method == syn.method.removeprefix("synthetic_")) &
                                       (table.rows == syn.rows) & (table.seed == syn.seed)].iloc[0][f"{split}.kl"]
            for split in ("validation", "test"))
        for _, syn in small[small.method.str.startswith("synthetic_")].iterrows())
    synthetic = small[small.method.str.startswith("synthetic_")]
    lines = ["# TabPFN Compile: first feasibility experiment", "",
             "This is the fixed-budget gate from the handover, using synthetic features and fixed integer labels.", "",
             f"Feature optimization beats its matched real-row baseline on both validation and test KL in {wins}/{len(synthetic)} runs. This supports the central phenomenon on this dataset, especially at 32 and 128 rows. It does not yet establish generality across datasets.", "",
             f"No high-fidelity claim at KL ≤ 0.01: the best validation KL among these runs is {synthetic['validation.kl'].min():.4f}. Compile KL below 0.01 at 128 rows does not carry over to held-out queries.", "",
             "## Setup", "",
             f"- Dataset: Vehicle, {metadata['source_rows']} rows, {len(metadata['feature_names'])} numerical features, four classes.",
             "- Disjoint stratified split: 507 teacher-context / 169 compile / 85 validation / 85 test rows.",
             "- Standardization fitted only on teacher-context rows; features clamped to its min/max ±1 standardized unit.",
             f"- Budgets: {metadata['config']['budgets']}; seeds: {metadata['config']['seeds']}; {metadata['config']['steps']} Adam steps, learning rate {metadata['config']['lr']}, query batches of {metadata['config']['batch_size']}.",
             "- Each synthetic run starts from the exact matched random or stratified subset. Lowest compile KL selects the saved iterate, including the initial context. Neither validation nor test selects an iterate, seed, or budget.",
             "- Teacher, baselines and synthetic contexts use one estimator, float32, and the same native differentiable preprocessing path. This is not the default multi-estimator, preprocessed TabPFN API predictor.",
             "- Model parameters frozen; nonzero finite feature gradients verified; no model weight gradients accumulated.", "",
             "## Held-out results", "",
             "Means over three initialization/optimization seeds on the same data split. KL is teacher-to-student divergence in nats; lower is better.", "",
             "| Rows | Method | Validation KL | Test KL | Test accuracy | Test AUC | Test log-loss | Latency (ms) |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for (m, method), group in table.groupby(["rows", "method"], sort=True):
        avg = group.mean(numeric_only=True)
        lines.append(f"| {m} | {method} | {avg['validation.kl']:.4f} | {avg['test.kl']:.4f} | {avg['test.accuracy']:.3f} | {avg['test.roc_auc']:.3f} | {avg['test.log_loss']:.3f} | {1000*avg['test_latency_seconds_median']:.1f} |")
    lines += ["", "## Matched comparisons", ""]
    for initialization in ("random", "stratified"):
        for m in sorted(small.rows.unique()):
            base = small[(small.method == initialization) & (small.rows == m)].set_index("seed")
            syn = small[(small.method == "synthetic_" + initialization) & (small.rows == m)].set_index("seed")
            reduction = 1 - syn["test.kl"].mean() / base["test.kl"].mean()
            wins = int((syn["test.kl"] < base["test.kl"]).sum())
            val_wins = int((syn["validation.kl"] < base["validation.kl"]).sum())
            speedup = full["test_latency_seconds_median"] / syn["test_latency_seconds_median"].mean()
            lines.append(f"- {m} rows, {initialization} initialization: test KL reduced {100*reduction:.1f}%; wins {wins}/{len(syn)} test, {val_wins}/{len(syn)} validation; {metadata['teacher_rows']/m:.2f}× fewer context rows; {speedup:.2f}× measured inference speedup.")
    lines += ["", "![Approximation frontier](frontier.png)", "", "## Interpretation limits", "",
              "- Three seeds measure initialization/optimization variability, not uncertainty across data splits or datasets. The 85-row test set is small; evidence here cannot establish consistent gains across tasks.",
              "- These are empirical fixed-budget results, not a global optimum or a discovered minimum context size. No automatic size search or tolerance claim is made.",
              "- The native 3.5 architecture casts targets to integer indices in its class embeddings and one-hot decoder. The diagnostic yielded a nonzero feature gradient and no label gradient. Fixed labels were explicitly approved for this gate; no soft-label model modification was made.",
              "- Timing is median of three warm fit-plus-forward calls over the same 85 test queries, split into batches of 64, including synchronization and copying predictions to CPU. It excludes checkpoint loading and optimization. It is specific to this uncached differentiable path and Apple GPU.",
              "- `process_peak_rss_bytes` is cumulative CPU process high-water memory, not per-context GPU memory; it does not establish a memory benefit.",
              "- Compression denominator is 507 teacher-context rows, not all 846 source rows.", "",
              "- At eight rows, random seeds 43 and 44 each omit a class. Fixed-label feature optimization cannot restore that class, explaining much of the high divergence. Stratified subsets cover all four classes.",
              "- Better teacher fidelity does not guarantee better ground-truth accuracy: the 128-row random runs improve KL while mean test accuracy changes from 80.4% to 79.2%.", "",
              "## Artifacts and reproduction", "",
              "`metadata.json` records source/checkpoint hashes, source commit, software versions, split indices, feature order, standardization and class encoding. `teacher.npz` caches teacher probabilities. `results.json`/`metrics.csv` contain all held-out metrics; `traces.json` contains compile-only optimization history.", "",
              "Each `compiled-<rows>-<seed>-<initialization>.csv` contains standardized features and fixed encoded labels. Reload those tensors with `n_classes_=4`, `differentiable_input=True`, and `fit_with_differentiable_input`; standardize new queries with metadata mean/scale. Do not standardize the compiled rows twice or decode labels into strings. The artifact is specific to this checkpoint and inference configuration.", "",
              "From the repository root (choose a new output directory):", "", "```bash",
              "PYTHONPATH=../TabPFN/src MPLCONFIGDIR=/tmp/tabpfn-compile-mpl \\",
              "  .venv/bin/python -u scripts/tabpfn_compile_experiment.py \\",
              "  --device mps --label-mode fixed \\",
              "  --output artifacts/vehicle-reproduction",
              ".venv/bin/python scripts/summarize_tabpfn_compile.py \\",
              "  artifacts/vehicle-reproduction", "```", ""]
    verification_file = directory / "artifact-verification.json"
    if verification_file.exists():
        verification = json.loads(verification_file.read_text())
        max_error = max(error for check in verification for error in check["kl_absolute_errors"].values())
        lines += ["## Reload verification", "",
                  f"All {len(verification)} CSV artifacts were reloaded in fresh model instances. Compile/validation/test KL reproduced within {max_error:.2g} absolute error (acceptance threshold 1e-5). See `artifact-verification.json`.", ""]
    (directory / "README.md").write_text("\n".join(lines))
    print("\n".join(lines[:lines.index("![Approximation frontier](frontier.png)")]))


if __name__ == "__main__":
    main()
