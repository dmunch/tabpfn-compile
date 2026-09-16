# Gate 2: cross-dataset replication

Protocol registered at `2026-09-15T19:49:43.203095+00:00`, before new predictions. Vehicle results were already known; the other seven datasets are prospective replication tasks.

Question: does feature-only context optimization consistently improve held-out teacher fidelity across numerical classification tasks?

## Frozen panel

| Dataset | Source rows | Context N | Features d | Classes K | Largest/smallest class |
|---|---:|---:|---:|---:|---:|
| Vehicle | 846 | 507 | 18 | 4 | 1.10 |
| Blood Transfusion | 748 | 448 | 4 | 2 | 3.20 |
| Diabetes | 768 | 460 | 8 | 2 | 1.87 |
| Phoneme | 5,404 | 3,242 | 5 | 2 | 2.41 |
| Spambase | 4,601 | 2,760 | 57 | 2 | 1.54 |
| Balance Scale | 625 | 375 | 4 | 3 | 5.88 |
| Image Segmentation | 2,310 | 1,386 | 19 | 7 | 1.00 |
| Landsat Satellite | 6,435 | 3,861 | 36 | 6 | 2.45 |

Five tasks use the existing local corpus. Three UCI numerical multiclass datasets complete the panel, because the local corpus contains only one complete numerical multiclass task. No result-dependent dataset selection, filtering, imputation, or feature encoding is used. Published UCI training and test files are concatenated before the common split.

Define reasonably balanced as largest/smallest class frequency ≤1.5, materially imbalanced as ≥2. Coverage: 4 binary, 4 multiclass, 4 low-dimensional (`d<15`), 4 medium/high-dimensional, 4 small contexts (`N<1000`), 3 larger contexts (`N>2000`), 2 balanced, 4 imbalanced.

## Frozen experiment

- Same supplied checkpoint, SHA-256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`.
- Same clean TabPFN commit `27f4b3032103a082e6e3d503b485f7f5205369e9`.
- Same 60/20/10/10 sequential stratified split with random states 42/43/44; context-only standardization and context min/max ±1 clamp.
- One estimator, float32, estimator random state 2, native differentiable numerical inference path, frozen weights and fixed labels.
- Stratified initialization; budgets 8/32/128; seeds 42/43/44; Adam defaults with LR 0.01; 50 updates; query batches of 64.
- Preserve Gate 1's iterate selection: lowest compile KL among initialization and steps 10/20/30/40/50. Always execute all 50 updates. Validation and test do not select the iterate.
- Same evaluation formulas and warm three-repeat fit-plus-forward latency measurement on Apple GPU.
- Reuse the nine stratified Vehicle runs; execute 63 new optimizations. Keep every dataset and run in the denominator.

Runner changes from Gate 1 are limited to selecting stratified-only execution, saving prediction arrays and additional bookkeeping, and checking pointwise predictions during reload. The split, mathematical metrics, initialization, minibatch sampling, optimization, and iterate-selection code are unchanged. The Gate 2 hashes lock this instrumented runner before execution.

## Preregistered decision

For each matched pair, `ΔKL = subset test KL − optimized test KL`. This absolute difference is primary. Also report `R_KL = ΔKL / subset test KL`, without using it for acceptance.

PASS requires all three:

1. At least 58 of 72 runs have strictly positive `ΔKL` (≥80%).
2. At least 6 of 8 datasets have positive mean `ΔKL` across their nine runs.
3. Median `ΔKL` across all 72 runs is strictly positive.

Five datasets with positive means is mixed evidence; four or fewer fails the general claim. A panel with six or more positive datasets still does not pass if either other criterion fails. Incomplete runs or invalid artifacts cannot support a PASS and are not excluded.

Record before/after compile/validation/test metrics; absolute and relative improvements; generalization gaps; class distributions; N/d/K; runtime; selected iterate; and the exported context. Reload each artifact in a fresh model at tolerance `1e-5`. New runs retain original prediction arrays; Vehicle's arrays are reconstructed and checked against its original saved KL, then verified again in fresh instances.

## Files

- [Machine-readable preregistration, coverage, data provenance and hashes](../artifacts/gate2/preregistration.json)
- [Resumable execution, verification, and reporting driver](../scripts/gate2.py)
- [Original immutable execution runner](../scripts/run_tabpfn_gate2.py)
- [Aggregation and diagonal plot](../scripts/summarize_tabpfn_gate2.py)
- Final report: `artifacts/gate2/README.md` (generated only after all eight datasets have been attempted).

No Gate 3 work is part of this run.

## Result

The run completed on 2026-09-16. The preregistered statistical criterion was met:

- 72/72 matched runs have positive held-out test `ΔKL`.
- 8/8 datasets have positive mean test `ΔKL`.
- Median test `ΔKL` is 0.081216 nats.
- The seven prospective datasets account for 63/63 wins; the conclusion does not depend on reusing Vehicle.

Mean absolute improvement is positive at every fixed budget: 0.198054 nats at 8 rows, 0.185286 at 32, and 0.067692 at 128.

The combined statistical and artifact verdict is **criterion met; artifact verification failure**. Sixty-three MPS-produced contexts reproduce within the 1e-5 KL threshold. Landsat could not run on MPS within its 8.29 GB allocation and used CPU as a documented hardware fallback. All nine Landsat runs improve held-out test KL, but none of their optimized CSVs reproduce the recorded predictions within the reload tolerance; maximum per-artifact KL error ranges from 0.000547 to 0.012855. The matched unoptimized Landsat subset reproduces pointwise within 9.9e-7, isolating the issue to optimized-context export/reload rather than generic CPU inference.

This is strong evidence for cross-dataset feature optimization, including all 63 previously unseen comparisons. It is not a clean end-to-end compiler PASS until the Landsat artifact mismatch is understood and fixed.

- [Complete report and limitations](../artifacts/gate2/README.md)
- [Preregistered gate decision](../artifacts/gate2/gate-decision.json)
- [Paired test-KL plot](../artifacts/gate2/paired-kl.png)
- [All paired run metrics](../artifacts/gate2/paired-metrics.csv)
