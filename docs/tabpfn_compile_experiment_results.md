# First TabPFN Compile experiment — 2026-09-15

The fixed-budget feasibility gate is complete on `data/vehicle.parquet`, using the supplied TabPFN-3.5 checkpoint and unchanged code at `../TabPFN` commit `27f4b3032103a082e6e3d503b485f7f5205369e9`.

**Result:** all 18 optimized contexts beat their matched real-row subsets on both validation and test teacher KL. This supports feature optimization on this dataset; it does not establish generality or a minimum sufficient context size.

Following the user's decision, labels stayed fixed. Native 3.5 casts labels to integer indices in both class embeddings and the decoder. The diagnostic produced feature-gradient norm `0.18594265` and no label gradient. Model weights remained frozen throughout.

## Main comparison

Means across seeds 42, 43 and 44, using stratified initialization and 50 Adam steps. Test KL measures divergence from the full 507-row teacher context; lower is better.

| Context rows | Stratified subset KL | Optimized KL | KL reduction | Context compression | Inference speedup |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.1024 | 0.9515 | 13.7% | 63.4× | 3.6× |
| 32 | 0.6301 | 0.1665 | 73.6% | 15.8× | 3.6× |
| 128 | 0.1637 | 0.0572 | 65.1% | 4.0× | 2.7× |

At 128 rows, optimized mean test accuracy was 81.2%, versus 79.6% for stratified subsets and 84.7% for the teacher. Fidelity improvements do not always improve accuracy: random initialization at 128 rows is a counterexample in the complete report.

No run reached validation KL ≤ 0.01. All six 128-row runs reached compile KL below 0.01, showing why separate validation matters. The experiment uses one fixed split (507 context / 169 compile / 85 validation / 85 test); seed variability is not dataset-level uncertainty. Inference speedups refer to warm fit-plus-forward with the same differentiable path on Apple GPU, not the default cached TabPFN API. Per-context GPU memory savings were not measured.

All 18 exported contexts reproduced compile/validation/test KL after reloading in fresh model instances, with maximum absolute error below 0.000002.

## Reproducible artifacts

- [Complete report, metrics, limitations, and commands](../artifacts/vehicle-fixed-labels/README.md)
- [Approximation frontier](../artifacts/vehicle-fixed-labels/frontier.png)
- [Run metadata, hashes, splits, and preprocessing](../artifacts/vehicle-fixed-labels/metadata.json)
- [Experiment runner](../scripts/tabpfn_compile_experiment.py)
- [Artifact reload verification](../scripts/verify_tabpfn_compile_artifacts.py)

The immediate next experiment is to repeat this fixed-label gate on additional numerical classification datasets before building automatic context-size search.
