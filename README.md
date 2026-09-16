# TabPFN Compile

TabPFN Compile treats an in-context dataset as an optimizable object. It keeps a frozen TabPFN-3.5 model and optimizes a small numerical context so its predictions approximate those induced by a much larger labelled dataset.

This repository contains the first fixed-budget feasibility experiment and the preregistered eight-dataset replication gate. The central replication result is strong: feature optimization improved held-out teacher KL in all 72 matched comparisons. Sixty-three artifacts reload within the registered tolerance; nine CPU-produced Landsat artifacts expose an unresolved export/reload mismatch.

## Layout

- `scripts/`: experiment, orchestration, verification, and reporting code.
- `config/`: immutable Gate 2 preregistration and repository-migration hashes.
- `docs/`: mathematical handover and Gate 1/Gate 2 reports.
- `data/`: the eight numerical classification datasets used in Gate 2.
- `artifacts/`: complete Gate 1 and Gate 2 outputs, including contexts, cached predictions, traces, logs, figures, and decision records.
- `weights/`: local TabPFN-3.5 checkpoint. The checkpoint is intentionally ignored by Git.

## Environment

The recorded experiments used the sibling TabPFN checkout at commit `27f4b3032103a082e6e3d503b485f7f5205369e9` and checkpoint `weights/tabpfn-v3.5-20260909.safetensors` with SHA-256 `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3`.

From this repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ../TabPFN
.venv/bin/pip install -e . pyarrow
```

Run a fixed-budget experiment:

```bash
PYTHONPATH=../TabPFN/src MPLCONFIGDIR=/tmp/tabpfn-compile-mpl \
  .venv/bin/python -u scripts/tabpfn_compile_experiment.py \
  --data data/vehicle.parquet \
  --target vehicle_type \
  --checkpoint weights/tabpfn-v3.5-20260909.safetensors \
  --output artifacts/vehicle-reproduction \
  --device mps --label-mode fixed --initializations stratified
```

Run or resume the complete Gate 2 pipeline:

```bash
PYTHONPATH=../TabPFN/src .venv/bin/python scripts/gate2.py
```

The driver validates the frozen inputs, skips complete experiments, fills missing
experiments and verification records, applies the registered Landsat CPU fallback
after an MPS out-of-memory failure, and regenerates the aggregate report. Inspect
the planned work without changing files with `scripts/gate2.py --dry-run`. Existing
incomplete artifacts are never overwritten; use `--force --dataset NAME` to preserve
one under `artifacts/gate2/superseded/` and rerun it.

The scripts accept historical artifact metadata containing the original nanotabicl paths and resolve it against this repository by filename.

## Results

- [Experiment handover](docs/tabpfn_compile_handover.md)
- [Gate 1 report](docs/tabpfn_compile_experiment_results.md)
- [Gate 2 report](docs/tabpfn_compile_gate2.md)
- [Complete Gate 2 artifact report](artifacts/gate2/README.md)
- [Gate 2 paired-KL plot](artifacts/gate2/paired-kl.png)
