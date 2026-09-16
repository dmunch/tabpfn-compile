"""Reload saved contexts in a fresh model and verify held-out metric reproduction."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier
from paths import resolve_checkpoint, resolve_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--record-reference", action="store_true", help="Reconstruct missing Gate 1 prediction references; requires a separate verification run afterward")
    parser.add_argument("--diagnostic-only", action="store_true", help="Record all tolerance failures instead of stopping at the first one")
    args = parser.parse_args()
    directory = args.directory
    meta = json.loads((directory / "metadata.json").read_text())
    results = json.loads((directory / "results.json").read_text())
    config = meta["config"]
    torch.set_num_threads(config["threads"])
    data = pd.read_parquet(resolve_data(config["data"]))
    query = (data[meta["feature_names"]].to_numpy() - np.array(meta["mean"])) / np.array(meta["scale"])
    teacher = np.load(directory / "teacher.npz")
    checks = []
    for result in results:
        if not result["method"].startswith("synthetic_"):
            continue
        init = result["method"].removeprefix("synthetic_")
        file = directory / f"compiled-{result['rows']}-{result['seed']}-{init}.csv"
        artifact = pd.read_csv(file)
        assert len(artifact) == result["rows"]
        assert np.isfinite(artifact.to_numpy()).all()
        assert (artifact[config["target"]] % 1 == 0).all()
        reference_file = directory / f"predictions-{result['rows']}-{result['seed']}-{result['method']}.npz"
        reference = np.load(reference_file) if reference_file.exists() else None
        assert reference is not None or args.record_reference, "Missing saved prediction reference"
        model = TabPFNClassifier(model_path=str(resolve_checkpoint(config["checkpoint"]).resolve()),
                                 device=config["device"], n_estimators=1, random_state=2,
                                 inference_precision=torch.float32, differentiable_input=True,
                                 ignore_pretraining_limits=True)
        model.n_classes_ = len(meta["target_classes"])
        with torch.no_grad():
            model.fit_with_differentiable_input(
                torch.tensor(artifact[meta["feature_names"]].to_numpy(), dtype=torch.float32, device=config["device"]),
                torch.tensor(artifact[config["target"]].to_numpy(), dtype=torch.float32, device=config["device"]))
            deltas = {}
            prediction_errors = {}
            reconstructed = {}
            for split in ("compile", "validation", "test"):
                x = torch.tensor(query[meta["split_indices"][split]], dtype=torch.float32, device=config["device"])
                p = torch.cat([model.forward(b, use_inference_mode=True).cpu() for b in x.split(config["batch_size"])]).clamp_min(1e-8)
                t = torch.tensor(teacher[split]).clamp_min(1e-8)
                value = float((t * (t.log() - p.log())).sum(-1).mean())
                delta = abs(value - result[split]["kl"])
                if not args.diagnostic_only:
                    assert delta < 1e-5, (str(file), split, value, result[split]["kl"])
                deltas[split] = delta
                reconstructed[split] = p.numpy()
                if reference is not None:
                    error = float(np.abs(p.numpy() - reference[split]).max())
                    prediction_errors[split] = error
        if reference is None:
            np.savez(reference_file, **reconstructed)
        checks.append(dict(file=file.name, kl_absolute_errors=deltas, prediction_max_absolute_errors=prediction_errors))
        print(checks[-1], flush=True)
        del model
    assert len(checks) == len(config["budgets"]) * len(config["seeds"]) * len(config.get("initializations", ["random", "stratified"]))
    name = ("reference-reconstruction.json" if args.record_reference else
            "artifact-verification-diagnostic.json" if args.diagnostic_only else
            "artifact-verification.json")
    (directory / name).write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    main()
