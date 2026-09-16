"""Fixed-budget feasibility gate from docs/tabpfn_compile_handover.md.

Run with PYTHONPATH=../TabPFN/src .venv/bin/python -u <this file>.
Teacher, subsets, and synthetic contexts use identical differentiable preprocessing.
Labels are fixed by default: the native 3.5 architecture casts them to integers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabpfn import TabPFNClassifier
from paths import ARTIFACTS, DATA, WEIGHTS, resolve_checkpoint, resolve_data


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def kl(p, q):
    p, q = p.clamp_min(1e-8), q.clamp_min(1e-8)
    return (p * (p.log() - q.log())).sum(-1).mean()


def metrics(p, teacher, y):
    midpoint = (p + teacher) / 2
    a = p.numpy()
    auc = roc_auc_score(y, a[:, 1]) if a.shape[1] == 2 else roc_auc_score(
        y, a, multi_class="ovr", labels=np.arange(a.shape[1]))
    return dict(kl=float(kl(teacher, p)),
                js=float((kl(teacher, midpoint) + kl(p, midpoint)) / 2),
                agreement=float((p.argmax(1) == teacher.argmax(1)).float().mean()),
                accuracy=float(accuracy_score(y, a.argmax(1))),
                log_loss=float(log_loss(y, a, labels=np.arange(a.shape[1]))),
                roc_auc=float(auc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA / "vehicle.parquet")
    parser.add_argument("--target", default="vehicle_type")
    parser.add_argument("--checkpoint", type=Path, default=WEIGHTS / "tabpfn-v3.5-20260909.safetensors")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "vehicle")
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--label-mode", choices=["continuous", "fixed"], default="fixed")
    parser.add_argument("--initializations", nargs="+", choices=["random", "stratified"], default=["random", "stratified"])
    args = parser.parse_args()
    args.data = resolve_data(args.data)
    args.checkpoint = resolve_checkpoint(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    frame = pd.read_parquet(args.data)
    features = frame.drop(columns=args.target)
    if not all(pd.api.types.is_numeric_dtype(t) for t in features.dtypes):
        raise ValueError("Only numerical features supported")
    if not np.isfinite(features.to_numpy(dtype=float)).all() or frame[args.target].isna().any():
        raise ValueError("This feasibility gate requires complete finite data")
    encoder = LabelEncoder().fit(frame[args.target])
    y = encoder.transform(frame[args.target])
    context, remaining = train_test_split(np.arange(len(y)), test_size=0.4, stratify=y, random_state=42)
    compile_ids, remaining = train_test_split(remaining, test_size=0.5, stratify=y[remaining], random_state=43)
    validation, test = train_test_split(remaining, test_size=0.5, stratify=y[remaining], random_state=44)
    ids = dict(context=context, compile=compile_ids, validation=validation, test=test)
    assert len(set(np.concatenate(list(ids.values())))) == len(y)
    scaler = StandardScaler().fit(features.iloc[context])
    X = torch.tensor(scaler.transform(features), dtype=torch.float32, device=args.device)
    Y = torch.tensor(y, dtype=torch.float32, device=args.device)
    low, high = X[context].amin(0) - 1, X[context].amax(0) + 1
    metadata = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    created_at=datetime.now(timezone.utc).isoformat(),
                    source_sha256=digest(args.data), checkpoint_sha256=digest(args.checkpoint),
                    feature_names=features.columns.tolist(), mean=scaler.mean_.tolist(),
                    scale=scaler.scale_.tolist(), target_classes=encoder.classes_.tolist(),
                    split_indices={k: v.tolist() for k, v in ids.items()},
                    source_rows=len(y), teacher_rows=len(context), torch_version=torch.__version__,
                    tabpfn_source=str(Path(__import__('tabpfn').__file__).resolve()),
                    tabpfn_git_commit=subprocess.check_output([
                        "git", "-C", str(Path(__import__('tabpfn').__file__).resolve().parents[2]),
                        "rev-parse", "HEAD"], text=True).strip(),
                    dependencies={name: importlib.metadata.version(name) for name in
                                  ("numpy", "pandas", "scikit-learn", "safetensors", "skrub", "torch")},
                    platform=platform.platform(),
                    inference="n_estimators=1; float32; differentiable_input=True; same path for all contexts",
                    selection="lowest compile KL; validation/test never select optimization steps",
                    labels=args.label_mode + "; reload via fit_with_differentiable_input; CSV features are standardized")
    save_json(args.output / "metadata.json", metadata)
    clf = TabPFNClassifier(model_path=str(args.checkpoint.resolve()), device=args.device,
                          n_estimators=1, random_state=2, inference_precision=torch.float32,
                          differentiable_input=True, ignore_pretraining_limits=True)
    clf.n_classes_ = len(encoder.classes_)
    clf.fit_with_differentiable_input(X[context], Y[context])
    for model in clf.models_:
        model.eval()
        model.requires_grad_(False)
    assert not any(p.requires_grad for model in clf.models_ for p in model.parameters())

    def synchronize():
        if args.device == "mps":
            torch.mps.synchronize()
        elif args.device.startswith("cuda"):
            torch.cuda.synchronize()

    def predict(cx, cy, query):
        with torch.no_grad():
            clf.fit_with_differentiable_input(cx, cy)
            return torch.cat([clf.forward(b, use_inference_mode=True).detach().cpu()
                              for b in query.split(args.batch_size)])

    print(f"Split sizes: { {k: len(v) for k, v in ids.items()} }", flush=True)
    teacher = {}
    for name in ("compile", "validation", "test"):
        start = time.perf_counter()
        teacher[name] = predict(X[context], Y[context], X[ids[name]])
        print(f"Teacher {name}: {time.perf_counter()-start:.2f}s", flush=True)
    np.savez(args.output / "teacher.npz", **{k: v.numpy() for k, v in teacher.items()})
    rows, traces = [], []

    def evaluate(cx, cy, method, m, seed):
        result = dict(method=method, rows=m, seed=seed, compression=len(context)/m,
                      N=len(context), d=features.shape[1], K=len(encoder.classes_),
                      context_class_counts=np.bincount(cy.detach().cpu().numpy().astype(int), minlength=len(encoder.classes_)).tolist())
        saved_predictions = {}
        for name in ("compile", "validation", "test"):
            predictions = predict(cx, cy, X[ids[name]])
            saved_predictions[name] = predictions.numpy()
            result[name] = metrics(predictions, teacher[name], y[ids[name]])
        np.savez(args.output / f"predictions-{m}-{seed}-{method}.npz", **saved_predictions)
        result["generalization_gap_validation"] = result["validation"]["kl"] - result["compile"]["kl"]
        result["generalization_gap_test"] = result["test"]["kl"] - result["compile"]["kl"]
        timings = []
        for _ in range(3):
            synchronize()
            start = time.perf_counter()
            predict(cx, cy, X[test])
            synchronize()
            timings.append(time.perf_counter() - start)
        result["test_latency_seconds_median"] = float(np.median(timings))
        result["test_latency_seconds_samples"] = timings
        # ru_maxrss is cumulative process peak, not context-specific memory.
        result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
        rows.append(result)
        save_json(args.output / "results.json", rows)
        print(json.dumps(result), flush=True)
        return result

    evaluate(X[context], Y[context], "full", len(context), 42)
    for m in args.budgets:
        if m >= len(context) or m < len(encoder.classes_):
            raise ValueError("Budget must cover classes and be smaller than teacher context")
        for seed in args.seeds:
            for initialization in args.initializations:
                rng = np.random.default_rng(seed)
                if initialization == "random":
                    selected = rng.choice(context, m, replace=False)
                else:
                    selected, _ = train_test_split(context, train_size=m, stratify=y[context], random_state=seed)
                cx, cy = X[selected].clone(), Y[selected].clone()
                initial_y = cy.detach().clone()
                baseline = evaluate(cx, cy, initialization, m, seed)
                cx.requires_grad_(True)
                cy.requires_grad_(args.label_mode == "continuous")
                optimizer = torch.optim.Adam([cx, cy] if cy.requires_grad else [cx], lr=args.lr)
                best_loss = baseline["compile"]["kl"]
                best_x, best_y, best_step = cx.detach().clone(), cy.detach().clone(), 0
                start = time.perf_counter()
                for step in range(1, args.steps + 1):
                    batch = rng.choice(len(compile_ids), min(args.batch_size, len(compile_ids)), replace=False)
                    optimizer.zero_grad(set_to_none=True)
                    clf.fit_with_differentiable_input(cx, cy)
                    p = clf.forward(X[compile_ids[batch]], use_inference_mode=True)
                    loss = kl(teacher["compile"][batch].to(args.device), p)
                    loss.backward()
                    if step == 1:
                        gradients = {name: None if tensor.grad is None else float(tensor.grad.norm())
                                     for name, tensor in (("X", cx), ("y", cy))}
                        print(f"Gradients m={m}: {gradients}", flush=True)
                        save_json(args.output / f"gradients-{m}-{seed}-{initialization}.json", gradients)
                        assert cx.grad is not None and torch.isfinite(cx.grad).all() and cx.grad.norm() > 0, "Invalid feature gradients"
                        if cy.requires_grad:
                            assert cy.grad is not None and torch.isfinite(cy.grad).all() and cy.grad.norm() > 0, "Native model does not provide usable label gradients"
                    optimizer.step()
                    with torch.no_grad():
                        cx.clamp_(low, high)
                    if step % args.eval_every == 0 or step == args.steps:
                        value = float(kl(teacher["compile"], predict(cx, cy, X[compile_ids])))
                        traces.append(dict(rows=m, seed=seed, initialization=initialization, step=step, compile_kl=value))
                        if value < best_loss:
                            best_loss, best_x, best_y, best_step = value, cx.detach().clone(), cy.detach().clone(), step
                        save_json(args.output / "traces.json", traces)
                        print(f"m={m} seed={seed} {initialization} step={step} compile KL={value:.6f} elapsed={time.perf_counter()-start:.1f}s", flush=True)
                training_seconds = time.perf_counter() - start
                if args.label_mode == "fixed":
                    assert torch.equal(initial_y, cy) and torch.equal(initial_y, best_y)
                result = evaluate(best_x, best_y, "synthetic_" + initialization, m, seed)
                result.update(best_step=best_step, training_seconds=training_seconds,
                              initialization_row_indices=selected.tolist(),
                              labels_unchanged=bool(torch.equal(initial_y, best_y)),
                              adam_steps_completed=args.steps,
                              label_min=float(best_y.min()), label_max=float(best_y.max()))
                artifact = pd.DataFrame(best_x.cpu().numpy(), columns=features.columns)
                artifact[args.target] = best_y.cpu().numpy()
                artifact.to_csv(args.output / f"compiled-{m}-{seed}-{initialization}.csv", index=False)
                save_json(args.output / "results.json", rows)
    assert all(p.grad is None for model in clf.models_ for p in model.parameters())
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["weights_frozen_verified"] = True
    save_json(args.output / "metadata.json", metadata)


if __name__ == "__main__":
    main()
