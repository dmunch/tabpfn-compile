"""Acquire missing numerical datasets and preregister Gate 2 before predictions."""
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from paths import ARTIFACTS, DATA, REPO_ROOT


ROOT = ARTIFACTS / "gate2"
PANEL = [
    ("vehicle", "vehicle_type"),
    ("blood-transfusion-service-center", "class"),
    ("diabetes", "diabetes_diagnosis"),
    ("phoneme", "phoneme_sound_class"),
    ("spambase", "is_spam"),
    ("balance-scale", "class"),
    ("image-segmentation", "class"),
    ("landsat-satellite", "class"),
]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "preregistration.json").exists():
        raise RuntimeError("Preregistration already exists; do not overwrite")
    sources = {}

    def fetch(url):
        raw = urlopen(url, timeout=60).read()
        sources[url] = hashlib.sha256(raw).hexdigest()
        return raw.decode("utf-8")

    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    balance = pd.read_csv(io.StringIO(fetch(base + "balance-scale/balance-scale.data")),
                          names=["class", "left_weight", "left_distance", "right_weight", "right_distance"])
    segmentation = []
    for part in ("data", "test"):
        lines = fetch(base + "image/segmentation." + part).splitlines()
        header = next(line for line in lines if line.startswith("REGION-CENTROID-COL"))
        records = [line for line in lines if line and line.split(",")[0] in
                   {"BRICKFACE", "SKY", "FOLIAGE", "CEMENT", "WINDOW", "PATH", "GRASS"}]
        segmentation.append(pd.read_csv(io.StringIO("\n".join(records)), names=["class"] + header.split(",")))
    satellite = [pd.read_csv(io.StringIO(fetch(base + "statlog/satimage/sat." + part)),
                            sep=r"\s+", names=[f"band_{i}" for i in range(36)] + ["class"])
                 for part in ("trn", "tst")]
    for name, frame in (("balance-scale", balance),
                        ("image-segmentation", pd.concat(segmentation, ignore_index=True)),
                        ("landsat-satellite", pd.concat(satellite, ignore_index=True))):
        path = DATA / (name + ".parquet")
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite {path}")
        frame.to_parquet(path, index=False)

    datasets = []
    for name, target in PANEL:
        path = DATA / (name + ".parquet")
        f = pd.read_parquet(path)
        x, y = f.drop(columns=target), f[target]
        assert all(pd.api.types.is_numeric_dtype(t) for t in x.dtypes)
        assert np.isfinite(x.to_numpy(dtype=float)).all() and not y.isna().any()
        context, rest = train_test_split(np.arange(len(f)), test_size=.4, stratify=y, random_state=42)
        compile_ids, rest = train_test_split(rest, test_size=.5, stratify=y.iloc[rest], random_state=43)
        validation, test = train_test_split(rest, test_size=.5, stratify=y.iloc[rest], random_state=44)
        counts = y.value_counts()
        assert 2 <= len(counts) <= 8 and len(context) > 128
        assert x.shape[1] < 2000 and len(context) < 50000
        for m in (8, 32, 128):
            for seed in (42, 43, 44):
                train_test_split(context, train_size=m, stratify=y.iloc[context], random_state=seed)
        datasets.append(dict(dataset=name, data=str(path), target=target, source_sha256=sha(path),
                             source_rows=len(f), N=len(context), d=x.shape[1], K=len(counts),
                             class_counts={str(k): int(v) for k, v in counts.items()},
                             class_ratio=float(counts.max()/counts.min()),
                             split_sizes=[len(context),len(compile_ids),len(validation),len(test)]))
    coverage = dict(binary=sum(d['K']==2 for d in datasets), multiclass=sum(d['K']>2 for d in datasets),
                    low_dimensional=sum(d['d']<15 for d in datasets), medium_high_dimensional=sum(d['d']>=15 for d in datasets),
                    small_context=sum(d['N']<1000 for d in datasets), larger_context=sum(d['N']>2000 for d in datasets),
                    balanced=sum(d['class_ratio']<=1.5 for d in datasets), imbalanced=sum(d['class_ratio']>=2 for d in datasets))
    assert coverage['binary'] >= 3 and coverage['multiclass'] >= 3
    assert all(v >= 2 for v in coverage.values())
    gate1 = json.loads((ARTIFACTS / "vehicle-fixed-labels/metadata.json").read_text())
    protocol = dict(registered_at=datetime.now(timezone.utc).isoformat(),
                    selection="Five eligible local complete numerical tasks plus three UCI numerical multiclass tasks; no Gate 2 predictions inspected. No row/feature filtering, imputation, or recoding of features. Published UCI train/test files concatenated in that order, then the Gate 1 split applied.",
                    balance_definition="max/min class count <=1.5 balanced; >=2 materially imbalanced; N denotes context rows",
                    datasets=datasets, coverage=coverage, downloaded_source_sha256=sources,
                    tabpfn_commit=gate1['tabpfn_git_commit'], checkpoint_sha256=gate1['checkpoint_sha256'],
                    checkpoint=gate1['config']['checkpoint'],
                    frozen=dict(budgets=[8,32,128], seeds=[42,43,44], steps=50, lr=.01, batch_size=64,
                                initialization="stratified", label_mode="fixed", eval_every=10,
                                optimizer="Adam default betas/eps/weight_decay", device="mps", threads=4,
                                split="60/20/10/10, stratified, sequential random_state=42,43,44",
                                selection="minimum compile KL at step 0,10,20,30,40,50; exactly 50 updates regardless",
                                inference="Gate 1 unchanged: one estimator, float32, random_state=2, differentiable input",
                                feature_constraint="context-only standardization; clamp context min/max +/-1 standardized unit"),
                    criterion=dict(total=72, required_strict_test_kl_wins=58, required_positive_dataset_means=6,
                                   median_delta_strictly_positive=True, mixed_dataset_count=5,
                                   failed_general_claim_dataset_count_max=4),
                    failures="No excluded datasets or runs; failed runs remain in denominator and are reported. Incomplete or invalid artifacts cannot support PASS.",
                    reload_tolerance=1e-5,
                    gate1_reuse="nine stratified Vehicle runs; existing KL reload verification; predictions reconstructed without reoptimization for direct prediction reload verification",
                    code_sha256={str(p):sha(p) for p in map(Path,[
                        REPO_ROOT / "scripts/tabpfn_compile_experiment.py",
                        REPO_ROOT / "scripts/verify_tabpfn_compile_artifacts.py"])})
    (ROOT / "preregistration.json").write_text(json.dumps(protocol, indent=2)+"\n")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
