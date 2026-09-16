"""Execute the immutable Gate 2 preregistration, preserving failures."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from paths import ARTIFACTS, REPO_ROOT, resolve_checkpoint, resolve_data, tabpfn_repo


ROOT = ARTIFACTS / "gate2"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    prereg = json.loads((ROOT / "preregistration.json").read_text())
    frozen_paths = [Path(path) for path in prereg["code_sha256"]]
    if all(path.exists() for path in frozen_paths):
        for path, expected in prereg["code_sha256"].items():
            assert sha(path) == expected, f"Frozen code changed: {path}"
    else:
        migration = json.loads((ROOT / "repository-migration.json").read_text())
        assert migration["historical_code_sha256"] == prereg["code_sha256"]
        for path, expected in migration["relocated_code_sha256"].items():
            assert sha(REPO_ROOT / path) == expected, f"Relocated code changed: {path}"
    checkpoint = resolve_checkpoint(prereg["checkpoint"])
    tabpfn = tabpfn_repo()
    assert sha(checkpoint) == prereg["checkpoint_sha256"]
    assert subprocess.check_output(["git", "-C", str(tabpfn), "rev-parse", "HEAD"], text=True).strip() == prereg["tabpfn_commit"]
    assert not subprocess.check_output(["git", "-C", str(tabpfn), "status", "--porcelain"], text=True).strip()
    env = dict(os.environ, PYTHONPATH=str(tabpfn / "src"),
               MPLCONFIGDIR="/tmp/tabpfn-compile-mpl")
    statuses = []

    def run(command, logfile):
        with logfile.open("w") as log:
            return subprocess.run([sys.executable, "-u"] + command, env=env,
                                  stdout=log, stderr=subprocess.STDOUT).returncode

    for dataset in prereg["datasets"]:
        name = dataset["dataset"]
        assert sha(resolve_data(dataset["data"])) == dataset["source_sha256"]
        destination = ROOT / name
        status = dict(dataset=name, started_at=datetime.now(timezone.utc).isoformat())
        print(f"START {name}", flush=True)
        if name == "vehicle":
            source = ARTIFACTS / "vehicle-fixed-labels"
            destination.mkdir(exist_ok=False)
            for filename in ("metadata.json", "teacher.npz"):
                shutil.copy2(source / filename, destination / filename)
            original = json.loads((source / "results.json").read_text())
            rows = [r for r in original if r["method"] in ("full", "stratified", "synthetic_stratified")]
            (destination / "results.json").write_text(json.dumps(rows, indent=2)+"\n")
            meta = json.loads((destination / "metadata.json").read_text())
            meta["config"]["initializations"] = ["stratified"]
            meta["reused_from"] = str(source)
            meta["original_results_sha256"] = sha(source / "results.json")
            meta["prediction_reference_origin"] = "Reconstructed from Gate 1 CSVs after original KL checks, not retained original prediction arrays"
            (destination / "metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
            for file in source.glob("compiled-*-stratified.csv"):
                shutil.copy2(file, destination / file.name)
            traces = [r for r in json.loads((source / "traces.json").read_text()) if r["initialization"] == "stratified"]
            (destination / "traces.json").write_text(json.dumps(traces, indent=2)+"\n")
            status["experiment_exit_code"] = 0
            status["reference_exit_code"] = run([
                str(REPO_ROOT / "scripts/verify_tabpfn_compile_artifacts.py"), str(destination), "--record-reference"],
                ROOT / f"{name}-reference.log")
        else:
            status["experiment_exit_code"] = run([
                str(REPO_ROOT / "scripts/tabpfn_compile_experiment.py"), "--data", str(resolve_data(dataset["data"])),
                "--target", dataset["target"], "--output", str(destination), "--device", "mps",
                "--label-mode", "fixed", "--initializations", "stratified"], ROOT / f"{name}-run.log")
        if status["experiment_exit_code"] == 0:
            print(f"VERIFY {name}", flush=True)
            status["verification_exit_code"] = run([
                str(REPO_ROOT / "scripts/verify_tabpfn_compile_artifacts.py"), str(destination)],
                ROOT / f"{name}-verify.log")
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        statuses.append(status)
        (ROOT / "execution-status.json").write_text(json.dumps(statuses, indent=2)+"\n")
        print(json.dumps(status), flush=True)
    print("DONE: all eight datasets attempted; no exclusions", flush=True)


if __name__ == "__main__":
    main()
