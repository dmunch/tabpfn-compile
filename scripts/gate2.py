"""Resume the frozen Gate 2 experiment, verify artifacts, and build its report."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from paths import ARTIFACTS, REPO_ROOT, resolve_checkpoint, resolve_data, tabpfn_repo


CONFIG = REPO_ROOT / "config"
CANONICAL_PREREGISTRATION = CONFIG / "gate2-preregistration.json"
CANONICAL_MIGRATION = CONFIG / "gate2-repository-migration.json"
DEFAULT_OUTPUT = ARTIFACTS / "gate2"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def inspect_experiment(directory: Path, dataset: dict, prereg: dict) -> list[str]:
    """Return reasons an experiment artifact is incomplete or incompatible."""
    if not directory.is_dir():
        return ["directory is missing"]
    required = ["metadata.json", "results.json", "teacher.npz", "traces.json"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        return ["missing " + ", ".join(missing)]
    try:
        metadata = read_json(directory / "metadata.json")
        results = read_json(directory / "results.json")
        traces = read_json(directory / "traces.json")
    except (OSError, ValueError) as error:
        return [f"unreadable metadata/results/traces: {error}"]

    issues = []
    config = metadata.get("config", {})
    if "completed_at" not in metadata:
        issues.append("metadata has no completed_at marker")
    for key in ("budgets", "seeds", "steps", "lr", "batch_size", "eval_every", "label_mode", "threads"):
        if config.get(key) != prereg["frozen"][key]:
            issues.append(f"configuration mismatch for {key}")
    if config.get("initializations") != ["stratified"]:
        issues.append("initialization is not exactly stratified")
    if metadata.get("source_sha256") != dataset["source_sha256"]:
        issues.append("dataset hash mismatch")
    if metadata.get("checkpoint_sha256") != prereg["checkpoint_sha256"]:
        issues.append("checkpoint hash mismatch")
    if metadata.get("tabpfn_git_commit") != prereg["tabpfn_commit"]:
        issues.append("TabPFN commit mismatch")

    for budget in prereg["frozen"]["budgets"]:
        for seed in prereg["frozen"]["seeds"]:
            pair = {
                row.get("method")
                for row in results
                if row.get("rows") == budget and row.get("seed") == seed
            }
            if not {"stratified", "synthetic_stratified"}.issubset(pair):
                issues.append(f"missing result pair for m={budget}, seed={seed}")
            expected_steps = list(range(prereg["frozen"]["eval_every"], prereg["frozen"]["steps"] + 1,
                                        prereg["frozen"]["eval_every"]))
            actual_steps = sorted(
                row.get("step") for row in traces
                if row.get("rows") == budget and row.get("seed") == seed
                and row.get("initialization") == "stratified"
            )
            if actual_steps != expected_steps:
                issues.append(f"incomplete trace for m={budget}, seed={seed}")
            for filename in (
                f"compiled-{budget}-{seed}-stratified.csv",
                f"predictions-{budget}-{seed}-synthetic_stratified.npz",
            ):
                if not (directory / filename).is_file():
                    issues.append(f"missing {filename}")
    return issues


def verification_state(directory: Path, prereg: dict) -> tuple[str, str]:
    expected = len(prereg["frozen"]["budgets"]) * len(prereg["frozen"]["seeds"])
    strict = directory / "artifact-verification.json"
    diagnostic = directory / "artifact-verification-diagnostic.json"
    for path, state in ((strict, "passed"), (diagnostic, "failed-recorded")):
        if not path.is_file():
            continue
        try:
            checks = read_json(path)
            if len(checks) != expected:
                return "invalid", f"{path.name} has {len(checks)} checks, expected {expected}"
            if state == "passed":
                maximum = max(max(row["kl_absolute_errors"].values()) for row in checks)
                if maximum >= prereg["reload_tolerance"]:
                    return "invalid", f"strict verification contains error {maximum:g}"
            return state, path.name
        except (KeyError, OSError, TypeError, ValueError) as error:
            return "invalid", f"cannot read {path.name}: {error}"
    return "missing", "no verification record"


class Gate2:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.output.resolve()
        self.prereg = read_json(CANONICAL_PREREGISTRATION)
        self.migration = read_json(CANONICAL_MIGRATION)
        self.tabpfn = tabpfn_repo()
        self.checkpoint = resolve_checkpoint(self.prereg["checkpoint"]).resolve()
        self.env = dict(os.environ)
        self.env["PYTHONPATH"] = str(self.tabpfn / "src")
        self.env.setdefault("MPLCONFIGDIR", "/tmp/tabpfn-compile-mpl")

    def preflight(self) -> None:
        if self.args.device != self.prereg["frozen"]["device"]:
            raise RuntimeError(
                f"Primary device {self.args.device!r} changes the frozen protocol; "
                f"expected {self.prereg['frozen']['device']!r}"
            )
        if self.migration["historical_code_sha256"] != self.prereg["code_sha256"]:
            raise RuntimeError("Migration record does not match the immutable preregistration")
        for relative, expected in self.migration["relocated_code_sha256"].items():
            path = REPO_ROOT / relative
            if sha256(path) != expected:
                raise RuntimeError(f"Frozen experiment code changed: {relative}")
        if not self.checkpoint.is_file() or sha256(self.checkpoint) != self.prereg["checkpoint_sha256"]:
            raise RuntimeError(f"Missing or changed checkpoint: {self.checkpoint}")
        head = subprocess.check_output(
            ["git", "-C", str(self.tabpfn), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != self.prereg["tabpfn_commit"]:
            raise RuntimeError(f"TabPFN commit is {head}, expected {self.prereg['tabpfn_commit']}")
        dirty = subprocess.check_output(
            ["git", "-C", str(self.tabpfn), "status", "--porcelain"], text=True
        ).strip()
        if dirty:
            raise RuntimeError(f"TabPFN worktree is dirty: {self.tabpfn}")
        for dataset in self.prereg["datasets"]:
            path = resolve_data(dataset["data"])
            if not path.is_file() or sha256(path) != dataset["source_sha256"]:
                raise RuntimeError(f"Missing or changed dataset: {path}")

    def sync_protocol_records(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for source, name in (
            (CANONICAL_PREREGISTRATION, "preregistration.json"),
            (CANONICAL_MIGRATION, "repository-migration.json"),
        ):
            destination = self.root / name
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise RuntimeError(f"Refusing to replace changed protocol record: {destination}")
            if not destination.exists():
                shutil.copy2(source, destination)

    def command(self, arguments: list[str], log: Path) -> int:
        try:
            display_log = log.relative_to(REPO_ROOT)
        except ValueError:
            display_log = log
        print(f"  log: {display_log}", flush=True)
        with log.open("w") as stream:
            return subprocess.run(
                [sys.executable, "-u", *arguments],
                cwd=REPO_ROOT,
                env=self.env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            ).returncode

    def preserve_failed_attempt(self, temporary: Path, dataset: str, device: str) -> Path | None:
        if not temporary.exists():
            return None
        failures = self.root / "failures"
        failures.mkdir(exist_ok=True)
        destination = failures / f"{dataset}-{device}-{timestamp()}-{uuid.uuid4().hex[:6]}"
        temporary.rename(destination)
        return destination

    def run_experiment(self, dataset: dict, destination: Path, device: str, log: Path) -> tuple[int, Path]:
        work = self.root / ".work"
        work.mkdir(exist_ok=True)
        temporary = work / f"{dataset['dataset']}-{device}-{uuid.uuid4().hex}"
        command = [
            str(REPO_ROOT / "scripts/tabpfn_compile_experiment.py"),
            "--data", str(resolve_data(dataset["data"])),
            "--target", dataset["target"],
            "--checkpoint", str(self.checkpoint),
            "--output", str(temporary),
            "--device", device,
            "--label-mode", "fixed",
            "--initializations", "stratified",
        ]
        code = self.command(command, log)
        if code == 0:
            issues = inspect_experiment(temporary, dataset, self.prereg)
            if issues:
                with log.open("a") as stream:
                    stream.write("\nDriver rejected incomplete output:\n" + "\n".join(issues) + "\n")
                code = 1
        if code == 0:
            temporary.rename(destination)
        return code, temporary

    def verify(self, dataset: str, destination: Path) -> tuple[int, int | None]:
        existing = [
            destination / name
            for name in ("artifact-verification.json", "artifact-verification-diagnostic.json")
            if (destination / name).exists()
        ]
        if existing:
            history = self.root / "verification-history" / dataset / f"{timestamp()}-{uuid.uuid4().hex[:6]}"
            history.mkdir(parents=True)
            for path in existing:
                path.rename(history / path.name)
        strict_log = self.root / f"{dataset}-verify.log"
        strict = self.command(
            [str(REPO_ROOT / "scripts/verify_tabpfn_compile_artifacts.py"), str(destination)],
            strict_log,
        )
        if strict == 0:
            return 0, None
        diagnostic_log = self.root / f"{dataset}-verify-diagnostic.log"
        diagnostic = self.command(
            [str(REPO_ROOT / "scripts/verify_tabpfn_compile_artifacts.py"), str(destination), "--diagnostic-only"],
            diagnostic_log,
        )
        return strict, diagnostic

    def execute_dataset(self, dataset: dict) -> dict:
        name = dataset["dataset"]
        destination = self.root / name
        started = datetime.now(timezone.utc).isoformat()
        issues = inspect_experiment(destination, dataset, self.prereg)
        selected = not self.args.dataset or name in self.args.dataset
        status = {"dataset": name, "started_at": started}

        if not selected:
            status.update(action="not-selected", experiment_exit_code=0 if not issues else 1)
        elif not issues and not self.args.force:
            print(f"SKIP {name}: experiment artifact is complete", flush=True)
            status.update(action="skipped-complete", experiment_exit_code=0)
        elif destination.exists() and not self.args.force:
            print(f"BLOCKED {name}: existing artifact is incomplete", flush=True)
            for issue in issues:
                print(f"  - {issue}", flush=True)
            status.update(action="blocked-incomplete", experiment_exit_code=1, issues=issues)
        else:
            backup = None
            if destination.exists():
                superseded = self.root / "superseded"
                superseded.mkdir(exist_ok=True)
                backup = superseded / f"{name}-{timestamp()}-{uuid.uuid4().hex[:6]}"
                destination.rename(backup)
                status["superseded_artifact"] = str(backup)
            print(f"RUN {name} on {self.args.device}", flush=True)
            log = self.root / f"{name}-run.log"
            code, temporary = self.run_experiment(dataset, destination, self.args.device, log)
            status.update(action="rerun" if backup else "created", primary_device=self.args.device,
                          experiment_exit_code=code)
            if code != 0:
                failure = self.preserve_failed_attempt(temporary, name, self.args.device)
                if failure:
                    status["failed_artifact"] = str(failure)
                log_text = log.read_text(errors="replace").lower()
                may_fallback = (
                    name == "landsat-satellite"
                    and self.args.device == "mps"
                    and "mps backend out of memory" in log_text
                )
                if may_fallback:
                    print(f"FALLBACK {name} to CPU after recorded MPS OOM", flush=True)
                    fallback_log = self.root / f"{name}-cpu-fallback.log"
                    fallback_code, fallback_temp = self.run_experiment(dataset, destination, "cpu", fallback_log)
                    status.update(fallback_device="cpu", fallback_exit_code=fallback_code,
                                  experiment_exit_code=fallback_code)
                    if fallback_code != 0:
                        failure = self.preserve_failed_attempt(fallback_temp, name, "cpu")
                        if failure:
                            status["fallback_failed_artifact"] = str(failure)
                if status["experiment_exit_code"] != 0 and backup and not destination.exists():
                    backup.rename(destination)
                    status["restored_artifact"] = str(destination)

        current_issues = inspect_experiment(destination, dataset, self.prereg)
        if not current_issues and selected and not self.args.no_verify:
            state, detail = verification_state(destination, self.prereg)
            if state in {"missing", "invalid"} or self.args.force_verification:
                print(f"VERIFY {name}", flush=True)
                strict, diagnostic = self.verify(name, destination)
                status["verification_exit_code"] = strict
                if diagnostic is not None:
                    status["diagnostic_verification_exit_code"] = diagnostic
            else:
                print(f"SKIP {name}: verification is {state}", flush=True)
                status["verification_exit_code"] = 0 if state == "passed" else 1
                status["verification_state"] = state
                status["verification_record"] = detail
        elif current_issues:
            status["issues"] = current_issues
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        return status

    def dry_run(self) -> int:
        print(f"Output: {self.root}")
        for dataset in self.prereg["datasets"]:
            name = dataset["dataset"]
            destination = self.root / name
            issues = inspect_experiment(destination, dataset, self.prereg)
            state, detail = verification_state(destination, self.prereg) if not issues else ("unavailable", "")
            selected = not self.args.dataset or name in self.args.dataset
            if not selected:
                action = "skip"
            elif self.args.force:
                action = "rerun"
            elif not destination.exists():
                action = "run"
            elif issues:
                action = "block"
            else:
                action = "skip"
            print(f"{name:36} experiment={action:5} verification={state:15} {detail}")
            for issue in issues:
                print(f"  - {issue}")
        return 0

    def run(self) -> int:
        self.preflight()
        if self.args.dry_run:
            return self.dry_run()
        self.sync_protocol_records()
        statuses = [self.execute_dataset(dataset) for dataset in self.prereg["datasets"]]
        pipeline_status = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "selected_datasets": self.args.dataset,
            "force": self.args.force,
            "force_verification": self.args.force_verification,
            "datasets": statuses,
        }
        (self.root / "pipeline-status.json").write_text(json.dumps(pipeline_status, indent=2) + "\n")

        incomplete = {
            dataset["dataset"]: inspect_experiment(self.root / dataset["dataset"], dataset, self.prereg)
            for dataset in self.prereg["datasets"]
        }
        incomplete = {name: issues for name, issues in incomplete.items() if issues}
        if incomplete:
            print("REPORT BLOCKED: incomplete experiment artifacts", file=sys.stderr)
            for name, issues in incomplete.items():
                print(f"  {name}: {'; '.join(issues)}", file=sys.stderr)
            return 1
        execution_status = self.root / "execution-status.json"
        if not execution_status.exists():
            execution_status.write_text(json.dumps(statuses, indent=2) + "\n")
        if self.args.no_report:
            return 0
        print("REPORT Gate 2", flush=True)
        code = self.command(
            [str(REPO_ROOT / "scripts/summarize_tabpfn_gate2.py"), "--root", str(self.root)],
            self.root / "report.log",
        )
        if code:
            print(f"Report generation failed; see {self.root / 'report.log'}", file=sys.stderr)
            return code
        decision = read_json(self.root / "gate-decision.json")
        print(
            f"DONE: {decision['verdict']} — {decision['wins']}/{decision['total_runs']} test-KL wins, "
            f"{decision['verified_artifacts']}/{decision['total_runs']} artifacts verified",
            flush=True,
        )
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Gate 2 artifact directory (default: artifacts/gate2)")
    parser.add_argument("--dataset", action="append",
                        help="Process only this dataset; repeat to select several")
    parser.add_argument("--device", default="mps",
                        help="Primary execution device from the frozen protocol (default: mps)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and show planned work without changing artifacts")
    parser.add_argument("--force", action="store_true",
                        help="Rerun selected experiments, preserving prior artifacts under superseded/")
    parser.add_argument("--force-verification", action="store_true",
                        help="Repeat verification even when a complete record exists")
    parser.add_argument("--no-verify", action="store_true", help="Do not fill missing verification records")
    parser.add_argument("--no-report", action="store_true", help="Do not regenerate aggregate outputs")
    args = parser.parse_args()
    known = {dataset["dataset"] for dataset in read_json(CANONICAL_PREREGISTRATION)["datasets"]}
    unknown = set(args.dataset or []) - known
    if unknown:
        parser.error("unknown dataset(s): " + ", ".join(sorted(unknown)))
    if args.no_verify and args.force_verification:
        parser.error("--no-verify and --force-verification are mutually exclusive")
    return args


if __name__ == "__main__":
    raise SystemExit(Gate2(parse_args()).run())
