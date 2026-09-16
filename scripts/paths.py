"""Repository paths plus compatibility for metadata created in nanotabicl."""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"
DATA = REPO_ROOT / "data"
WEIGHTS = REPO_ROOT / "weights"


def tabpfn_repo() -> Path:
    configured = os.environ.get("TABPFN_REPO")
    candidates = ([Path(configured)] if configured else []) + [
        REPO_ROOT.parent / "TabPFN",
        REPO_ROOT.parent / "tabpfn",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Set TABPFN_REPO to the TabPFN source checkout")


def resolve_data(path: str | Path) -> Path:
    path = Path(path)
    return path if path.exists() else DATA / path.name


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    return path if path.exists() else WEIGHTS / path.name


def resolve_artifact(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    parts = path.parts
    if "tabpfn-compile" in parts:
        suffix = parts[parts.index("tabpfn-compile") + 1 :]
        return ARTIFACTS.joinpath(*suffix)
    return ARTIFACTS / path.name
