from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import yaml

from . import __version__
from .errors import ArtifactError


@dataclass
class ModelArtifact:
    artifact_version: int
    pipeline: Any
    feature_names: list[str]
    classes: list[Any]
    schema: dict[str, Any]
    metadata: dict[str, Any]
    extra_columns: str = "ignore"


def create_run_dir(base: Path, name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = base / "runs" / f"{stamp}__{name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_model(path: Path, artifact: ModelArtifact) -> None:
    joblib.dump(artifact, path)


def load_model(path: str | Path) -> ModelArtifact:
    try: artifact = joblib.load(path)
    except Exception as exc: raise ArtifactError(f"No se pudo cargar el modelo: {exc}") from exc
    if not isinstance(artifact, ModelArtifact) or artifact.artifact_version != 1:
        raise ArtifactError("Artefacto incompatible; se esperaba artifact_version=1")
    return artifact


def environment() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version
    packages: dict[str, str | None] = {}
    for name in ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "pydantic", "joblib"]:
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "dependencies": packages, "amk": __version__}


def git_state() -> dict[str, Any]:
    def run(*args):
        return subprocess.run(["git", *args], text=True, capture_output=True, check=False).stdout.strip()
    return {"commit": run("rev-parse", "HEAD") or None, "branch": run("branch", "--show-current") or None,
            "dirty": bool(run("status", "--porcelain"))}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_base(run_dir: Path, cfg, schema: dict[str, Any], manifest: dict[str, Any]) -> None:
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(run_dir / "environment.json", environment()); write_json(run_dir / "dataset_schema.json", schema)
    write_json(run_dir / "manifest.json", {**manifest, "git": git_state()})


def html_report(run_dir: Path, title: str, sections: dict[str, Any]) -> None:
    import html
    blocks = [f"<h2>{html.escape(name)}</h2><pre>{html.escape(json.dumps(value, indent=2, ensure_ascii=False, default=str))}</pre>" for name, value in sections.items()]
    page = f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>body{{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}pre{{white-space:pre-wrap;background:#f4f4f4;padding:1rem}}</style></head><body><h1>{html.escape(title)}</h1>{''.join(blocks)}</body></html>"
    (run_dir / "report.html").write_text(page, encoding="utf-8")
