from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def now_timestamp() -> str:
    # Ej: 20251231-153012
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        f.write(text)


def read_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def parse_scalar(value: str) -> Any:
    """
    Parse simple CLI override values.
    Supports:
      - yaml-like scalars: true/false/null, numbers
      - inline lists/dicts via yaml.safe_load if starts with '[' or '{'
      - default: string
    """
    v = value.strip()
    if v.startswith("[") or v.startswith("{"):
        return yaml.safe_load(v)

    low = v.lower()
    if low in ("true", "false", "null", "none"):
        return yaml.safe_load(low)

    # try int
    try:
        if v.startswith("0") and len(v) > 1 and v[1].isdigit():
            raise ValueError
        return int(v)
    except ValueError:
        pass

    # try float
    try:
        return float(v)
    except ValueError:
        pass

    return v


def set_by_dotted_path(d: Dict[str, Any], dotted_path: str, value: Any) -> None:
    """
    Set nested dict value by path like 'classifier.params.k'.
    Creates dicts if needed.
    """
    keys = dotted_path.split(".")
    cur: Dict[str, Any] = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    meta_json: Path
    predictions_csv: Path
    fold_predictions_csv: Path
    confusion_csv: Path
    confusion_norm_rows_csv: Path
    confusion_norm_cols_csv: Path
    metrics_json: Path

    @staticmethod
    def create(
        base_outputs_dir: str | Path, run_name: str, timestamp: str
    ) -> "RunPaths":
        run_dir = ensure_dir(
            Path(base_outputs_dir) / "runs" / f"{timestamp}__{run_name}"
        )
        return RunPaths(
            run_dir=run_dir,
            meta_json=run_dir / "meta.json",
            predictions_csv=run_dir / "predictions.csv",
            fold_predictions_csv=run_dir / "fold_predictions.csv",
            confusion_csv=run_dir / "confusion_matrix.csv",
            confusion_norm_rows_csv=run_dir / "confusion_matrix_norm_rows.csv",
            confusion_norm_cols_csv=run_dir / "confusion_matrix_norm_cols.csv",
            metrics_json=run_dir / "metrics.json",
        )
