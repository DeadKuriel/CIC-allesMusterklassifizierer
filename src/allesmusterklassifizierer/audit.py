from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import TabularDataset
from .errors import AuditError
from .v1config import ExperimentConfig


@dataclass
class AuditResult:
    dataset: TabularDataset
    report: dict[str, Any]
    cleaning_log: pd.DataFrame
    redundant: pd.DataFrame
    indiscernible: pd.DataFrame


def _pattern_groups(
    frame: pd.DataFrame, features: list[str]
) -> list[tuple[Any, pd.DataFrame]]:
    return list(frame.groupby(features, dropna=False, sort=False))


def audit_and_clean(dataset: TabularDataset, cfg: ExperimentConfig) -> AuditResult:
    frame = dataset.frame.copy()
    frame["__original_index__"] = dataset.original_index.to_numpy()
    log: list[dict[str, Any]] = []
    if frame[cfg.target].isna().any() and cfg.audit.target_missing == "drop_rows":
        for idx in frame.loc[frame[cfg.target].isna(), "__original_index__"]:
            log.append(
                {"original_index": idx, "action": "drop", "reason": "missing_target"}
            )
        frame = frame.loc[frame[cfg.target].notna()].copy()
    features = [c.name for c in cfg.columns if c.role == "feature"]
    redundant_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    drop_positions: set[Any] = set()
    mapping: dict[str, Any] = {}
    if cfg.audit.indiscernible_policy == "resolve_by_mapping":
        if (
            not cfg.audit.resolution_mapping
            or not cfg.audit.resolution_mapping.is_file()
        ):
            raise AuditError(
                "resolve_by_mapping requiere un archivo resolution_mapping existente"
            )
        mapped = pd.read_csv(cfg.audit.resolution_mapping)
        if not {"pattern_key", "resolved_target"}.issubset(mapped.columns):
            raise AuditError("El mapping requiere columnas pattern_key,resolved_target")
        mapping = dict(
            zip(
                mapped["pattern_key"].astype(str),
                mapped["resolved_target"],
                strict=True,
            )
        )
    for key, group in _pattern_groups(frame, features):
        labels = group[cfg.target].dropna().unique()
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_text = "|".join(map(str, key_tuple))
        if len(labels) == 1 and len(group) > 1:
            representative = group.iloc[
                -1 if cfg.audit.redundant_policy == "keep_last" else 0
            ]["__original_index__"]
            for _, row in group.iterrows():
                redundant_rows.append(
                    {
                        "pattern_key": key_text,
                        "original_index": row["__original_index__"],
                        "representative_index": representative,
                        "target": row[cfg.target],
                        "action": cfg.audit.redundant_policy,
                        "reason": "identical_X_and_y",
                    }
                )
            if cfg.audit.redundant_policy == "fail":
                raise AuditError(f"Patrones redundantes detectados para {key_text}")
            if cfg.audit.redundant_policy in {"keep_first", "keep_last"}:
                drop_positions.update(
                    group.index[group["__original_index__"] != representative]
                )
        elif len(labels) > 1:
            action = cfg.audit.indiscernible_policy
            for _, row in group.iterrows():
                conflict_rows.append(
                    {
                        "pattern_key": key_text,
                        "original_index": row["__original_index__"],
                        "target": row[cfg.target],
                        "action": action,
                        "reason": "identical_X_different_y",
                    }
                )
            if action == "fail":
                raise AuditError(f"Patrones indiscernibles detectados para {key_text}")
            if action == "remove_all":
                drop_positions.update(group.index)
            elif action == "resolve_by_mapping":
                if key_text not in mapping:
                    raise AuditError(
                        f"No hay resolución manual para pattern_key={key_text}"
                    )
                frame.loc[group.index, cfg.target] = mapping[key_text]
    for pos in sorted(drop_positions):
        log.append(
            {
                "original_index": frame.loc[pos, "__original_index__"],
                "action": "drop",
                "reason": "pattern_policy",
            }
        )
    frame = frame.drop(index=list(drop_positions)).copy()
    original = frame.pop("__original_index__").reset_index(drop=True)
    frame = frame.reset_index(drop=True)
    frame.attrs.update(dataset.frame.attrs)
    cleaned = TabularDataset(frame, original, dataset.sha256, dataset.source)
    columns = []
    for col in cfg.columns:
        if col.name not in frame:
            continue
        s = frame[col.name]
        columns.append(
            {
                "column": col.name,
                "role": col.role,
                "semantic_type": col.semantic_type,
                "physical_type": str(s.dtype),
                "missing_count": int(s.isna().sum()),
                "missing_ratio": float(s.isna().mean()),
                "unique_count": int(s.nunique(dropna=True)),
            }
        )
    counts = frame[cfg.target].value_counts(dropna=False)
    ratio = float(counts.max() / counts.min()) if len(counts) and counts.min() else None
    report = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "column_summary": columns,
        "class_distribution": {str(k): int(v) for k, v in counts.items()},
        "imbalance_ratio": ratio,
        "constant_columns": [c for c in frame if frame[c].nunique(dropna=False) == 1],
        "near_constant_columns": [
            c
            for c in frame
            if frame[c].value_counts(normalize=True, dropna=False).max()
            >= cfg.audit.near_constant_threshold
        ],
        "high_cardinality_identifiers": [
            c.name
            for c in cfg.columns
            if c.role == "feature" and frame[c.name].nunique() / len(frame) > 0.9
        ],
        "redundant_groups": len({r["pattern_key"] for r in redundant_rows}),
        "indiscernible_groups": len({r["pattern_key"] for r in conflict_rows}),
        "warnings": (
            [f"Desbalance {ratio:.2f}:1"]
            if ratio and ratio >= cfg.audit.imbalance_ratio_warning
            else []
        ),
    }
    return AuditResult(
        cleaned,
        report,
        pd.DataFrame(log),
        pd.DataFrame(redundant_rows),
        pd.DataFrame(conflict_rows),
    )


def write_audit(result: AuditResult, directory: Path) -> None:
    import json

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "data_audit.json").write_text(
        json.dumps(result.report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result.cleaning_log.to_csv(directory / "cleaning_log.csv", index=False)
    result.redundant.to_csv(directory / "redundant_patterns.csv", index=False)
    result.indiscernible.to_csv(directory / "indiscernible_patterns.csv", index=False)
