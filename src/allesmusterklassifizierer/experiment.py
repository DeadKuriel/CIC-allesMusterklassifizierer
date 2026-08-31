from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .artifacts import (
    ModelArtifact,
    create_run_dir,
    html_report,
    save_model,
    write_base,
    write_json,
)
from .audit import audit_and_clean, write_audit
from .dataset import load_dataset
from .evaluation import classification_metrics, confusion_tables
from .models import REGISTRY, ALIASES, create_model
from .preprocessing import build_pipeline
from .splitting import DataSplit, make_splits
from .v1config import ExperimentConfig, ModelConfig


def prepare(cfg: ExperimentConfig):
    dataset = load_dataset(cfg)
    audit = audit_and_clean(dataset, cfg)
    frame = audit.dataset.frame
    y = frame[cfg.target]
    group_col = cfg.column("group")
    time_col = cfg.column("timestamp")
    splits = make_splits(
        cfg.split,
        y,
        groups=frame[group_col.name] if group_col else None,
        timestamps=frame[time_col.name] if time_col else None,
        seed=cfg.seed,
    )
    return audit, splits


def _cv(splits: list[DataSplit]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(s.train_idx, s.test_idx) for s in splits]


def _fit(
    cfg: ExperimentConfig,
    model_cfg: ModelConfig,
    X: Any,
    y: Any,
    splits: list[DataSplit],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    estimator = create_model(model_cfg.name, model_cfg.params, cfg.seed)
    pipeline = build_pipeline(cfg, estimator)
    canonical_name = ALIASES.get(model_cfg.name) or model_cfg.name
    space = (
        cfg.tuning.spaces.get(model_cfg.name)
        or REGISTRY[canonical_name].search_space
    )
    space = {f"model__{k}": v for k, v in space.items()}
    if cfg.tuning.method == "none":
        pipeline.fit(X, y)
        return pipeline, {}, []
    if cfg.tuning.method in {"grid", "halving"}:
        search: Any = GridSearchCV(
            pipeline,
            space,
            scoring=cfg.evaluation.primary_metric,
            cv=_cv(splits),
            n_jobs=cfg.tuning.n_jobs,
            refit=True,
            error_score=cfg.tuning.error_score,
            return_train_score=False,
        )
    else:
        search = RandomizedSearchCV(
            pipeline,
            space,
            n_iter=cfg.tuning.n_iter,
            scoring=cfg.evaluation.primary_metric,
            cv=_cv(splits),
            n_jobs=cfg.tuning.n_jobs,
            refit=True,
            random_state=cfg.seed,
            error_score=cfg.tuning.error_score,
            return_train_score=False,
        )
    search.fit(X, y)
    cv_records = [
        {str(key): value for key, value in row.items()}
        for row in pd.DataFrame(search.cv_results_).to_dict("records")
    ]
    return (
        search.best_estimator_,
        search.best_params_,
        cv_records,
    )


def run(cfg: ExperimentConfig, *, compare=False) -> Path:
    # Everything above this line is validation/read-only; no partial run directory exists on invalid input.
    audit, splits = prepare(cfg)
    run_dir = create_run_dir(cfg.output_dir, cfg.run_name)
    started = time.perf_counter()
    write_audit(audit, run_dir)
    X, y = audit.dataset.X, audit.dataset.y
    final = splits[0]
    # A final holdout is isolated. CV/tuning receives only train and remapped indices.
    final_holdout = cfg.split.strategy in {
        "holdout",
        "stratified_holdout",
        "train_validation_test",
        "group_holdout",
        "temporal",
    }
    train_idx, test_idx = (
        (final.train_idx, final.test_idx)
        if final_holdout
        else (np.arange(len(y)), np.array([], dtype=int))
    )
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    if final_holdout:
        inner_cfg = cfg.split.model_copy(
            update={"strategy": "stratified_kfold", "n_splits": cfg.tuning.cv}
        )
        try:
            tuning_splits = make_splits(inner_cfg, y_train, seed=cfg.seed)
        except Exception:
            if cfg.tuning.method != "none":
                raise
            tuning_splits = []
    else:
        tuning_splits = splits
    rows = []
    fold_rows = []
    model_metrics = {}
    chosen = None
    chosen_metric = -np.inf
    best_payload = None
    for mc in [m for m in cfg.models if m.enabled]:
        model_label = mc.label or mc.name
        t0 = time.perf_counter()
        eval_X: pd.DataFrame | None = None
        try:
            if final_holdout:
                pipeline, params, cv_results = _fit(
                    cfg, mc, X_train, y_train, tuning_splits
                )
                fit_time = time.perf_counter() - t0
                p0 = time.perf_counter()
                evaluation_idx = test_idx
                eval_X = X.iloc[evaluation_idx]
                eval_y = y.iloc[evaluation_idx]
                pred = pipeline.predict(eval_X)
                predict_time = time.perf_counter() - p0
                probs = (
                    pipeline.predict_proba(eval_X)
                    if hasattr(pipeline, "predict_proba")
                    else None
                )
            else:
                if cfg.tuning.method != "none":
                    raise ValueError(
                        "K-fold externo con tuning requiere validación cruzada anidada"
                    )
                fold_true = []
                fold_pred = []
                fold_prob = []
                fold_indices = []
                fit_time = 0.0
                predict_time = 0.0
                for sp in splits:
                    f0 = time.perf_counter()
                    fold_pipeline, _, _ = _fit(
                        cfg, mc, X.iloc[sp.train_idx], y.iloc[sp.train_idx], []
                    )
                    fit_time += time.perf_counter() - f0
                    p0 = time.perf_counter()
                    fold_prediction = fold_pipeline.predict(X.iloc[sp.test_idx])
                    predict_time += time.perf_counter() - p0
                    fold_probability = (
                        fold_pipeline.predict_proba(X.iloc[sp.test_idx])
                        if hasattr(fold_pipeline, "predict_proba")
                        else None
                    )
                    fold_score = (
                        fold_probability[:, 1]
                        if fold_probability is not None
                        and fold_probability.shape[1] == 2
                        else None
                    )
                    fm = classification_metrics(
                        y.iloc[sp.test_idx],
                        fold_prediction,
                        labels=list(fold_pipeline.classes_),
                        scores=fold_score,
                        probabilities=fold_probability,
                        average=cfg.evaluation.average,
                        positive_class=cfg.evaluation.positive_class,
                        zero_division=cfg.evaluation.zero_division,
                    )
                    fold_rows.append(
                        {
                            "model": model_label,
                            "algorithm": mc.name,
                            "fold": sp.fold,
                            "train_size": len(sp.train_idx),
                            "test_size": len(sp.test_idx),
                            **{
                                k: v
                                for k, v in fm.items()
                                if k not in {"per_class", "labels"}
                            },
                        }
                    )
                    fold_true.append(y.iloc[sp.test_idx].to_numpy())
                    fold_pred.append(fold_prediction)
                    fold_indices.append(sp.test_idx)
                    if fold_probability is not None:
                        fold_prob.append(fold_probability)
                f0 = time.perf_counter()
                pipeline, params, cv_results = _fit(cfg, mc, X, y, [])
                fit_time += time.perf_counter() - f0
                evaluation_idx = np.concatenate(fold_indices)
                eval_y = pd.Series(np.concatenate(fold_true))
                pred = np.concatenate(fold_pred)
                probs = (
                    np.concatenate(fold_prob) if len(fold_prob) == len(splits) else None
                )
            if final_holdout and eval_X is not None and hasattr(
                pipeline, "decision_function"
            ):
                scores = pipeline.decision_function(eval_X)
            else:
                scores = (
                    probs[:, 1]
                    if probs is not None and probs.shape[1] == 2
                    else None
                )
            labels = list(pipeline.classes_)
            metrics = classification_metrics(
                eval_y,
                pred,
                labels=labels,
                scores=scores,
                probabilities=probs,
                average=cfg.evaluation.average,
                positive_class=cfg.evaluation.positive_class,
                zero_division=cfg.evaluation.zero_division,
            )
            score = float(
                metrics.get(cfg.evaluation.primary_metric, metrics["balanced_accuracy"])
            )
            model_metrics[model_label] = metrics
            rows.append(
                {
                    "model": model_label,
                    "algorithm": mc.name,
                    "selection_metric": score,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "mcc": metrics["mcc"],
                    "cohen_kappa": metrics["cohen_kappa"],
                    "roc_auc": metrics.get("roc_auc"),
                    "pr_auc": metrics.get("pr_auc"),
                    "log_loss": metrics.get("log_loss"),
                    "fit_seconds": fit_time,
                    "predict_seconds": predict_time,
                    "best_params": json.dumps(params),
                    "status": "success",
                    "error": "",
                }
            )
            if score > chosen_metric:
                chosen_metric = score
                chosen = model_label
                best_payload = (
                    pipeline,
                    params,
                    metrics,
                    eval_y,
                    pred,
                    probs,
                    evaluation_idx,
                    cv_results,
                )
        except Exception as exc:
            rows.append({"model": mc.name, "status": "failed", "error": str(exc)})
            if not compare:
                raise
    if best_payload is None:
        raise RuntimeError("Ningún modelo pudo entrenarse")
    pipeline, params, metrics, eval_y, pred, probs, evaluation_idx, cv_results = (
        best_payload
    )
    pd.DataFrame(rows).to_csv(run_dir / "cv_results.csv", index=False)
    pd.DataFrame(fold_rows or cv_results).to_csv(
        run_dir / "fold_metrics.csv", index=False
    )
    pred_frame = pd.DataFrame(
        {
            "original_index": audit.dataset.original_index.iloc[
                evaluation_idx
            ].to_numpy(),
            "y_true": eval_y.to_numpy(),
            "y_pred": pred,
        }
    )
    if probs is not None:
        for i, c in enumerate(pipeline.classes_):
            pred_frame[f"probability_{c}"] = probs[:, i]
    pred_frame.to_csv(run_dir / "predictions.csv", index=False)
    cm, cmr, cmc = confusion_tables(eval_y, pred, list(pipeline.classes_))
    cm.to_csv(run_dir / "confusion_matrix.csv")
    cmr.to_csv(run_dir / "confusion_matrix_normalized_rows.csv")
    cmc.to_csv(run_dir / "confusion_matrix_normalized_columns.csv")
    write_json(run_dir / "final_metrics.json", metrics)
    write_json(run_dir / "model_metrics.json", model_metrics)
    write_json(
        run_dir / "splits.json",
        [
            {
                "fold": s.fold,
                "train": s.train_idx.tolist(),
                "validation": (
                    s.validation_idx.tolist() if s.validation_idx is not None else None
                ),
                "test": s.test_idx.tolist(),
            }
            for s in splits
        ],
    )
    schema = {
        "target": cfg.target,
        "feature_names": audit.dataset.feature_names,
        "columns": [c.model_dump() for c in cfg.columns],
        "classes": list(map(str, pipeline.classes_)),
    }
    if cfg.persistence.save_pipeline:
        save_model(
            run_dir / "pipeline.joblib",
            ModelArtifact(
                1,
                pipeline,
                audit.dataset.feature_names,
                list(pipeline.classes_),
                schema,
                {
                    "model": chosen,
                    "params": params,
                    "dataset_sha256": audit.dataset.sha256,
                },
                cfg.persistence.extra_columns,
            ),
        )
    manifest = {
        "status": "success",
        "dataset": {"path": str(audit.dataset.source), "sha256": audit.dataset.sha256},
        "seed": cfg.seed,
        "selected_model": chosen,
        "metrics": metrics,
        "duration_seconds": time.perf_counter() - started,
    }
    write_base(run_dir, cfg, schema, manifest)
    if cfg.persistence.report_html:
        html_report(
            run_dir,
            cfg.run_name,
            {
                "Resumen ejecutivo": manifest,
                "Calidad del dataset": audit.report,
                "Comparación": rows,
                "Métricas por fold": fold_rows,
                "Métricas por modelo": model_metrics,
                "Métricas del modelo seleccionado": metrics,
                "Reproducibilidad": {"seed": cfg.seed, "splits": "splits.json"},
            },
        )
    return run_dir
