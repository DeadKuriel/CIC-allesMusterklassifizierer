from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .errors import ClassifierError, ConfigError
from .io import Dataset, load_csv_dataset
from .metrics import (
    compute_metrics,
    confusion_matrix_df,
    normalize_by_columns,
    normalize_by_rows,
)
from .utils import RunPaths, now_timestamp, write_json
from .validation import HoldoutSplitter, LeaveOneOutSplitter, KFoldSplitter
from .metrics.visualization import save_confusion_matrix_display

def _confmat_display_enabled(cfg: dict) -> bool:
    report = cfg.get("report", {}) or {}
    return bool(report.get("confusion_matrix_display", False))

def _export_partitions_enabled(cfg: dict) -> bool:
    validation = cfg.get("validation", {}) or {}
    export = validation.get("export", {}) or {}
    return bool(export.get("csv", False))

def _build_splitter(vcfg: Dict[str, Any]):
    vtype = vcfg.get("type")
    params = vcfg.get("params", {}) or {}

    if vtype == "holdout":
        return HoldoutSplitter(
            test_size=float(params.get("test_size", 0.2)),
            seed=int(params.get("seed", 42)),
            stratify=bool(params.get("stratify", True)),
        )

    if vtype == "loo":
        return LeaveOneOutSplitter()

    if vtype in ("kfold", "k-fold", "k_fold"):
        return KFoldSplitter(
            n_splits=int(params.get("n_splits", 5)),
            seed=int(params.get("seed", 42)),
            shuffle=bool(params.get("shuffle", True)),
            stratify=bool(params.get("stratify", True)),
        )

    raise ConfigError(f"validation.type inválido: {vtype}. Usa 'holdout', 'loo' o 'kfold'.")


def _build_classifier(ccfg: Dict[str, Any]):
    ctype = ccfg.get("type")
    params = ccfg.get("params", {}) or {}

    if ctype == "euclidiano":
        if "distance" in params:
            raise ClassifierError(
                "El clasificador 'euclidiano' usa distancia L2 fija. "
                "No se permite classifier.params.distance en el YAML."
            )
        from .classifiers.euclidean_centroid import EuclideanCentroidClassifier
        return EuclideanCentroidClassifier()

    if ctype == "knn":
        from .classifiers.knn import KNNClassifier
        k = params.get("k", 1)
        dist = params.get("distance", "euclidean")
        return KNNClassifier(k=int(k), distance=dist)

    raise ConfigError(f"classifier.type inválido: {ctype}. Usa 'euclidiano' o 'knn'.")


def _dataset_from_cfg(cfg: Dict[str, Any]) -> Dataset:
    io_cfg = cfg.get("io", {}) or {}
    dataset_path = io_cfg.get("dataset")
    target_col = io_cfg.get("target_col", "class")
    drop_cols = io_cfg.get("drop_cols", []) or []

    if not dataset_path:
        raise ConfigError("Falta io.dataset (ruta CSV).")
    return load_csv_dataset(str(dataset_path), target_col=str(target_col), drop_cols=list(drop_cols))


def validate_only(cfg: Dict[str, Any], outputs_dir: str = "outputs") -> RunPaths:
    run_name = str(cfg.get("run_name", "validate"))
    ts = now_timestamp()
    paths = RunPaths.create(outputs_dir, run_name, ts)

    ds = _dataset_from_cfg(cfg)
    splitter = _build_splitter(cfg.get("validation", {}) or {})

    # stats
    class_counts = {str(c): int(np.sum(ds.y == c)) for c in ds.classes}
    folds = []
    for split in splitter.split(ds.y):
        ytr = ds.y[split.train_idx]
        yte = ds.y[split.test_idx]
        folds.append(
            {
                "fold": int(split.fold),
                "train_size": int(split.train_idx.size),
                "test_size": int(split.test_idx.size),
                "train_class_counts": {str(c): int(np.sum(ytr == c)) for c in ds.classes},
                "test_class_counts": {str(c): int(np.sum(yte == c)) for c in ds.classes},
            }
        )

    # Export CSV partitions
    if _export_partitions_enabled(cfg):
        partitions_dir = paths.run_dir / "partitions"
        partitions_dir.mkdir(parents=True, exist_ok=True)

        # reconstruimos DataFrame completo original
        full_df = pd.read_csv(cfg["io"]["dataset"])

        drop_cols = cfg["io"].get("drop_cols", []) or []

        for split in splitter.split(ds.y):
            fold_dir = partitions_dir / f"fold_{split.fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_df = full_df.iloc[split.train_idx].drop(columns=drop_cols, errors="ignore")
            test_df = full_df.iloc[split.test_idx].drop(columns=drop_cols, errors="ignore")

            train_df.to_csv(fold_dir / "train.csv", index=False)
            test_df.to_csv(fold_dir / "test.csv", index=False)


    meta = {
        "mode": "validate",
        "run_name": run_name,
        "timestamp": ts,
        "config": cfg,
        "dataset": {
            "path": cfg.get("io", {}).get("dataset"),
            "n_samples": int(ds.X.shape[0]),
            "n_features": int(ds.X.shape[1]),
            "target_col": ds.target_name,
            "classes": [str(c) for c in ds.classes],
            "class_counts": class_counts,
        },
        "partitions": {
            "exported": _export_partitions_enabled(cfg),
            "format": "csv",
            "path": "partitions/"
        },
        "folds": folds[:50] if len(folds) > 50 else folds,
        "folds_truncated": len(folds) > 50,
        "n_folds": len(folds),
    }
    write_json(paths.meta_json, meta)
    return paths


def classify_only(cfg: Dict[str, Any], outputs_dir: str = "outputs") -> RunPaths:
    """
    Support:
      A) io.train_dataset + io.test_dataset
      B) io.dataset + mode.dataset_split.split = holdout/loo/kfold (classify-only 1 split)
    """
    run_name = str(cfg.get("run_name", "classify"))
    ts = now_timestamp()
    paths = RunPaths.create(outputs_dir, run_name, ts)

    io_cfg = cfg.get("io", {}) or {}
    target_col = str(io_cfg.get("target_col", "class"))
    drop_cols = list((io_cfg.get("drop_cols", []) or []))

    classifier = _build_classifier(cfg.get("classifier", {}) or {})

    train_path = io_cfg.get("train_dataset")
    test_path = io_cfg.get("test_dataset")

    if train_path and test_path:
        train_ds = load_csv_dataset(str(train_path), target_col=target_col, drop_cols=drop_cols)
        test_ds = load_csv_dataset(str(test_path), target_col=target_col, drop_cols=drop_cols)
        if train_ds.feature_names != test_ds.feature_names:
            raise ConfigError(f"Esquema train/test incompatible: train={train_ds.feature_names}, test={test_ds.feature_names}")
        _run_single_split(
            cfg=cfg,
            paths=paths,
            classifier=classifier,
            train_ds=train_ds,
            test_ds=test_ds,
            dataset_labels=_merge_labels(train_ds.classes, test_ds.classes),
            mode_name="classify_train_test",
        )
        return paths

    # dataset + split inside
    ds = _dataset_from_cfg(cfg)

    mode_cfg = cfg.get("mode", {}) or {}
    if mode_cfg.get("type") != "dataset_split":
        raise ConfigError(
            "Para classify-only debes configurar io.train_dataset+io.test_dataset "
            "o mode.type=dataset_split con mode.split."
        )

    split_cfg = mode_cfg.get("split", {}) or {}
    splitter = _build_splitter(split_cfg)

    splits = list(splitter.split(ds.y))
    if len(splits) != 1:
        raise ConfigError(
            f"En classify-only se espera 1 split (típicamente holdout). "
            f"Recibido n_splits={len(splits)}. Para kfold/loo usa 'amk run'."
        )
    sp = splits[0]

    train_ds = _subset_dataset(ds, sp.train_idx)
    test_ds = _subset_dataset(ds, sp.test_idx)

    _run_single_split(
        cfg=cfg,
        paths=paths,
        classifier=classifier,
        train_ds=train_ds,
        test_ds=test_ds,
        dataset_labels=ds.classes,
        mode_name="classify_dataset_split",
        original_index_test=test_ds.index,
    )
    return paths


def run_experiment(cfg: Dict[str, Any], outputs_dir: str = "outputs") -> RunPaths:
    run_name = str(cfg.get("run_name", "run"))
    ts = now_timestamp()
    paths = RunPaths.create(outputs_dir, run_name, ts)

    ds = _dataset_from_cfg(cfg)
    splitter = _build_splitter(cfg.get("validation", {}) or {})
    classifier = _build_classifier(cfg.get("classifier", {}) or {})

    fold_rows = []
    all_true = []
    all_pred = []
    all_test_indices = []

    for split in splitter.split(ds.y):
        Xtr, ytr = ds.X[split.train_idx], ds.y[split.train_idx]
        Xte, yte = ds.X[split.test_idx], ds.y[split.test_idx]

        classifier_fold = _clone_classifier(classifier)
        classifier_fold.fit(Xtr, ytr)
        yhat = classifier_fold.predict(Xte)

        all_true.append(yte)
        all_pred.append(yhat)
        all_test_indices.append(ds.index[split.test_idx])

        for j in range(yte.shape[0]):
            fold_rows.append(
                {
                    "fold": int(split.fold),
                    "index": int(ds.index[split.test_idx][j]),
                    "y_true": yte[j],
                    "y_pred": yhat[j],
                }
            )

    y_true_all = np.concatenate(all_true) if all_true else np.array([], dtype=object)
    y_pred_all = np.concatenate(all_pred) if all_pred else np.array([], dtype=object)

    fold_pred_df = pd.DataFrame(fold_rows)
    fold_pred_df.to_csv(paths.fold_predictions_csv, index=False)

    cm_df = confusion_matrix_df(y_true_all, y_pred_all, labels=ds.classes)
    cm_norm_rows = normalize_by_rows(cm_df)
    cm_norm_cols = normalize_by_columns(cm_df)

    cm_df.to_csv(paths.confusion_csv, index=True)
    cm_norm_rows.to_csv(paths.confusion_norm_rows_csv, index=True)
    cm_norm_cols.to_csv(paths.confusion_norm_cols_csv, index=True)
    
        # ConfusionMatrixDisplay
    if _confmat_display_enabled(cfg):
        labels = [str(l) for l in ds.classes]

        save_confusion_matrix_display(
            cm_df=cm_df,
            labels=labels,
            out_path=str(paths.run_dir / "confusion_matrix.png"),
            title="Confusion Matrix",
            normalize=False,
        )

        save_confusion_matrix_display(
            cm_df=cm_norm_rows,
            labels=labels,
            out_path=str(paths.run_dir / "confusion_matrix_norm_rows.png"),
            title="Confusion Matrix (Normalized by Rows)",
            normalize=True,
        )

        save_confusion_matrix_display(
            cm_df=cm_norm_cols,
            labels=labels,
            out_path=str(paths.run_dir / "confusion_matrix_norm_cols.png"),
            title="Confusion Matrix (Normalized by Columns)",
            normalize=True,
        )

    metrics = compute_metrics(cm_df, labels=ds.classes, y_true=y_true_all, y_pred=y_pred_all)
    write_json(paths.metrics_json, metrics)

    meta = {
        "mode": "run",
        "run_name": run_name,
        "timestamp": ts,
        "config": cfg,
        "dataset": {
            "path": cfg.get("io", {}).get("dataset"),
            "n_samples": int(ds.X.shape[0]),
            "n_features": int(ds.X.shape[1]),
            "target_col": ds.target_name,
            "classes": [str(c) for c in ds.classes],
        },
        "n_folds": int(len(set(int(r["fold"]) for r in fold_rows))) if fold_rows else 0,
        "outputs": {
            "fold_predictions_csv": str(paths.fold_predictions_csv),
            "confusion_matrix_csv": str(paths.confusion_csv),
            "confusion_matrix_norm_rows_csv": str(paths.confusion_norm_rows_csv),
            "confusion_matrix_norm_cols_csv": str(paths.confusion_norm_cols_csv),
            "metrics_json": str(paths.metrics_json),
        },
        "summary": {
            "accuracy": metrics.get("accuracy", 0.0),
            "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
            "mcc": metrics.get("mcc", 0.0),
        },
    }
    write_json(paths.meta_json, meta)
    return paths


# helpers

def _subset_dataset(ds: Dataset, idx: np.ndarray) -> Dataset:
    idx = np.asarray(idx, dtype=int)
    return Dataset(
        X=ds.X[idx],
        y=ds.y[idx],
        feature_names=ds.feature_names,
        target_name=ds.target_name,
        index=ds.index[idx],
        classes=ds.classes,
    )


def _merge_labels(a: List[Any], b: List[Any]) -> List[Any]:
    s = {str(x): x for x in a}
    for x in b:
        s.setdefault(str(x), x)
    return [s[k] for k in sorted(s.keys())]


def _clone_classifier(clf):
    t = type(clf)
    if hasattr(clf, "__dict__"):
        if t.__name__ == "KNNClassifier":
            return t(k=getattr(clf, "k"), distance=getattr(clf, "distance"))
        if t.__name__ == "EuclideanCentroidClassifier":
            return t()
    return t()


def _run_single_split(
    cfg: Dict[str, Any],
    paths: RunPaths,
    classifier,
    train_ds: Dataset,
    test_ds: Dataset,
    dataset_labels: List[Any],
    mode_name: str,
    original_index_test: Optional[np.ndarray] = None,
) -> None:
    classifier.fit(train_ds.X, train_ds.y)
    yhat = classifier.predict(test_ds.X)

    idx_col = original_index_test if original_index_test is not None else test_ds.index

    pred_df = pd.DataFrame({"index": idx_col, "y_true": test_ds.y, "y_pred": yhat})
    pred_df.to_csv(paths.predictions_csv, index=False)

    cm_df = confusion_matrix_df(test_ds.y, yhat, labels=dataset_labels)
    cm_norm_rows = normalize_by_rows(cm_df)
    cm_norm_cols = normalize_by_columns(cm_df)

    cm_df.to_csv(paths.confusion_csv, index=True)
    cm_norm_rows.to_csv(paths.confusion_norm_rows_csv, index=True)
    cm_norm_cols.to_csv(paths.confusion_norm_cols_csv, index=True)
    
        # ConfusionMatrixDisplay
    if _confmat_display_enabled(cfg):
        labels = [str(l) for l in dataset_labels]

        save_confusion_matrix_display(
            cm_df=cm_df,
            labels=labels,
            out_path=str(paths.run_dir / "confusion_matrix.png"),
            title="Confusion Matrix",
            normalize=False,
        )

        save_confusion_matrix_display(
            cm_df=cm_norm_rows,
            labels=labels,
            out_path=str(paths.run_dir / "confusion_matrix_norm_rows.png"),
            title="Confusion Matrix (Normalized by Rows)",
            normalize=True,
        )

        save_confusion_matrix_display(
            cm_df=cm_norm_cols,
            labels=labels,
            out_path=str(paths.run_dir / "confusion_matrix_norm_cols.png"),
            title="Confusion Matrix (Normalized by Columns)",
            normalize=True,
        )

    metrics = compute_metrics(cm_df, labels=dataset_labels, y_true=test_ds.y, y_pred=yhat)
    write_json(paths.metrics_json, metrics)

    meta = {
        "mode": mode_name,
        "run_name": str(cfg.get("run_name", mode_name)),
        "timestamp": str(paths.run_dir.name.split("__", 1)[0]),
        "config": cfg,
        "train": {"n_samples": int(train_ds.X.shape[0]), "n_features": int(train_ds.X.shape[1])},
        "test": {"n_samples": int(test_ds.X.shape[0]), "n_features": int(test_ds.X.shape[1])},
        "labels": [str(l) for l in dataset_labels],
        "summary": {
            "accuracy": metrics.get("accuracy", 0.0),
            "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
            "mcc": metrics.get("mcc", 0.0),
        },
        "outputs": {
            "predictions_csv": str(paths.predictions_csv),
            "confusion_matrix_csv": str(paths.confusion_csv),
            "confusion_matrix_norm_rows_csv": str(paths.confusion_norm_rows_csv),
            "confusion_matrix_norm_cols_csv": str(paths.confusion_norm_cols_csv),
            "metrics_json": str(paths.metrics_json),
        },
    }
    write_json(paths.meta_json, meta)
