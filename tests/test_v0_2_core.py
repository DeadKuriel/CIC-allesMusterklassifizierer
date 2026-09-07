import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef

from allesmusterklassifizierer.distances.metrics import pairwise_distance
from allesmusterklassifizierer.evaluation import (
    classification_metrics,
    confusion_tables,
)
from allesmusterklassifizierer.errors import ValidationError
from allesmusterklassifizierer.experiment import (
    _confusion_matrix_labels,
    _write_model_confusion_matrices,
)
from allesmusterklassifizierer.splitting import make_splits
from allesmusterklassifizierer.v1config import ExperimentConfig, SplitConfig


def test_distance_aliases_and_minkowski():
    X = np.array([[0.0, 0.0], [2.0, 3.0]])
    x = np.array([1.0, 1.0])
    np.testing.assert_allclose(
        pairwise_distance(X, x, "manhattan"), pairwise_distance(X, x, "cityblock")
    )
    np.testing.assert_allclose(
        pairwise_distance(X, x, "chebyshev"), pairwise_distance(X, x, "chessboard")
    )
    np.testing.assert_allclose(
        pairwise_distance(X, x, {"type": "minkowski", "params": {"p": 2}}),
        pairwise_distance(X, x, "euclidean"),
    )


def test_confusion_rejects_unknown():
    with pytest.raises(ValidationError):
        confusion_tables([0, 2], [0, 1], [0, 1])


def test_confusion_can_put_positive_class_first():
    labels = _confusion_matrix_labels(["p", "e"], ["e", "p"])
    raw, _, _ = confusion_tables(["p", "p", "e", "e"], ["p", "e", "p", "e"], labels)

    assert list(raw.index) == ["true_p", "true_e"]
    assert list(raw.columns) == ["pred_p", "pred_e"]
    assert raw.to_numpy().tolist() == [[1, 1], [1, 1]]


def test_writes_confusion_matrices_and_plots_for_each_model(tmp_path):
    artifacts = _write_model_confusion_matrices(
        tmp_path,
        "3nn euclidean",
        ["e", "e", "p", "p"],
        ["e", "p", "p", "p"],
        ["e", "p"],
        plots=True,
    )

    assert set(artifacts) == {
        "raw_csv",
        "raw_png",
        "normalized_rows_csv",
        "normalized_rows_png",
        "normalized_columns_csv",
        "normalized_columns_png",
    }
    assert all(
        (tmp_path / relative_path).is_file() for relative_path in artifacts.values()
    )


def test_metrics_match_sklearn():
    yt = np.array([0, 0, 1, 1, 1])
    yp = np.array([0, 1, 1, 1, 0])
    m = classification_metrics(yt, yp, labels=[0, 1])
    assert m["accuracy"] == accuracy_score(yt, yp)
    assert m["balanced_accuracy"] == balanced_accuracy_score(yt, yp)
    assert m["mcc"] == matthews_corrcoef(yt, yp)


def test_reproducible_disjoint_stratified_holdout():
    y = np.array([0] * 10 + [1] * 10)
    cfg = SplitConfig(strategy="stratified_holdout", test_size=0.2)
    a = make_splits(cfg, y, seed=7)[0]
    b = make_splits(cfg, y, seed=7)[0]
    np.testing.assert_array_equal(a.test_idx, b.test_idx)
    assert not set(a.train_idx) & set(a.test_idx)


def test_group_and_temporal_invariants():
    y = np.array([0, 1] * 6)
    groups = np.repeat(np.arange(6), 2)
    sp = make_splits(
        SplitConfig(strategy="group_holdout", test_size=0.3), y, groups=groups
    )[0]
    assert not set(groups[sp.train_idx]) & set(groups[sp.test_idx])
    ts = pd.date_range("2020", periods=12)
    tp = make_splits(
        SplitConfig(strategy="temporal", test_size=0.25), y, timestamps=ts
    )[0]
    assert ts[tp.train_idx].max() < ts[tp.test_idx].min()


def test_ordinal_order_is_required(tmp_path):
    raw = {
        "format_version": 1,
        "run_name": "x",
        "dataset": {"path": tmp_path / "x.csv"},
        "target": "y",
        "columns": [
            {"name": "x", "role": "feature", "semantic_type": "ordinal"},
            {"name": "y", "role": "target", "semantic_type": "nominal"},
        ],
        "models": [{"name": "knn"}],
    }
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(raw)


def test_repeated_model_names_require_unique_labels(tmp_path):
    raw = {
        "format_version": 1,
        "run_name": "x",
        "dataset": {"path": tmp_path / "x.csv"},
        "target": "y",
        "columns": [
            {"name": "x", "role": "feature", "semantic_type": "numeric"},
            {"name": "y", "role": "target", "semantic_type": "nominal"},
        ],
        "models": [{"name": "knn", "label": "1nn"}, {"name": "knn", "label": "3nn"}],
    }
    cfg = ExperimentConfig.model_validate(raw)
    assert [m.label for m in cfg.models] == ["1nn", "3nn"]
