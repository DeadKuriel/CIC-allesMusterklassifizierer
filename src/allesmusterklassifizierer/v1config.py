from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ConfigError

Role = Literal["feature", "target", "group", "timestamp", "identifier", "ignored"]
Semantic = Literal["numeric", "boolean", "nominal", "ordinal", "datetime", "identifier"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ColumnConfig(StrictModel):
    name: str
    role: Role = "feature"
    semantic_type: Semantic
    order: list[Any] | None = None
    true_values: list[Any] = Field(default_factory=lambda: [True, 1, "true", "yes"])
    false_values: list[Any] = Field(default_factory=lambda: [False, 0, "false", "no"])

    @model_validator(mode="after")
    def validate_order(self) -> "ColumnConfig":
        if self.semantic_type == "ordinal" and (not self.order or len(self.order) < 2):
            raise ValueError(
                f"La columna ordinal '{self.name}' requiere order con al menos dos categorías"
            )
        if self.semantic_type != "ordinal" and self.order is not None:
            raise ValueError("order sólo es válido para columnas ordinales")
        return self


class DatasetConfig(StrictModel):
    path: Path
    format: Literal["csv", "tsv", "parquet", "excel"] | None = None
    encoding: str = "utf-8"
    delimiter: str | None = None
    decimal: str = "."
    sheet_name: str | int = 0
    missing_values: list[Any] = Field(default_factory=list)
    row_group_size: int | None = Field(default=None, ge=1)
    row_group_column: str | None = None

    @model_validator(mode="after")
    def validate_row_groups(self) -> "DatasetConfig":
        if (self.row_group_size is None) != (self.row_group_column is None):
            raise ValueError(
                "row_group_size y row_group_column deben configurarse juntos"
            )
        return self


class AuditConfig(StrictModel):
    redundant_policy: Literal["report_only", "keep_first", "keep_last", "fail"] = (
        "report_only"
    )
    indiscernible_policy: Literal[
        "report_only", "fail", "remove_all", "resolve_by_mapping"
    ] = "report_only"
    resolution_mapping: Path | None = None
    target_missing: Literal["fail", "drop_rows"] = "fail"
    near_constant_threshold: float = Field(default=0.99, ge=0.5, le=1.0)
    imbalance_ratio_warning: float = Field(default=3.0, ge=1.0)


class PreprocessingConfig(StrictModel):
    numeric_imputation: Literal["mean", "median", "constant"] = "median"
    numeric_constant: float = 0.0
    categorical_imputation: Literal["most_frequent", "constant", "missing_category"] = (
        "missing_category"
    )
    categorical_constant: str = "__MISSING__"
    add_missing_indicators: bool = False
    min_category_frequency: int | float | None = None
    scale: Literal["none", "standard", "minmax", "robust", "normalize", "power"] = (
        "standard"
    )


class SplitConfig(StrictModel):
    strategy: Literal[
        "holdout",
        "stratified_holdout",
        "train_validation_test",
        "kfold",
        "stratified_kfold",
        "loo",
        "group_holdout",
        "group_kfold",
        "stratified_group_kfold",
        "temporal",
        "walk_forward",
    ] = "stratified_holdout"
    test_size: float = Field(default=0.2, gt=0, lt=1)
    validation_size: float = Field(default=0.2, gt=0, lt=1)
    n_splits: int = Field(default=5, ge=2)
    shuffle: bool = True


class BalanceConfig(StrictModel):
    strategy: Literal["none", "random_under", "random_over", "smote", "smotenc"] = (
        "none"
    )
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(StrictModel):
    name: str
    label: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class TuningConfig(StrictModel):
    method: Literal["none", "grid", "random", "halving"] = "none"
    cv: int = Field(default=5, ge=2)
    n_iter: int = Field(default=20, ge=1)
    n_jobs: int = 1
    error_score: Literal["raise"] | float = "raise"
    spaces: dict[str, dict[str, Any]] = Field(default_factory=dict)


class EvaluationConfig(StrictModel):
    primary_metric: str = "balanced_accuracy"
    average: Literal["macro", "micro", "weighted", "binary"] = "macro"
    positive_class: Any | None = None
    confusion_matrix_order: list[Any] | None = None
    zero_division: Literal["warn", 0, 1] = "warn"
    confidence_interval: bool = False


class PersistenceConfig(StrictModel):
    save_pipeline: bool = True
    report_html: bool = True
    plots: bool = True
    extra_columns: Literal["ignore", "fail"] = "ignore"


class ExperimentConfig(StrictModel):
    format_version: Literal[1] = 1
    run_name: str = Field(min_length=1)
    seed: int = 42
    output_dir: Path = Path("outputs")
    dataset: DatasetConfig
    target: str
    columns: list[ColumnConfig]
    audit: AuditConfig = Field(default_factory=AuditConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    balance: BalanceConfig = Field(default_factory=BalanceConfig)
    models: list[ModelConfig]
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)

    @model_validator(mode="after")
    def validate_consistency(self) -> "ExperimentConfig":
        names = [c.name for c in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("Los nombres declarados en columns deben ser únicos")
        targets = [c.name for c in self.columns if c.role == "target"]
        if targets != [self.target]:
            raise ValueError(
                "Debe existir exactamente una columna role=target y coincidir con target"
            )
        if not any(c.role == "feature" for c in self.columns):
            raise ValueError("Se requiere al menos una feature")
        groups = [c for c in self.columns if c.role == "group"]
        times = [c for c in self.columns if c.role == "timestamp"]
        if (
            self.split.strategy
            in {"group_holdout", "group_kfold", "stratified_group_kfold"}
            and len(groups) != 1
        ):
            raise ValueError(
                "La división por grupos requiere exactamente una columna role=group"
            )
        if self.split.strategy in {"temporal", "walk_forward"} and len(times) != 1:
            raise ValueError(
                "La división temporal requiere exactamente una columna role=timestamp"
            )
        if (
            self.evaluation.average == "binary"
            and self.evaluation.positive_class is None
        ):
            raise ValueError("average=binary requiere positive_class explícita")
        if not any(m.enabled for m in self.models):
            raise ValueError("Se requiere al menos un modelo habilitado")
        model_labels = [m.label or m.name for m in self.models if m.enabled]
        if len(model_labels) != len(set(model_labels)):
            raise ValueError(
                "Cada modelo habilitado requiere un label único cuando se repite name"
            )
        return self

    def column(self, role: Role) -> ColumnConfig | None:
        return next((c for c in self.columns if c.role == role), None)


def load_v1_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigError("La configuración YAML debe ser un mapping")
        cfg = ExperimentConfig.model_validate(raw)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"Configuración AMK 0.2 inválida: {exc}") from exc
    if not cfg.dataset.path.is_absolute():
        cfg.dataset.path = (p.parent / cfg.dataset.path).resolve()
    if cfg.audit.resolution_mapping and not cfg.audit.resolution_mapping.is_absolute():
        cfg.audit.resolution_mapping = (
            p.parent / cfg.audit.resolution_mapping
        ).resolve()
    return cfg
