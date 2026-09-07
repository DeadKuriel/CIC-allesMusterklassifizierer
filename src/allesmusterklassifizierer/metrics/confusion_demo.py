from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..errors import AMKError

CONCEPT_LABELS = np.array([["TP", "FN"], ["FP", "TN"]], dtype=object)
SUPPORTED_FORMATS = {".png", ".svg", ".pdf"}


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def save_confusion_demo(
    output: str | Path = "confusion_demo.png",
    *,
    values: Sequence[float] | None = None,
    color: str = "Reds",
    title: str = "Matriz de confusión",
    class_names: Sequence[str] = ("Positivo", "Negativo"),
    dpi: int = 300,
    size: tuple[float, float] = (7.0, 6.0),
    font_size: float = 20.0,
    text_color: str | None = None,
    background_color: str = "white",
    show_axis_labels: bool = True,
    show_colorbar: bool = False,
    transparent: bool = False,
    language: str = "es",
) -> Path:
    """Create a presentation matrix without fitting or evaluating a classifier."""
    target = Path(output)
    if target.suffix.lower() not in SUPPORTED_FORMATS:
        raise AMKError("El formato de salida debe ser PNG, SVG o PDF")
    if color not in matplotlib.colormaps:
        raise AMKError(f"Paleta de color desconocida: {color}")
    if len(class_names) != 2:
        raise AMKError("class_names debe contener exactamente dos nombres")
    if dpi <= 0 or font_size <= 0 or size[0] <= 0 or size[1] <= 0:
        raise AMKError("dpi, font_size y size deben ser mayores que cero")
    if language not in {"es", "en"}:
        raise AMKError("language debe ser 'es' o 'en'")

    if values is None:
        annotations = CONCEPT_LABELS
        heatmap = np.array([[1.0, 0.55], [0.55, 1.0]])
    else:
        if len(values) != 4:
            raise AMKError("values debe contener TP FN FP TN, en ese orden")
        numeric: np.ndarray[Any, Any] = np.asarray(values, dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise AMKError("Los valores deben ser números finitos no negativos")
        heatmap = numeric.reshape(2, 2)
        annotations = np.array(
            [_format_number(float(v)) for v in numeric], dtype=object
        ).reshape(2, 2)

    fig, ax = plt.subplots(figsize=size, facecolor=background_color)
    ax.set_facecolor(background_color)
    image = ax.imshow(heatmap, cmap=color)
    ax.set_xticks([0, 1], labels=list(class_names), fontsize=max(8, font_size * 0.65))
    ax.set_yticks([0, 1], labels=list(class_names), fontsize=max(8, font_size * 0.65))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    if show_axis_labels:
        ax.set_xlabel(
            "Predicción" if language == "es" else "Prediction",
            fontsize=max(8, font_size * 0.7),
            labelpad=12,
        )
        ax.set_ylabel(
            "Valor real" if language == "es" else "Actual value",
            fontsize=max(8, font_size * 0.7),
        )
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.set_title(title, fontsize=font_size * 1.05, pad=48 if show_axis_labels else 28)

    threshold = float(np.nanmin(heatmap) + np.nanmax(heatmap)) / 2
    for row in range(2):
        for column in range(2):
            automatic = "white" if heatmap[row, column] > threshold else "black"
            ax.text(
                column,
                row,
                annotations[row, column],
                ha="center",
                va="center",
                color=text_color or automatic,
                fontsize=font_size,
                fontweight="bold",
            )
    if show_colorbar:
        fig.colorbar(image, ax=ax)
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, transparent=transparent, facecolor=fig.get_facecolor())
    plt.close(fig)
    return target
