from __future__ import annotations

from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd


def save_confusion_matrix_display(
    cm_df: pd.DataFrame,
    labels: List[str],
    out_path: str,
    title: str,
    normalize: bool = False,
    cmap: str = "Reds",
) -> None:
    """
    Saves a PNG image of the confusion matrix using ConfusionMatrixDisplay.
    - cm_df: Confusion matrix DataFrame (rows = true labels, columns = predicted labels)
    - labels: list of class labels in order
    - normalize: only affects the title text (values are already normalized if applicable)
    """
    cm = cm_df.to_numpy()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(
        include_values=True,
        cmap=cmap,
        ax=ax,
        xticks_rotation=45,
        values_format=".2f" if normalize else "d",
        colorbar=True,
    )

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
