from pathlib import Path

import pytest

from allesmusterklassifizierer.errors import AMKError
from allesmusterklassifizierer.metrics.confusion_demo import (
    CONCEPT_LABELS,
    save_confusion_demo,
)


def test_default_layout_is_tp_fn_fp_tn(tmp_path: Path):
    assert CONCEPT_LABELS.tolist() == [["TP", "FN"], ["FP", "TN"]]
    output = save_confusion_demo(tmp_path / "conceptual.png")
    assert output.is_file()
    assert output.stat().st_size > 0


@pytest.mark.parametrize("suffix", [".png", ".svg", ".pdf"])
def test_manual_values_support_presentation_formats(tmp_path: Path, suffix: str):
    output = save_confusion_demo(
        tmp_path / f"manual{suffix}",
        values=[50, 10, 5, 35],
        color="Blues",
        title="Ejemplo",
        class_names=["Sí", "No"],
    )
    assert output.is_file()


def test_rejects_invalid_values_and_color(tmp_path: Path):
    with pytest.raises(AMKError, match="no negativos"):
        save_confusion_demo(tmp_path / "bad.png", values=[1, -1, 2, 3])
    with pytest.raises(AMKError, match="Paleta"):
        save_confusion_demo(tmp_path / "bad.png", color="not-a-cmap")
