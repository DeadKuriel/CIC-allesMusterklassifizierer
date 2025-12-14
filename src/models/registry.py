from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

# Import de modelos existentes
from .euclidean import EuclideanClassifier


@dataclass(frozen=True)
class ModelSpec:
    """
    Especificación de un modelo registrable.
    builder: función que construye la instancia del modelo.
    """
    name: str
    builder: Callable[..., Any]


# Registro central de modelos
_MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "euclidean": ModelSpec(
        name="euclidean",
        builder=lambda **kwargs: EuclideanClassifier()
    ),
    # Aquí luego se agregara "1nn", "svm", etc.
}


def make_model(model_name: str, **kwargs) -> Any:
    """
    Crea un modelo a partir de su nombre.
    kwargs se pasa al builder del modelo.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name debe ser un string no vacío.")

    key = model_name.strip().lower()
    spec = _MODEL_REGISTRY.get(key)

    if spec is None:
        raise ValueError(
            f"Modelo no soportado: '{model_name}'. "
            f"Disponibles: {', '.join(list_models())}"
        )

    return spec.builder(**kwargs)


def list_models() -> List[str]:
    """Lista de modelos disponibles (ordenada)."""
    return sorted(_MODEL_REGISTRY.keys())


def register_model(name: str, builder: Callable[..., Any]) -> None:
    """
    Permite registrar modelos en runtime (opcional).
    Útil si luego quieres plugins.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name debe ser un string no vacío.")
    key = name.strip().lower()

    if key in _MODEL_REGISTRY:
        raise ValueError(f"El modelo '{key}' ya está registrado.")

    _MODEL_REGISTRY[key] = ModelSpec(name=key, builder=builder)
