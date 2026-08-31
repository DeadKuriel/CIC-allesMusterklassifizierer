from __future__ import annotations

from typing import Any, Dict, List, Optional

from .errors import ConfigError
from .utils import parse_scalar, read_yaml, set_by_dotted_path


def load_config(path: str, overrides: List[str] | None = None) -> Dict[str, Any]:
    cfg = read_yaml(path)

    overrides = overrides or []
    for ov in overrides:
        if "=" not in ov:
            raise ConfigError(f"Override inválido '{ov}'. Usa --set key.path=value")
        k, v = ov.split("=", 1)
        set_by_dotted_path(cfg, k.strip(), parse_scalar(v))

    return cfg


def require_dict(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = cfg.get(key)
    if not isinstance(v, dict):
        raise ConfigError(f"Falta o inválido '{key}' (se espera un mapping/dict).")
    return v


def optional_dict(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = cfg.get(key, {})
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ConfigError(f"'{key}' debe ser dict si existe.")
    return v


def get_str(cfg: Dict[str, Any], key: str, default: Optional[str] = None) -> str:
    v = cfg.get(key, default)
    if not isinstance(v, str) or not v.strip():
        raise ConfigError(f"'{key}' debe ser string no vacío.")
    return v


def get_bool(d: Dict[str, Any], key: str, default: bool) -> bool:
    v = d.get(key, default)
    if not isinstance(v, bool):
        raise ConfigError(f"'{key}' debe ser boolean.")
    return v


def get_float(d: Dict[str, Any], key: str, default: float) -> float:
    v = d.get(key, default)
    if not isinstance(v, (int, float)):
        raise ConfigError(f"'{key}' debe ser numérico.")
    return float(v)


def get_int(d: Dict[str, Any], key: str, default: int) -> int:
    v = d.get(key, default)
    if not isinstance(v, int):
        raise ConfigError(f"'{key}' debe ser entero.")
    return v


def get_list_of_str(
    d: Dict[str, Any], key: str, default: List[str] | None = None
) -> List[str]:
    v = d.get(key, default if default is not None else [])
    if v is None:
        return []
    if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
        raise ConfigError(f"'{key}' debe ser lista de strings.")
    return v
