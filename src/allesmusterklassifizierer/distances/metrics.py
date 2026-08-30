from __future__ import annotations

from typing import Literal

import numpy as np

DistanceName = Literal["euclidean", "manhattan", "cityblock", "chebyshev", "chessboard", "minkowski"]

def _parse_metric(metric):
    if isinstance(metric, str):
        return metric, {}

    if isinstance(metric, dict):
        mtype = metric.get("type")
        params = metric.get("params", {}) or {}
        return mtype, params

    raise ValueError(f"Distancia no soportada: {metric}")

def pairwise_distance(Xtr, x, metric):
    mtype, params = _parse_metric(metric)

    if mtype == "euclidean":
        return np.sqrt(np.sum((Xtr - x) ** 2, axis=1))
    if mtype in ("manhattan", "cityblock"):
        return np.sum(np.abs(Xtr - x), axis=1)
    if mtype in ("chebyshev", "chessboard"):
        return np.max(np.abs(Xtr - x), axis=1)
    if mtype == "minkowski":
        p = params.get("p", None)
        if p is None or not isinstance(p, (int, float)) or float(p) <= 0:
            raise ValueError("Para Minkowski se requiere el parámetro p.")
        p = float(p)
        return np.sum(np.abs(Xtr - x) ** p, axis=1) ** (1.0 / p)

    raise ValueError(f"Distancia no soportada: {metric}")
