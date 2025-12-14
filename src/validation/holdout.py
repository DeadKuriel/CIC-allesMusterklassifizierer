from __future__ import annotations
import pandas as pd
import random
import os
from datetime import datetime
from collections import Counter
import numpy as np


def holdout_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.3,
    random_state: int | None = None,
    output_dir: str | None = None,
    dataset_name: str = "dataset",
    save: bool = True,
):
    """
    Realiza Hold-Out estratificado manual sobre un DataFrame.
    Para cada clase c calcula:
        n_train_c = round((1 - test_size) * n_c)
    y garantiza que haya al menos 1 patrón de cada clase en TRAIN y en TEST.
    Devuelve train_df, test_df, la semilla usada y el timestamp de la corrida.
    """

    if target_col not in df.columns:
        raise ValueError(
            f"La columna objetivo '{target_col}' no existe en el DataFrame.\n"
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    # Semilla
    if random_state is None:
        random_state = random.randint(0, 999999)
        print(f"Semilla aleatoria generada: {random_state}")
    else:
        print(f"Semilla fija utilizada: {random_state}")

    rng = np.random.default_rng(random_state)

    y = df[target_col]
    print("\nDistribución original de clases:", Counter(y))

    train_indices: list[int] = []
    test_indices: list[int] = []

    train_frac = 1.0 - float(test_size)

    # Estratificación manual
    for cls in sorted(y.unique()):
        cls_mask = (y == cls)
        cls_idx = np.flatnonzero(cls_mask)  # índices de esa clase
        n_c = len(cls_idx)

        # cálculo con redondeo por clase
        n_train_c = int(round(train_frac * n_c))

        # aseguramos que haya al menos 1 patrón en TRAIN y 1 en TEST
        if n_train_c <= 0:
            n_train_c = 1
        if n_train_c >= n_c:
            n_train_c = n_c - 1

        rng.shuffle(cls_idx)
        cls_train_idx = cls_idx[:n_train_c]
        cls_test_idx = cls_idx[n_train_c:]

        train_indices.extend(cls_train_idx.tolist())
        test_indices.extend(cls_test_idx.tolist())

    # Creamos los DataFrames finales
    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    print("\nDistribución TRAIN:", Counter(train_df[target_col]))
    print("Distribución TEST:", Counter(test_df[target_col]))

    # Timestamp único
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if save and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        # porcentaje de TEST (por si algún día usas test_size como entero)
        if isinstance(test_size, float):
            test_pct = int(round(test_size * 100))
        else:
            test_pct = int(round(len(test_df) * 100 / len(df)))

        base_name = dataset_name

        train_path = (
            f"{output_dir}/"
            f"{timestamp}_{base_name}_seed{random_state}_test{test_pct}_entrenamiento.csv"
        )
        test_path = (
            f"{output_dir}/"
            f"{timestamp}_{base_name}_seed{random_state}_test{test_pct}_prueba.csv"
        )

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print("\nArchivos generados:")
        print(f" {train_path}")
        print(f" {test_path}")

    return train_df, test_df, random_state, timestamp
