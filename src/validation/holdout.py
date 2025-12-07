from __future__ import annotations
import pandas as pd
import random
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from collections import Counter

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
    Realiza Hold-Out estratificado sobre un DataFrame.
    Devuelve train_df, test_df, la semilla usada y el timestamp de la corrida.
    Si save=True y output_dir no es None, guarda los CSVs de train/test.
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

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print("\nDistribución original de clases:", Counter(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print("\nDistribución TRAIN:", Counter(y_train))
    print("Distribución TEST:", Counter(y_test))

    train_df = X_train.copy()
    train_df[target_col] = y_train

    test_df = X_test.copy()
    test_df[target_col] = y_test

    # Timestamp único de esta corrida (se usa tanto en CSV como en resultados)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if save and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

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
