from __future__ import annotations
import os
import pandas as pd

def load_csv(dataset_name: str, input_dir: str = "input"):
    """
    Carga un CSV desde input/ usando dataset_name (sin extensión).
    Ej: dataset_name="haberman_58" -> input/haberman_58.csv
    """
    filename = f"{dataset_name}.csv"
    path = os.path.join(input_dir, filename)
    print(f"Leyendo dataset desde: {path}")
    df = pd.read_csv(path)
    return df, dataset_name
