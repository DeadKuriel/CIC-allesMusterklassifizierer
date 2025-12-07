import os
import sys

# Añadir la raíz del proyecto al sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipelines.run_experiment import run_holdout_euclidean

if __name__ == "__main__":
    run_holdout_euclidean(           # Ejecutar experimento Holdout Euclidean
        dataset_name="nutt",         # Cargar dataset
        target_col="class",          # Columna objetivo
        test_size=0.8,               # Porcentaje del conjunto de Prueba
        random_state=None,           # Semilla aleatoria
                                     # None o un numero para la semilla aleatoria
    )
