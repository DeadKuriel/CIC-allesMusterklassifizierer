# allesmusterklassifizierer

**allesmusterklassifizierer (amk)** es un framework experimental en Python para **clasificación de patrones**, diseñado para explorar de forma reproducible distintos **métodos de validación** y **clasificadores basados en distancia** sobre datasets en formato CSV.

El proyecto está pensado con fines **académicos y experimentales** (para el curso de Clasificación Inteligente de Patrones / CI / MCIC / CIC), priorizando:

* Claridad conceptual
* Reproducibilidad
* Separación entre validación y clasificación
* Trazabilidad completa de los resultados

---

## Características principales

* **Métodos de validación**

  * Holdout
  * Leave-One-Out (LOO)
  * K-Fold (estratificado opcional)

* **Clasificadores**

  * Euclidiano (Nearest Centroid / Minimum Distance, distancia L2 fija)
  * KNN general (k = 1 equivale a 1-NN)

* **Distancias (solo para KNN)**

  * Euclidiana (L2)
  * Manhattan / City Block (L1)
  * Chebyshev / Chessboard (L∞)

* **Métricas**

  * Matriz de confusión
  * Accuracy
  * Precision, Recall/Sensitivity, Specificity, F1 (por clase y macro)
  * Balanced Accuracy
  * MCC (multiclase)

* **Visualización**

  * Exportación opcional de matrices de confusión como imágenes (`PNG`)
    usando `ConfusionMatrixDisplay` de scikit-learn

* **Configuración reproducible**

  * Experimentos definidos mediante archivos YAML
  * Soporte para overrides por CLI (`--set`)

---

## Instalación

Requisitos:

* Python **3.10+**

Clonar el repositorio y crear un entorno virtual:

```bash
git clone https://github.com/DeadKuriel/CIC-allesMusterklassifizierer.git
cd allesmusterklassifizierer
python -m venv .venv
source .venv/bin/activate
```

Instalar el proyecto en modo editable:

```bash
python3 -m pip install -e .
```

Verificar instalación:

```bash
amk --help
```

---

## Dependencias principales

Definidas en `pyproject.toml`:

* `numpy`
* `pandas`
* `pyyaml`
* `scikit-learn` (solo para visualización y utilidades)
* `matplotlib` (para imágenes de matrices de confusión)

---

## Estructura del proyecto

```text
.
├─ src/allesmusterklassifizierer/   # Código fuente
├─ configs/                         # Configs YAML reproducibles
├─ input/                           # Datasets CSV (no versionados)
├─ outputs/                         # Resultados generados
├─ README.md
├─ pyproject.toml
└─ Makefile
```

---

## Uso del CLI

El comando base es:

```bash
amk <validate | classify | run> --config <archivo.yaml>
```

### `amk validate`

Analiza **solo el esquema de validación** (no entrena, no clasifica).

```bash
amk validate --config configs/validate_holdout.yaml
```

* Genera estadísticas de partición
* Opcionalmente exporta CSVs completos por fold
* No produce matriz de confusión

---

### `amk classify`

Ejecuta **una sola clasificación** (un split).

```bash
amk classify --config configs/classify_knn_within_dataset_holdout.yaml
```

Soporta:

* train/test explícitos
* dataset + split interno (holdout)

Produce:

* predicciones
* matriz de confusión
* métricas

---

### `amk run`

Ejecuta el **experimento completo** (validación + clasificación).

```bash
amk run --config configs/run_kfold_knn.yaml
```

* Recorre todos los folds
* Agrega predicciones
* Calcula métricas globales

---

## Configuración mediante YAML

Los experimentos se definen completamente mediante archivos YAML en `configs/`.

### Secciones comunes

#### `io`

Define los datasets de entrada.

```yaml
io:
  dataset: input/data.csv
  target_col: class
  drop_cols: []
```

O bien:

```yaml
io:
  train_dataset: input/train.csv
  test_dataset: input/test.csv
```

---

#### `validation`

Define el método de validación (para `validate` y `run`).

```yaml
validation:
  type: kfold
  params:
    n_splits: 5
    stratify: true
```

Opcionalmente, exportar particiones completas:

```yaml
validation:
  type: kfold
  export:
    csv: true
```

---

#### `classifier`

Define el clasificador.

```yaml
classifier:
  type: knn
  params:
    k: 5
    distance: euclidean
```

⚠️ El clasificador `euclidiano` **no permite** definir distancia.

---

#### `report`

Controla la visualización de matrices.

```yaml
report:
  confusion_matrix_display: true
```

---

## Configs disponibles actualmente

### Clasificación (`classify`)

* `classify_euclidean_train_test.yaml`
* `classify_knn_within_dataset_holdout.yaml`

### Experimentos completos (`run`)

* `run_holdout_knn.yaml`
* `run_loo_1nn.yaml`
* `run_loo_knn7_nutt.yaml`
* `run_kfold_knn.yaml`

### Validación (`validate`)

* `validate_holdout.yaml`
* `validate_loo.yaml`
* `validate_kfold.yaml` (exporta CSVs por fold)

---

## Salidas generadas

Cada ejecución crea una carpeta:

```text
outputs/runs/<timestamp>__<run_name>/
```

Con archivos como:

```text
predictions.csv / fold_predictions.csv
confusion_matrix.csv
confusion_matrix_norm_rows.csv
confusion_matrix_norm_cols.csv
confusion_matrix.png
metrics.json
meta.json
```
