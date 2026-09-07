# Referencia de configuración YAML de AMK 0.2

Esta guía describe el esquema que valida actualmente AMK 0.2. Las claves no declaradas producen un
error: la configuración es estricta. Las rutas relativas de `dataset.path` y
`audit.resolution_mapping` se resuelven respecto del archivo YAML.

## Plantilla completa

```yaml
format_version: 1
run_name: nombre_de_ejecucion
seed: 42
output_dir: outputs

dataset:
  path: ../input/datos.csv
  format: csv
  encoding: utf-8
  delimiter: ","
  decimal: "."
  sheet_name: 0
  missing_values: ["", "NA", "?"]

target: clase

columns:
  - name: edad
    role: feature
    semantic_type: numeric
  - name: color
    role: feature
    semantic_type: nominal
  - name: nivel
    role: feature
    semantic_type: ordinal
    order: [bajo, medio, alto]
  - name: activo
    role: feature
    semantic_type: boolean
    true_values: [true, 1, "si"]
    false_values: [false, 0, "no"]
  - name: fecha
    role: timestamp
    semantic_type: datetime
  - name: sujeto
    role: group
    semantic_type: identifier
  - name: clase
    role: target
    semantic_type: nominal

audit:
  redundant_policy: report_only
  indiscernible_policy: report_only
  resolution_mapping: null
  target_missing: fail
  near_constant_threshold: 0.99
  imbalance_ratio_warning: 3.0

preprocessing:
  numeric_imputation: median
  numeric_constant: 0.0
  categorical_imputation: missing_category
  categorical_constant: __MISSING__
  add_missing_indicators: false
  min_category_frequency: null
  scale: standard

split:
  strategy: stratified_holdout
  test_size: 0.2
  validation_size: 0.2
  n_splits: 5
  shuffle: true

balance:
  strategy: none
  params: {}

models:
  - name: logistic_regression
    enabled: true
    params: {}
  - name: random_forest
    enabled: true
    params:
      n_estimators: 100

tuning:
  method: none
  cv: 5
  n_iter: 20
  n_jobs: 1
  error_score: raise
  spaces: {}

evaluation:
  primary_metric: balanced_accuracy
  average: macro
  positive_class: null
  zero_division: warn
  confidence_interval: false

persistence:
  save_pipeline: true
  report_html: true
  plots: false
  extra_columns: ignore
```

No es necesario escribir todas las claves: muchas tienen valores predeterminados. La configuración
mínima debe incluir `run_name`, `dataset`, `target`, `columns` y al menos un modelo habilitado.

## Configuración general

| Clave | Tipo | Predeterminado | Descripción |
|---|---:|---:|---|
| `format_version` | entero literal | `1` | Versión del formato YAML; no es la versión del paquete. |
| `run_name` | texto no vacío | requerido | Nombre usado en las carpetas de resultados. |
| `seed` | entero | `42` | Semilla global para particiones, modelos y samplers compatibles. |
| `output_dir` | ruta | `outputs` | Directorio base de artefactos. |
| `target` | texto | requerido | Nombre exacto de la variable objetivo. |

## `dataset`

| Clave | Valores | Predeterminado | Notas |
|---|---|---|---|
| `path` | ruta | requerido | Debe existir cuando se carga el experimento. |
| `format` | `csv`, `tsv`, `parquet`, `excel`, `null` | `null` | Con `null` se infiere por extensión. |
| `encoding` | nombre de codificación | `utf-8` | Aplica a CSV y TSV. |
| `delimiter` | texto o `null` | `null` | Por defecto coma para CSV y tabulador para TSV. |
| `decimal` | texto | `.` | Separador decimal para CSV y TSV. |
| `sheet_name` | texto o entero | `0` | Hoja de Excel por nombre o posición. |
| `missing_values` | lista | `[]` | Valores adicionales tratados como faltantes. |
| `row_group_size` | entero o `null` | `null` | Tamaño fijo de bloques consecutivos usados como grupos. |
| `row_group_column` | texto o `null` | `null` | Nombre de la columna de grupo derivada; se configura junto con `row_group_size`. |

Parquet requiere `pyarrow`; Excel requiere `openpyxl`. Se instalan con
`pip install -e '.[formats]'`.

## `columns`

Cada columna declarada admite:

| Clave | Valores | Predeterminado |
|---|---|---|
| `name` | texto | requerido |
| `role` | `feature`, `target`, `group`, `timestamp`, `identifier`, `ignored` | `feature` |
| `semantic_type` | `numeric`, `boolean`, `nominal`, `ordinal`, `datetime`, `identifier` | requerido |
| `order` | lista o `null` | `null` |
| `true_values` | lista | `[true, 1, "true", "yes"]` |
| `false_values` | lista | `[false, 0, "false", "no"]` |

Reglas:

- Debe existir exactamente una columna con `role: target`, y su nombre debe coincidir con `target`.
- Debe existir al menos una `feature`.
- Los nombres declarados no pueden repetirse.
- `ordinal` exige `order` con al menos dos categorías; ninguna otra clase semántica admite `order`.
- Las nominales se codifican con one-hot y aceptan categorías desconocidas al predecir.
- Las ordinales usan el orden declarado; categorías desconocidas se representan con `-1`.
- Los booleanos rechazan valores que no aparezcan en `true_values` o `false_values`.
- Las fechas generan año, mes, día, día de semana, trimestre y seno/coseno del mes.
- `group` y `timestamp` sirven para la partición y no entran como features salvo que se declare otra
  columna independiente para ese propósito.
- `identifier` e `ignored` no se usan para entrenar.
- Una columna con `semantic_type: identifier` tampoco posee un transformador de features en 0.2;
  debe utilizar normalmente `role: identifier` o `role: group`.

## `audit`

### Patrones redundantes: mismo X y mismo objetivo

`redundant_policy` acepta:

- `report_only`: informa sin eliminar.
- `keep_first`: conserva el primer registro del patrón.
- `keep_last`: conserva el último.
- `fail`: detiene la ejecución.

### Patrones indiscernibles: mismo X y objetivos distintos

`indiscernible_policy` acepta:

- `report_only`: informa sin modificar.
- `fail`: detiene la ejecución.
- `remove_all`: elimina todos los registros del conflicto.
- `resolve_by_mapping`: utiliza `resolution_mapping`.

Para `resolve_by_mapping`, el CSV debe contener `pattern_key` y `resolved_target`.

Otras opciones:

| Clave | Valores | Predeterminado |
|---|---|---|
| `target_missing` | `fail`, `drop_rows` | `fail` |
| `near_constant_threshold` | número entre `0.5` y `1.0` | `0.99` |
| `imbalance_ratio_warning` | número mayor o igual que `1.0` | `3.0` |

## `preprocessing`

| Clave | Valores | Predeterminado |
|---|---|---|
| `numeric_imputation` | `mean`, `median`, `constant` | `median` |
| `numeric_constant` | número | `0.0` |
| `categorical_imputation` | `most_frequent`, `constant`, `missing_category` | `missing_category` |
| `categorical_constant` | texto | `__MISSING__` |
| `add_missing_indicators` | booleano | `false` |
| `min_category_frequency` | entero, proporción o `null` | `null` |
| `scale` | `none`, `standard`, `minmax`, `robust`, `normalize`, `power` | `standard` |

`min_category_frequency` agrupa categorías infrecuentes mediante `OneHotEncoder`: un entero es el
número mínimo de observaciones y un decimal entre 0 y 1 representa una proporción.

## `split`

`strategy` acepta:

- `holdout`: train/test sin estratificación.
- `stratified_holdout`: train/test preservando clases.
- `train_validation_test`: separa test y deriva validation desde train.
- `kfold`: K-fold convencional.
- `stratified_kfold`: K-fold estratificado.
- `loo`: Leave-One-Out.
- `group_holdout`: holdout por grupos.
- `group_kfold`: K-fold por grupos.
- `stratified_group_kfold`: estratificado sin compartir grupos.
- `temporal`: pasado para train y futuro para test.
- `walk_forward`: ventanas temporales de entrenamiento creciente.

Parámetros:

| Clave | Restricción | Predeterminado | Se usa principalmente en |
|---|---|---|---|
| `test_size` | mayor que 0 y menor que 1 | `0.2` | holdout y temporal |
| `validation_size` | mayor que 0 y menor que 1 | `0.2` | train/validation/test |
| `n_splits` | entero mayor o igual que 2 | `5` | K-fold y walk-forward |
| `shuffle` | booleano | `true` | K-fold compatibles |

Las estrategias de grupo requieren exactamente una columna `role: group`; las temporales requieren
exactamente una columna `role: timestamp`.

## `balance`

`strategy` acepta:

- `none`.
- `random_under`: submuestreo aleatorio.
- `random_over`: sobremuestreo aleatorio.
- `smote`.
- `smotenc`.

`params` se pasa al sampler de imbalanced-learn, agregando `random_state: seed`. Ejemplo:

```yaml
balance:
  strategy: smote
  params:
    k_neighbors: 3
```

`smotenc` todavía requiere declarar manualmente `categorical_features` en el espacio ya transformado;
su configuración automática para datos mixtos está pendiente.

## `models`

Cada elemento admite `name`, `label`, `enabled` —predeterminado `true`— y `params`, que se entrega al
constructor de sklearn.

`label` permite distinguir variantes del mismo algoritmo, por ejemplo `1nn_euclidean` y
`3nn_euclidean`; debe ser único entre los modelos habilitados.

| Nombre | Implementación | Valores predeterminados adicionales |
|---|---|---|
| `logistic_regression` | `LogisticRegression` | `max_iter: 1000` |
| `knn` | `KNeighborsClassifier` | sklearn |
| `centroid` | `NearestCentroid` | sklearn |
| `decision_tree` | `DecisionTreeClassifier` | sklearn |
| `random_forest` | `RandomForestClassifier` | `n_estimators: 200` |
| `extra_trees` | `ExtraTreesClassifier` | `n_estimators: 200` |
| `linear_svm` | `LinearSVC` | sklearn |
| `kernel_svm` | `SVC` | `probability: true` |
| `gaussian_nb` | `GaussianNB` | sklearn |
| `gradient_boosting` | `GradientBoostingClassifier` | sklearn |
| `hist_gradient_boosting` | `HistGradientBoostingClassifier` | sklearn |
| `adaboost` | `AdaBoostClassifier` | sklearn |
| `mlp` | `MLPClassifier` | `max_iter: 500` |

Alias: `euclidiano` y `euclidean_centroid` equivalen a `centroid`; `svm` equivale a `kernel_svm`.

Para KNN, `distance` admite `euclidean`, `manhattan`/`cityblock`,
`chebyshev`/`chessboard` y Minkowski:

```yaml
- name: knn
  params:
    n_neighbors: 5
    distance:
      type: minkowski
      params: {p: 3}
```

## `tuning`

| Clave | Valores | Predeterminado |
|---|---|---|
| `method` | `none`, `grid`, `random`, `halving` | `none` |
| `cv` | entero mayor o igual que 2 | `5` |
| `n_iter` | entero mayor o igual que 1 | `20` |
| `n_jobs` | entero | `1` |
| `error_score` | `raise` o número | `raise` |
| `spaces` | mapping por nombre de modelo | `{}` |

Ejemplo:

```yaml
tuning:
  method: grid
  cv: 5
  n_jobs: -1
  spaces:
    logistic_regression:
      C: [0.1, 1.0, 10.0]
    random_forest:
      n_estimators: [100, 300]
      max_depth: [null, 5, 10]
```

No se escribe el prefijo `model__`; AMK lo agrega. `random` también usa `n_iter`. Aunque el esquema
acepta `halving`, en 0.2 todavía no implementa successive halving real y no debe usarse como función
terminada.

## `evaluation`

| Clave | Valores | Predeterminado |
|---|---|---|
| `primary_metric` | nombre de scoring compatible | `balanced_accuracy` |
| `average` | `macro`, `micro`, `weighted`, `binary` | `macro` |
| `positive_class` | cualquier etiqueta o `null` | `null` |
| `zero_division` | `warn`, `0`, `1` | `warn` |
| `confidence_interval` | booleano | `false` |

`average: binary` exige `positive_class`. La evaluación final produce accuracy, balanced accuracy,
precision, recall, F1, MCC, kappa, especificidad y soporte por clase. Con probabilidades/scores también
puede producir log loss, ROC-AUC y PR-AUC binaria.

`confidence_interval` se valida pero su integración en el orquestador todavía está pendiente.

## `persistence`

| Clave | Valores | Predeterminado | Efecto |
|---|---|---|---|
| `save_pipeline` | booleano | `true` | Guarda `pipeline.joblib`. |
| `report_html` | booleano | `true` | Genera `report.html`. |
| `plots` | booleano | `true` | Guarda en PNG las matrices de confusión de cada modelo. |
| `extra_columns` | `ignore`, `fail` | `ignore` | Política para columnas adicionales al predecir. |

Al predecir, siempre se falla si falta una feature requerida y las columnas se reordenan por nombre.

## Validación y diagnóstico

```bash
amk config validate --config configs/v0_2_numeric.yaml
```

Un error de configuración devuelve código de salida 2. La configuración validada puede inspeccionarse
en `resolved_config.yaml` después de una ejecución completa.
