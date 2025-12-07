from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ..data.loader import load_csv
from ..validation.holdout import holdout_split
from ..models.euclidean import EuclideanClassifier
from ..metrics.classification import accuracy, imbalance_ratio


def run_holdout_euclidean(
    dataset_name: str,
    target_col: str,
    test_size: float = 0.3,
    random_state: int | None = None,
    input_dir: str = "input",
    output_dir_splits: str = "output/splits",
):
    # 1. Cargar datos completos (solo para leer el CSV)
    df, dataset_resolved = load_csv(dataset_name, input_dir=input_dir)

    # 2. Hacer hold-out estratificado
    train_df, test_df, used_seed, timestamp = holdout_split(
        df=df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
        output_dir=output_dir_splits,
        dataset_name=dataset_resolved,
        save=True,
    )

    # 3. Separar X / y para TRAIN y TEST
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # 4. Calcular IR SOLO en el conjunto de entrenamiento
    ir_train = imbalance_ratio(y_train)
    print(f"\nIR (Imbalance Ratio) en TRAIN: {ir_train:.4f}")

    # Regla:
    # IR <= 1.5  -> balanceado  -> razonable reportar accuracy
    # IR >  1.5  -> desbalanceo -> no reportar accuracy
    IR_THRESHOLD = 1.5

    if ir_train <= IR_THRESHOLD:
        print(
            "Conjunto de ENTRENAMIENTO BALANCEADO (IR ≤ 1.5). "
            "Es razonable usar accuracy."
        )
    else:
        print(
            "Conjunto de ENTRENAMIENTO DESBALANCEADO (IR > 1.5). "
            "No es razonable reportar solo accuracy."
        )

    # 5. Entrenar clasificador euclidiano
    clf = EuclideanClassifier()
    clf.fit(X_train, y_train)

    # 6. Predecir SIEMPRE (aunque luego no usemos accuracy si hay desbalanceo)
    y_pred = clf.predict(X_test)

    # 7. Decidir si calculamos accuracy según IR de TRAIN
    acc: float | None = None
    if ir_train <= IR_THRESHOLD:
        acc = accuracy(y_test, y_pred)
        print(f"\nAccuracy, clasificador Euclidiano (hold-out): {acc:.4f}")
    else:
        print(
            "\nNo se reporta accuracy porque el conjunto de entrenamiento "
            f"está desbalanceado (IR_train = {ir_train:.4f} > {IR_THRESHOLD})."
        )

    print(f"Semilla usada: {used_seed}")

    # 8. Imprimir tabla de centroides en consola
    print("\nCentroides por clase (promedios en el espacio de características):")
    for cls, centroid in clf.class_means_.items():
        print(f"Clase {cls}: {centroid}")

    # 9. Calcular distancias de cada patrón de TEST a cada centroide
    dist_info = clf.distances_to_centroids(X_test)
    dist_matrix = dist_info["distances"]   # (N_test, n_clases)
    centroid_classes = dist_info["classes"]

    print("\nDistancias de cada patrón de TEST a cada centroide:")
    header = (
        "Patrón | Clase Real | "
        + " | ".join([f"Dist a clase {c}" for c in centroid_classes])
        + " | Clase Asignada | Resultado"
    )
    print(header)

    y_test_array = y_test.to_numpy()
    correct_mask = (y_pred == y_test_array)
    num_correct = int(correct_mask.sum())
    num_errors = int(len(y_test_array) - num_correct)

    for idx in range(len(X_test)):
        real_class = y_test_array[idx]
        assigned = y_pred[idx]
        dist_list = [f"{dist_matrix[idx, j]:.4f}" for j in range(len(centroid_classes))]
        dist_str = " | ".join(dist_list)
        resultado = "ACIERTO" if assigned == real_class else "ERROR"
        print(f"{idx:6d} | {real_class:10} | {dist_str} | {assigned:14} | {resultado}")

    # 9.1 Matrices de confusión (cruda, normalizada por filas y por columnas)
    #     Usamos las clases presentes al menos en TRAIN
    classes = sorted(set(y_train.unique()) | set(y_test.unique()))
    cm = confusion_matrix(y_test_array, y_pred, labels=classes)

    # Normalización por FILAS (cada fila suma 1)
    cm_rows = cm.astype(float)
    row_sums = cm_rows.sum(axis=1, keepdims=True)  # suma por fila (real)
    cm_rows = np.divide(cm_rows, row_sums, where=row_sums != 0)

    # Normalización por COLUMNAS (cada columna suma 1)
    cm_cols = cm.astype(float)
    col_sums = cm_cols.sum(axis=0, keepdims=True)  # suma por columna (pred)
    cm_cols = np.divide(cm_cols, col_sums, where=col_sums != 0)

    print("\nMatriz de confusión (cruda):")
    print(pd.DataFrame(
        cm,
        index=[f"real_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes],
    ))

    print("\nMatriz de confusión normalizada por FILAS (cada fila suma 1):")
    print(pd.DataFrame(
        cm_rows,
        index=[f"real_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes],
    ))

    print("\nMatriz de confusión normalizada por COLUMNAS (cada columna suma 1):")
    print(pd.DataFrame(
        cm_cols,
        index=[f"real_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes],
    ))

    # 10. Guardar resultados en archivos (texto + CSVs)
    resultados_dir = Path("output/euclidean")
    resultados_dir.mkdir(parents=True, exist_ok=True)

    # Calcular porcentaje de test de manera similar al holdout
    if isinstance(test_size, float):
        test_pct = int(round(test_size * 100))
    else:
        total = len(df)
        test_pct = int(round(len(test_df) * 100 / total))

    base_name = dataset_resolved

    # Archivos:
    # - resumen (txt)
    # - centroides (csv)
    # - distancias (csv)
    # - matrices de confusión (csv)
    results_txt_path = (
        resultados_dir
        / f"{timestamp}_{base_name}_seed{used_seed}_test{test_pct}_euclidean_resumen.txt"
    )
    centroids_csv_path = (
        resultados_dir
        / f"{timestamp}_{base_name}_seed{used_seed}_test{test_pct}_euclidean_centroides.csv"
    )
    distances_csv_path = (
        resultados_dir
        / f"{timestamp}_{base_name}_seed{used_seed}_test{test_pct}_euclidean_distancias.csv"
    )
    cm_raw_csv_path = (
        resultados_dir
        / f"{timestamp}_{base_name}_seed{used_seed}_test{test_pct}_euclidean_cm_cruda.csv"
    )
    cm_rows_csv_path = (
        resultados_dir
        / f"{timestamp}_{base_name}_seed{used_seed}_test{test_pct}_euclidean_cm_filas.csv"
    )
    cm_cols_csv_path = (
        resultados_dir
        / f"{timestamp}_{base_name}_seed{used_seed}_test{test_pct}_euclidean_cm_columnas.csv"
    )

    # 10.1 Guardar resumen en TXT
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_resolved}\n")
        f.write(f"Target: {target_col}\n")
        f.write(f"Validación: hold-out (test_size={test_size})\n")
        f.write(f"Semilla: {used_seed}\n")
        f.write(f"IR en TRAIN (Imbalance Ratio): {ir_train:.4f}\n")
        f.write("Criterio: IR ≤ 1.5 -> balanceado, IR > 1.5 -> desbalanceado\n")
        f.write(f"Aciertos: {num_correct}\n")
        f.write(f"Errores: {num_errors}\n")
        f.write(f"Total patrones TEST: {len(y_test_array)}\n")

        if acc is not None:
            f.write(f"Accuracy Euclidiano: {acc:.4f}\n")
        else:
            f.write(
                "Accuracy NO reportado: el conjunto de entrenamiento es "
                "desbalanceado (IR_train > 1.5), por lo que no es razonable "
                "usar solo accuracy.\n"
            )

    # 10.2 Guardar centroides en CSV
    centroid_rows = []
    for cls, centroid in clf.class_means_.items():
        row = {"class": cls}
        for feat_name, value in zip(X_train.columns, centroid):
            row[str(feat_name)] = value
        centroid_rows.append(row)

    centroids_df = pd.DataFrame(centroid_rows)
    centroids_df.to_csv(centroids_csv_path, index=False)

    # 10.3 Guardar distancias por patrón de TEST en CSV
    dist_rows = []
    for idx in range(len(X_test)):
        true_cls = y_test_array[idx]
        assigned_cls = y_pred[idx]
        row = {
            "index": idx,
            "true_class": true_cls,
        }
        for j, c in enumerate(centroid_classes):
            row[f"dist_to_class_{c}"] = dist_matrix[idx, j]
        row["assigned_class"] = assigned_cls
        row["resultado"] = "ACIERTO" if assigned_cls == true_cls else "ERROR"
        dist_rows.append(row)

    distances_df = pd.DataFrame(dist_rows)
    distances_df.to_csv(distances_csv_path, index=False)

    # 10.4 Guardar matrices de confusión en CSV (cruda, filas, columnas)
    cm_raw_df = pd.DataFrame(
        cm,
        index=[f"real_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes],
    )
    cm_rows_df = pd.DataFrame(
        cm_rows,
        index=[f"real_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes],
    )
    cm_cols_df = pd.DataFrame(
        cm_cols,
        index=[f"real_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes],
    )

    cm_raw_df.to_csv(cm_raw_csv_path, index=False)
    cm_rows_df.to_csv(cm_rows_csv_path, index=False)
    cm_cols_df.to_csv(cm_cols_csv_path, index=False)

    print("\nResultados guardados en:")
    print(f"  Resumen TXT:              {results_txt_path}")
    print(f"  Centroides CSV:           {centroids_csv_path}")
    print(f"  Distancias CSV:           {distances_csv_path}")
    print(f"  Matriz confusión cruda:   {cm_raw_csv_path}")
    print(f"  Matriz conf. por filas:   {cm_rows_csv_path}")
    print(f"  Matriz conf. por columnas:{cm_cols_csv_path}")
