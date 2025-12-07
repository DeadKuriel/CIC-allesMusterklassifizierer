from __future__ import annotations
from pathlib import Path

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
        print("Conjunto de ENTRENAMIENTO BALANCEADO (IR ≤ 1.5). Es razonable usar accuracy.")
    else:
        print(
            "Conjunto de ENTRENAMIENTO DESBALANCEADO (IR > 1.5). "
            "No es razonable reportar solo accuracy."
        )

    # 5. Entrenar clasificador euclidiano
    clf = EuclideanClassifier()
    clf.fit(X_train, y_train)

    # 6. Decidir si calculamos accuracy según IR de TRAIN
    acc: float | None = None
    if ir_train <= IR_THRESHOLD:
        y_pred = clf.predict(X_test)
        acc = accuracy(y_test, y_pred)
        print(f"\nAccuracy, clasificador Euclidiano (hold-out): {acc:.4f}")
    else:
        print(
            "\nNo se reporta accuracy porque el conjunto de entrenamiento está desbalanceado "
            f"(IR_train = {ir_train:.4f} > {IR_THRESHOLD})."
        )

    print(f"Semilla usada: {used_seed}")

    # 7. Guardar resultados simples a texto
    resultados_dir = Path("output/euclidean")
    resultados_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(test_size, float):
        test_pct = int(round(test_size * 100))
    else:
        total = len(train_df) + len(test_df)
        test_pct = int(round(len(test_df) * 100 / total))

    results_filename = (
        f"{timestamp}_{dataset_resolved}_seed{used_seed}_test{test_pct}_euclidean.txt"
    )
    results_path = resultados_dir / results_filename

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_resolved}\n")
        f.write(f"Target: {target_col}\n")
        f.write(f"Validación: hold-out (test_size={test_size})\n")
        f.write(f"Semilla: {used_seed}\n")
        f.write(f"IR en TRAIN (Imbalance Ratio): {ir_train:.4f}\n")
        f.write("Criterio: IR ≤ 1.5 -> balanceado, IR > 1.5 -> desbalanceado\n")

        if acc is not None:
            f.write(f"Accuracy: {acc:.4f}\n")
        else:
            f.write(
                "Accuracy NO reportado: el conjunto de entrenamiento es desbalanceado "
                "(IR_train > 1.5), por lo que no es razonable usar solo accuracy.\n"
            )

    print(f"\nResultados guardados en: {results_path}")
