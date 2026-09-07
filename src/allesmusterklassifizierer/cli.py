from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
from .artifacts import load_model
from .audit import write_audit
from .dataset import validate_prediction_schema
from .errors import AMKError
from .experiment import prepare, run
from .metrics.confusion_demo import save_confusion_demo
from .v1config import load_v1_config


def parser():
    p = argparse.ArgumentParser(
        prog="amk", description="Clasificación tabular reproducible AMK 0.2"
    )
    sub = p.add_subparsers(dest="command", required=True)
    config = sub.add_parser("config")
    cs = config.add_subparsers(dest="action", required=True)
    cv = cs.add_parser("validate")
    cv.add_argument("--config", required=True)
    for name in ["audit", "split", "train", "evaluate", "run", "compare"]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
    predict = sub.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--data", required=True)
    predict.add_argument("--output")
    inspect = sub.add_parser("inspect-model")
    inspect.add_argument("--model", required=True)
    demo = sub.add_parser(
        "confusion-demo",
        help="Genera una matriz binaria conceptual o con valores manuales",
    )
    demo.add_argument(
        "--values",
        nargs=4,
        type=float,
        metavar=("TP", "FN", "FP", "TN"),
        help="Valores manuales en el orden TP FN FP TN",
    )
    demo.add_argument("--color", default="Reds", help="Paleta de matplotlib")
    demo.add_argument("--title", default="Matriz de confusión")
    demo.add_argument("--class-names", nargs=2, default=("Positivo", "Negativo"))
    demo.add_argument("--output", default="confusion_demo.png")
    demo.add_argument("--dpi", type=int, default=300)
    demo.add_argument(
        "--size", nargs=2, type=float, default=(7.0, 6.0), metavar=("ANCHO", "ALTO")
    )
    demo.add_argument("--font-size", type=float, default=20.0)
    demo.add_argument("--text-color")
    demo.add_argument("--background-color", default="white")
    demo.add_argument("--hide-axis-labels", action="store_true")
    demo.add_argument("--show-colorbar", action="store_true")
    demo.add_argument("--transparent", action="store_true")
    demo.add_argument("--language", choices=("es", "en"), default="es")
    for name in ["validate", "classify"]:
        sp = sub.add_parser(name, help=f"Flujo legado {name}")
        sp.add_argument("--config", required=True)
        sp.add_argument("--outputs-dir", default="outputs")
        sp.add_argument("--set", action="append", default=[])
    return p


def _read(path: Path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise AMKError(f"Formato de predicción no soportado: {path.suffix}")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "config":
            cfg = load_v1_config(args.config)
            print(f"OK config v{cfg.format_version}: {cfg.run_name}")
            return
        if args.command in {"audit", "split"}:
            cfg = load_v1_config(args.config)
            audit, splits = prepare(cfg)
            out = cfg.output_dir / args.command / cfg.run_name
            out.mkdir(parents=True, exist_ok=True)
            write_audit(audit, out)
            if args.command == "split":
                (out / "splits.json").write_text(
                    json.dumps(
                        [
                            {
                                "fold": s.fold,
                                "train": s.train_idx.tolist(),
                                "validation": (
                                    s.validation_idx.tolist()
                                    if s.validation_idx is not None
                                    else None
                                ),
                                "test": s.test_idx.tolist(),
                            }
                            for s in splits
                        ],
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            print(f"OK {args.command} -> {out}")
            return
        if args.command in {"train", "evaluate", "run", "compare"}:
            print(
                f"OK {args.command} -> {run(load_v1_config(args.config),compare=args.command=='compare')}"
            )
            return
        if args.command == "predict":
            a = load_model(args.model)
            X = validate_prediction_schema(
                _read(Path(args.data)), a.feature_names, extra=a.extra_columns
            )
            pred = a.pipeline.predict(X)
            out = pd.DataFrame({"prediction": pred})
            if hasattr(a.pipeline, "predict_proba"):
                probs = a.pipeline.predict_proba(X)
                for i, c in enumerate(a.classes):
                    out[f"probability_{c}"] = probs[:, i]
            target = Path(args.output or "predictions.csv")
            out.to_csv(target, index=False)
            print(f"OK predict -> {target}")
            return
        if args.command == "inspect-model":
            a = load_model(args.model)
            print(
                json.dumps(
                    {
                        "artifact_version": a.artifact_version,
                        "features": a.feature_names,
                        "classes": list(map(str, a.classes)),
                        "schema": a.schema,
                        "metadata": a.metadata,
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )
            return
        if args.command == "confusion-demo":
            target = save_confusion_demo(
                args.output,
                values=args.values,
                color=args.color,
                title=args.title,
                class_names=args.class_names,
                dpi=args.dpi,
                size=tuple(args.size),
                font_size=args.font_size,
                text_color=args.text_color,
                background_color=args.background_color,
                show_axis_labels=not args.hide_axis_labels,
                show_colorbar=args.show_colorbar,
                transparent=args.transparent,
                language=args.language,
            )
            print(f"OK confusion-demo -> {target}")
            return
        from .config import load_config
        from .runner import classify_only, validate_only

        cfg = load_config(args.config, overrides=args.set)
        fn = validate_only if args.command == "validate" else classify_only
        print(
            f"OK legacy {args.command} -> {fn(cfg,outputs_dir=args.outputs_dir).run_dir}"
        )
    except AMKError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except Exception as exc:
        print(f"ERROR inesperado: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
