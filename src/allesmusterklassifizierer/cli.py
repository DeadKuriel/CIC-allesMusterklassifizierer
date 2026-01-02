from __future__ import annotations

import argparse
import sys
from typing import List

from .config import load_config
from .errors import AMKError
from .runner import classify_only, run_experiment, validate_only


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="amk", description="allesmusterklassifizierer CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser):
        sp.add_argument("--config", required=True, help="Ruta a YAML en configs/")
        sp.add_argument(
            "--set",
            action="append",
            default=[],
            help="Override estilo key.path=value (puede repetirse). Ej: --set classifier.params.k=7",
        )
        sp.add_argument("--outputs-dir", default="outputs", help="Directorio base de outputs (default: outputs)")

    s_validate = sub.add_parser("validate", help="Corre SOLO validación: splits + stats")
    add_common(s_validate)

    s_classify = sub.add_parser("classify", help="Corre SOLO clasificador")
    add_common(s_classify)

    s_run = sub.add_parser("run", help="Corre validación + clasificador (experimento completo)")
    add_common(s_run)

    return p


def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, overrides=args.set)

        if args.cmd == "validate":
            paths = validate_only(cfg, outputs_dir=args.outputs_dir)
            print(f"OK validate -> {paths.run_dir}")
            print(f"  meta: {paths.meta_json}")

        elif args.cmd == "classify":
            paths = classify_only(cfg, outputs_dir=args.outputs_dir)
            print(f"OK classify -> {paths.run_dir}")
            print(f"  meta: {paths.meta_json}")
            print(f"  predictions: {paths.predictions_csv}")
            print(f"  metrics: {paths.metrics_json}")

        elif args.cmd == "run":
            paths = run_experiment(cfg, outputs_dir=args.outputs_dir)
            print(f"OK run -> {paths.run_dir}")
            print(f"  meta: {paths.meta_json}")
            print(f"  fold_predictions: {paths.fold_predictions_csv}")
            print(f"  metrics: {paths.metrics_json}")

        else:
            raise RuntimeError("Comando no reconocido (bug).")

    except AMKError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
