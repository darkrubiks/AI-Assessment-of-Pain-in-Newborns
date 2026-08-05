"""Command-line entry point for the end-to-end real-time benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from benchmarking.realtime import (
    BenchmarkError,
    inspect_benchmark_config,
    load_benchmark_config,
    run_benchmark_suite,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mede latência, FPS, memória, filas, detecção, MCDP, ensemble, XAI e suavização."
    )
    parser.add_argument(
        "--config", default="models/configs/realtime_benchmark.yaml",
        help="Arquivo YAML do benchmark.",
    )
    parser.add_argument(
        "--scenario", action="append", dest="scenarios",
        help="Nome de cenário a executar (repita a opção para selecionar mais de um).",
    )
    parser.add_argument("--source", help="Sobrescreve source.path (ou índice, para câmera).")
    parser.add_argument("--output-dir", help="Sobrescreve output_dir.")
    parser.add_argument(
        "--check-config", action="store_true",
        help="Valida fonte/checkpoints e encerra sem carregar modelos nem escrever resultados.",
    )
    parser.add_argument("--list-scenarios", action="store_true", help="Lista os cenários habilitados.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        config = load_benchmark_config(args.config)
        if args.source is not None:
            config["source"]["path"] = args.source
            if config["source"].get("kind") == "camera":
                config["source"]["camera_index"] = int(args.source)
        if args.list_scenarios or args.check_config:
            inventory = inspect_benchmark_config(config, args.scenarios)
            print(json.dumps(inventory, ensure_ascii=False, indent=2))
            return 0
        output = run_benchmark_suite(
            config,
            scenario_names=args.scenarios,
            source_override=args.source,
            output_override=args.output_dir,
        )
        print(f"Benchmark concluído: {output}")
        return 0
    except (BenchmarkError, FileNotFoundError, ValueError, ImportError) as exc:
        logging.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
