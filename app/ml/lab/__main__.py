from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ml.lab.pipeline import benchmark_model, load_dataset, train_lab_model, write_dataset
from app.ml.lab.scenarios import GenerationConfig, generate_dataset


def _json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic ML laboratory for Zikit Sentinel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate labeled, reproducible scenario runs")
    generate.add_argument("--output", type=Path, default=Path("data/lab"))
    generate.add_argument("--normal-runs", type=int, default=500)
    generate.add_argument("--fault-runs", type=int, default=1_000)
    generate.add_argument("--run-length", type=int, default=120)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--train-fraction", type=float, default=0.6)
    generate.add_argument("--validation-fraction", type=float, default=0.2)

    train_parser = subparsers.add_parser("train", help="Train on normal windows and calibrate on validation")
    train_parser.add_argument("--input", type=Path, default=Path("data/lab"))
    train_parser.add_argument("--output", type=Path, default=Path("models/isolation_forest.joblib"))
    train_parser.add_argument("--target-fpr", type=float, default=0.01)
    train_parser.add_argument("--seed", type=int, default=42)

    benchmark = subparsers.add_parser("benchmark", help="Evaluate held-out runs by scenario")
    benchmark.add_argument("--input", type=Path, default=Path("data/lab"))
    benchmark.add_argument("--model", type=Path, default=Path("models/isolation_forest.joblib"))
    benchmark.add_argument("--output", type=Path, default=Path("reports/ml-benchmark.json"))
    benchmark.add_argument("--persistence-windows", type=int, default=3)
    benchmark.add_argument("--minimum-recall", type=float, default=0.80)
    benchmark.add_argument("--maximum-fpr", type=float, default=0.03)
    benchmark.add_argument("--maximum-p95-delay", type=float, default=12)
    benchmark.add_argument("--maximum-false-alert-run-rate", type=float, default=0.10)
    benchmark.add_argument("--minimum-scenario-detection-rate", type=float, default=0.90)
    benchmark.add_argument("--fail-on-gates", action="store_true")

    args = parser.parse_args()
    if args.command == "generate":
        config = GenerationConfig(
            normal_runs=args.normal_runs,
            fault_runs=args.fault_runs,
            run_length=args.run_length,
            seed=args.seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
        )
        manifest = write_dataset(generate_dataset(config), args.output, config)
        print(_json(manifest))
        return

    frame = load_dataset(args.input)
    if args.command == "train":
        report = train_lab_model(
            frame,
            args.output,
            target_false_positive_rate=args.target_fpr,
            seed=args.seed,
        )
        print(_json(report))
        return

    report = benchmark_model(
        frame,
        args.model,
        persistence_windows=args.persistence_windows,
        minimum_recall=args.minimum_recall,
        maximum_false_positive_rate=args.maximum_fpr,
        maximum_p95_delay_steps=args.maximum_p95_delay,
        maximum_false_alert_run_rate=args.maximum_false_alert_run_rate,
        minimum_scenario_detection_rate=args.minimum_scenario_detection_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_json(report) + "\n", encoding="utf-8")
    print(_json(report))
    if args.fail_on_gates and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
