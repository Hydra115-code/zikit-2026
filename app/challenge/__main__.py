from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.challenge.predict import predict_run
from app.challenge.submission import validate_submission_file


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.challenge",
        description="Zikit challenge causal prediction interfaces",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser(
        "predict",
        help="clean one public run and produce the official alert CSV",
    )
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument(
        "--model",
        type=Path,
        default=Path("models/challenge_early_warning.joblib"),
    )
    predict.add_argument(
        "--policy",
        type=Path,
        default=Path("models/challenge_alert_policy.json"),
    )
    validate = commands.add_parser(
        "validate-submission",
        help="validate an alert CSV against the evaluator-facing contract",
    )
    validate.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "predict":
        result = predict_run(
            args.input,
            args.output,
            args.model,
            args.policy,
            require_all_tenants=False,
        )
        print(
            json.dumps(
                {
                    "output_path": result["output_path"],
                    "manifest_path": result["manifest_path"],
                    "incident_alerts": result["output"]["incident_alerts"],
                    "data_quality_alerts": result["output"][
                        "data_quality_alerts"
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "validate-submission":
        print(
            json.dumps(
                validate_submission_file(args.input),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
