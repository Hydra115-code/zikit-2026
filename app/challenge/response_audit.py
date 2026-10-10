from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from app.challenge.ingestion import assert_allowed_path
from app.challenge.response import build_response_audit, response_catalog


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit_replay_response(
    replay_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    replay = assert_allowed_path(replay_directory)
    alerts_path = assert_allowed_path(replay / "alerts.full.parquet")
    replay_report_path = assert_allowed_path(replay / "replay-report.json")
    alerts = pd.read_parquet(alerts_path)
    report = build_response_audit(alerts)
    report["lineage"] = {
        "alerts_sha256": _sha256_file(alerts_path),
        "replay_report_sha256": _sha256_file(replay_report_path),
    }

    output = assert_allowed_path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "response-audit.json"
    catalog_path = output / "response-catalog.json"
    _atomic_json(report, report_path)
    _atomic_json({"plans": response_catalog()}, catalog_path)
    manifest = {
        "status": "passed" if report["passed"] else "failed",
        "outputs": {
            report_path.name: _sha256_file(report_path),
            catalog_path.name: _sha256_file(catalog_path),
        },
    }
    _atomic_json(manifest, output / "manifest.json")
    return {**report, "output_directory": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit deterministic Kaiku response plans without executing them"
    )
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-gates", action="store_true")
    args = parser.parse_args()
    result = audit_replay_response(args.replay, args.output)
    print(
        json.dumps(
            {
                "status": "passed" if result["passed"] else "failed",
                "alerts_evaluated": result["alerts_evaluated"],
                "output_directory": result["output_directory"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_gates and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
