from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

RCAEVAL_URI = "hf://datasets/phamquiluan/RCAEval/cases.parquet"


def load_cases(uri: str = RCAEVAL_URI) -> pd.DataFrame:
    """Load RCAEval exactly as published; this function does not infer feature mappings."""
    return pd.read_parquet(uri)


def describe_schema(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "columns": [{"name": name, "dtype": str(dtype)} for name, dtype in frame.dtypes.items()],
        "null_counts": {name: int(value) for name, value in frame.isna().sum().items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the optional RCAEval dataset")
    parser.add_argument("--output", type=Path, help="Optional local parquet destination")
    args = parser.parse_args()
    frame = load_cases()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.output, index=False)
    print(json.dumps(describe_schema(frame), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

