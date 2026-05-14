#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from summarize_oracle_trajectories import (  # noqa: E402
    EXPECTED_ENV_VERSION,
    _read_jsonl,
    _validate_record,
)


SAFE_METADATA_KEYS = (
    "env_id",
    "env_version",
    "expected_env_version",
    "example_id",
    "format",
    "move_budget",
    "prompt_style",
    "reward_style",
    "row_index",
    "scramble_depth",
    "seed",
    "split",
)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sft_record(record: dict[str, Any], *, include_metadata: bool) -> dict[str, Any]:
    converted = {
        "messages": record["messages"],
        "tools": record["tools"],
    }
    if include_metadata:
        converted["metadata"] = {
            key: record.get(key)
            for key in SAFE_METADATA_KEYS
            if key in record
        }
    return converted


def convert_oracle_to_sft_jsonl(
    *,
    input_path: Path,
    output_path: Path,
    expected_env_version: str = EXPECTED_ENV_VERSION,
    include_metadata: bool = True,
) -> int:
    records = _read_jsonl(input_path)
    validation_errors: list[str] = []
    for record in records:
        validation_errors.extend(_validate_record(record, expected_env_version=expected_env_version))
    if validation_errors:
        preview = "\n".join(validation_errors[:10])
        raise ValueError(
            f"{input_path}: validation failed with {len(validation_errors)} error(s)\n{preview}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_json_dumps(_sft_record(record, include_metadata=include_metadata)))
            handle.write("\n")
    return len(records)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert validated Megaminx oracle trajectory JSONL into lean "
            "messages/tools SFT JSONL."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-env-version", default=EXPECTED_ENV_VERSION)
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Write only messages/tools, without safe metadata.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    count = convert_oracle_to_sft_jsonl(
        input_path=args.input,
        output_path=args.output,
        expected_env_version=args.expected_env_version,
        include_metadata=not args.no_metadata,
    )
    print(f"Wrote {count} SFT records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
