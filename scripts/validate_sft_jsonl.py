#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from summarize_oracle_trajectories import (  # noqa: E402
    EXPECTED_ENV_VERSION,
    EXPECTED_ROLES,
    EXPECTED_TOOL,
    EXPECTED_TOOL_CALL_IDS,
    _json_counter,
    _read_jsonl,
    _sha256,
)


FORBIDDEN_TOP_LEVEL_KEYS = {
    "actions",
    "candidate_faces_after_first",
    "candidate_faces_initial",
    "inverse_solution",
    "metrics",
    "scramble",
}
FORBIDDEN_METADATA_KEYS = {
    "actions",
    "candidate_faces_after_first",
    "candidate_faces_initial",
    "inverse_solution",
    "metrics",
    "scramble",
}


def _tool_call_args(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    raw_args = tool_call.get("arguments")
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _validate_sft_record(record: dict[str, Any], *, expected_env_version: str) -> list[str]:
    errors: list[str] = []
    metadata = record.get("metadata", {})
    row_id = metadata.get("row_index", metadata.get("example_id", "?")) if isinstance(metadata, dict) else "?"
    prefix = f"row {row_id}: "

    forbidden_top = FORBIDDEN_TOP_LEVEL_KEYS.intersection(record)
    if forbidden_top:
        errors.append(prefix + f"forbidden top-level keys present: {sorted(forbidden_top)}")

    if set(record) - {"messages", "metadata", "tools"}:
        errors.append(prefix + f"unexpected top-level keys: {sorted(set(record) - {'messages', 'metadata', 'tools'})}")

    if not isinstance(metadata, dict):
        errors.append(prefix + "metadata must be an object")
    else:
        forbidden_metadata = FORBIDDEN_METADATA_KEYS.intersection(metadata)
        if forbidden_metadata:
            errors.append(prefix + f"forbidden metadata keys present: {sorted(forbidden_metadata)}")
        if metadata.get("env_version") != expected_env_version:
            errors.append(prefix + f"unexpected metadata env_version {metadata.get('env_version')!r}")
        if metadata.get("expected_env_version") != expected_env_version:
            errors.append(
                prefix + f"unexpected metadata expected_env_version {metadata.get('expected_env_version')!r}"
            )

    tools = record.get("tools")
    if not isinstance(tools, list) or len(tools) != 1:
        errors.append(prefix + "expected exactly one tool definition")
    elif tools[0].get("name") != EXPECTED_TOOL:
        errors.append(prefix + f"unexpected tool {tools[0].get('name')!r}")

    messages = record.get("messages")
    if not isinstance(messages, list):
        errors.append(prefix + "messages must be a list")
        return errors

    roles = tuple(message.get("role") for message in messages if isinstance(message, dict))
    if roles != EXPECTED_ROLES:
        errors.append(prefix + f"unexpected message roles {roles!r}")

    assistant_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if len(assistant_messages) != 2:
        errors.append(prefix + f"expected two assistant tool-call messages, got {len(assistant_messages)}")

    for index, message in enumerate(assistant_messages, start=1):
        if message.get("content", "") != "":
            errors.append(prefix + f"assistant message {index} must have empty content")
        for field in ("reasoning", "reasoning_content", "thinking", "thinking_content"):
            if field in message:
                errors.append(prefix + f"assistant message {index} must not contain {field!r}")
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or len(tool_calls) != 1:
            errors.append(prefix + f"assistant message {index} must have exactly one tool call")
            continue
        tool_call = tool_calls[0]
        if not isinstance(tool_call, dict):
            errors.append(prefix + f"assistant message {index} tool call must be an object")
            continue
        if tool_call.get("name") != EXPECTED_TOOL:
            errors.append(prefix + f"assistant message {index} calls {tool_call.get('name')!r}")
        expected_tool_call_id = (
            EXPECTED_TOOL_CALL_IDS[index - 1]
            if index <= len(EXPECTED_TOOL_CALL_IDS)
            else f"unexpected-{index}"
        )
        if tool_call.get("id") != expected_tool_call_id:
            errors.append(
                prefix
                + f"assistant message {index} tool_call id must be {expected_tool_call_id!r}, got {tool_call.get('id')!r}"
            )
        args = _tool_call_args(tool_call)
        if args is None:
            errors.append(prefix + f"assistant message {index} has invalid tool arguments")
            continue
        expected_arg_string = json.dumps(args, sort_keys=True, separators=(",", ":"))
        if tool_call.get("arguments") != expected_arg_string:
            errors.append(
                prefix
                + f"assistant message {index} arguments must be compact sorted JSON string"
            )
        if args.get("index") not in {1, 2, 3, 4}:
            errors.append(prefix + f"assistant message {index} has invalid index {args.get('index')!r}")
        if args.get("direction") not in {"cw", "ccw"}:
            errors.append(prefix + f"assistant message {index} has invalid direction {args.get('direction')!r}")

    tool_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    ]
    for index, message in enumerate(tool_messages, start=1):
        expected_tool_call_id = (
            EXPECTED_TOOL_CALL_IDS[index - 1]
            if index <= len(EXPECTED_TOOL_CALL_IDS)
            else f"unexpected-{index}"
        )
        if message.get("tool_call_id") != expected_tool_call_id:
            errors.append(
                prefix
                + f"tool message {index} tool_call_id must be {expected_tool_call_id!r}, got {message.get('tool_call_id')!r}"
            )

    prompt = "\n".join(
        message.get("content", "")
        for message in messages
        if isinstance(message, dict) and message.get("role") in {"system", "user"}
    )
    if "Example:" in prompt:
        errors.append(prefix + "prompt leaks old Example: marker")
    if isinstance(metadata, dict) and isinstance(metadata.get("row_index"), int):
        if f"{metadata['row_index']:05d}" in prompt:
            errors.append(prefix + "prompt leaks zero-padded row id")
        transcript = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        if f"oracle-{metadata['row_index']}" in transcript:
            errors.append(prefix + "transcript leaks row-derived oracle tool_call id")
        if "oracle-" in transcript:
            errors.append(prefix + "transcript contains legacy oracle-* tool_call id")

    return errors


def validate_sft_jsonl(
    path: Path,
    *,
    expected_env_version: str = EXPECTED_ENV_VERSION,
) -> dict[str, Any]:
    records = _read_jsonl(path)
    errors: list[str] = []
    for record in records:
        errors.extend(_validate_sft_record(record, expected_env_version=expected_env_version))
    if errors:
        preview = "\n".join(errors[:10])
        raise ValueError(f"{path}: validation failed with {len(errors)} error(s)\n{preview}")

    metadata_rows = [record.get("metadata", {}) for record in records]
    return {
        "path": str(path),
        "rows": len(records),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "env_versions": _json_counter(Counter(row.get("env_version") for row in metadata_rows)),
        "splits": _json_counter(Counter(row.get("split") for row in metadata_rows)),
        "message_role_patterns": _json_counter(
            Counter(tuple(message.get("role") for message in record["messages"]) for record in records)
        ),
        "tool_names": _json_counter(Counter(record["tools"][0]["name"] for record in records)),
        "prompt_leak_count": 0,
        "forbidden_payload_fields": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate lean Megaminx SFT JSONL records.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--expected-env-version", default=EXPECTED_ENV_VERSION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = validate_sft_jsonl(args.input, expected_env_version=args.expected_env_version)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
