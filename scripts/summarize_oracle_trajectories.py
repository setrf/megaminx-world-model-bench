#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


EXPECTED_ENV_VERSION = "0.2.56"
EXPECTED_FORMAT = "megaminx-oracle-select-candidate-jsonl/v1"
EXPECTED_TOOL = "select_candidate"
EXPECTED_ROLES = ("system", "user", "assistant", "tool", "assistant", "tool")
EXPECTED_TOOL_CALL_IDS = ("call_1", "call_2")


def _json_counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=lambda item: str(item))}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def _validate_record(record: dict[str, Any], *, expected_env_version: str) -> list[str]:
    errors: list[str] = []
    row_id = record.get("row_index", record.get("example_id", "?"))
    prefix = f"row {row_id}: "

    if record.get("format") != EXPECTED_FORMAT:
        errors.append(prefix + f"unexpected format {record.get('format')!r}")
    if record.get("env_version") != expected_env_version:
        errors.append(prefix + f"unexpected env_version {record.get('env_version')!r}")
    if record.get("expected_env_version") != expected_env_version:
        errors.append(prefix + f"unexpected expected_env_version {record.get('expected_env_version')!r}")
    if record.get("scramble_depth") != 2:
        errors.append(prefix + f"unexpected scramble_depth {record.get('scramble_depth')!r}")
    if record.get("move_budget") != 2:
        errors.append(prefix + f"unexpected move_budget {record.get('move_budget')!r}")

    actions = record.get("actions")
    if not isinstance(actions, list) or len(actions) != 2:
        errors.append(prefix + "expected exactly two actions")
    else:
        for turn, action in enumerate(actions, start=1):
            if not isinstance(action, dict):
                errors.append(prefix + f"action {turn} is not an object")
                continue
            if action.get("tool") != EXPECTED_TOOL:
                errors.append(prefix + f"action {turn} uses tool {action.get('tool')!r}")
            args = action.get("arguments")
            if not isinstance(args, dict):
                errors.append(prefix + f"action {turn} has invalid arguments")
            else:
                if args.get("index") not in {1, 2, 3, 4}:
                    errors.append(prefix + f"action {turn} has invalid index {args.get('index')!r}")
                if args.get("direction") not in {"cw", "ccw"}:
                    errors.append(prefix + f"action {turn} has invalid direction {args.get('direction')!r}")

    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        errors.append(prefix + "missing metrics")
    else:
        if metrics.get("first_step_reward") != 0.25:
            errors.append(prefix + f"unexpected first_step_reward {metrics.get('first_step_reward')!r}")
        if metrics.get("final_reward") != 1.0:
            errors.append(prefix + f"unexpected final_reward {metrics.get('final_reward')!r}")
        if metrics.get("solved") is not True:
            errors.append(prefix + f"unexpected solved {metrics.get('solved')!r}")
        if metrics.get("candidate_path_completed") != 1.0:
            errors.append(
                prefix + f"unexpected candidate_path_completed {metrics.get('candidate_path_completed')!r}"
            )

    messages = record.get("messages")
    if not isinstance(messages, list):
        errors.append(prefix + "missing messages")
    else:
        roles = tuple(message.get("role") for message in messages if isinstance(message, dict))
        if roles != EXPECTED_ROLES:
            errors.append(prefix + f"unexpected message roles {roles!r}")
        assistant_messages = [
            message
            for message in messages
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        tool_messages = [
            message
            for message in messages
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        for turn, (message, expected_id) in enumerate(
            zip(assistant_messages, EXPECTED_TOOL_CALL_IDS, strict=False),
            start=1,
        ):
            if message.get("content", "") != "":
                errors.append(prefix + f"assistant message {turn} must have empty content")
            for field in ("reasoning", "reasoning_content", "thinking", "thinking_content"):
                if field in message:
                    errors.append(prefix + f"assistant message {turn} must not contain {field!r}")
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list) or len(tool_calls) != 1:
                errors.append(prefix + f"assistant message {turn} must have exactly one tool call")
                continue
            tool_call = tool_calls[0]
            if not isinstance(tool_call, dict):
                errors.append(prefix + f"assistant message {turn} tool call must be an object")
                continue
            if tool_call.get("id") != expected_id:
                errors.append(
                    prefix
                    + f"assistant message {turn} tool_call id must be {expected_id!r}, got {tool_call.get('id')!r}"
                )
            if tool_call.get("name") != EXPECTED_TOOL:
                errors.append(prefix + f"assistant message {turn} calls {tool_call.get('name')!r}")
            actions = record.get("actions")
            expected_args = None
            if isinstance(actions, list) and turn <= len(actions) and isinstance(actions[turn - 1], dict):
                expected_args = actions[turn - 1].get("arguments")
            if not isinstance(expected_args, dict):
                errors.append(prefix + f"missing action arguments for assistant message {turn}")
            elif tool_call.get("arguments") != _json_dumps(expected_args):
                errors.append(
                    prefix
                    + f"assistant message {turn} arguments must be exact compact JSON for oracle action"
                )
        for turn, (message, expected_id) in enumerate(
            zip(tool_messages, EXPECTED_TOOL_CALL_IDS, strict=False),
            start=1,
        ):
            if message.get("tool_call_id") != expected_id:
                errors.append(
                    prefix
                    + f"tool message {turn} tool_call_id must be {expected_id!r}, got {message.get('tool_call_id')!r}"
                )
        prompt = "\n".join(
            message.get("content", "")
            for message in messages
            if isinstance(message, dict) and message.get("role") in {"system", "user"}
        )
        if "Example:" in prompt:
            errors.append(prefix + "prompt leaks old Example: marker")
        if isinstance(record.get("row_index"), int) and f"{record['row_index']:05d}" in prompt:
            errors.append(prefix + "prompt leaks zero-padded row id")
        transcript = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        if isinstance(record.get("row_index"), int) and f"oracle-{record['row_index']}" in transcript:
            errors.append(prefix + "transcript leaks row-derived oracle tool_call id")
        if "oracle-" in transcript:
            errors.append(prefix + "transcript contains legacy oracle-* tool_call id")

    return errors


def summarize_oracle_trajectories(
    path: Path,
    *,
    expected_env_version: str = EXPECTED_ENV_VERSION,
) -> dict[str, Any]:
    records = _read_jsonl(path)
    errors: list[str] = []
    for record in records:
        errors.extend(_validate_record(record, expected_env_version=expected_env_version))
    if errors:
        preview = "\n".join(errors[:10])
        raise ValueError(f"{path}: validation failed with {len(errors)} error(s)\n{preview}")

    first_actions = [record["actions"][0] for record in records]
    second_actions = [record["actions"][1] for record in records]
    all_actions = first_actions + second_actions

    return {
        "path": str(path),
        "rows": len(records),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "env_versions": _json_counter(Counter(record.get("env_version") for record in records)),
        "splits": _json_counter(Counter(record.get("split") for record in records)),
        "final_rewards": _json_counter(Counter(record["metrics"]["final_reward"] for record in records)),
        "solved": _json_counter(Counter(record["metrics"]["solved"] for record in records)),
        "action_counts": _json_counter(Counter(len(record["actions"]) for record in records)),
        "first_slot_counts": _json_counter(
            Counter(action["arguments"]["index"] for action in first_actions)
        ),
        "second_slot_counts": _json_counter(
            Counter(action["arguments"]["index"] for action in second_actions)
        ),
        "all_slot_counts": _json_counter(Counter(action["arguments"]["index"] for action in all_actions)),
        "first_direction_counts": _json_counter(
            Counter(action["arguments"]["direction"] for action in first_actions)
        ),
        "second_direction_counts": _json_counter(
            Counter(action["arguments"]["direction"] for action in second_actions)
        ),
        "all_direction_counts": _json_counter(
            Counter(action["arguments"]["direction"] for action in all_actions)
        ),
        "prompt_leak_count": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize Megaminx oracle trajectory JSONL."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--expected-env-version", default=EXPECTED_ENV_VERSION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_oracle_trajectories(
        args.input,
        expected_env_version=args.expected_env_version,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
