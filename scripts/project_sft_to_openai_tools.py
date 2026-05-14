#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from summarize_oracle_trajectories import EXPECTED_ENV_VERSION, _read_jsonl
from validate_sft_jsonl import validate_sft_jsonl


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _project_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["parameters"],
        },
    }


def _project_message(message: dict[str, Any], tool_call_names: dict[str, str]) -> dict[str, Any]:
    role = message.get("role")
    if role == "assistant":
        projected = {"role": "assistant", "content": message.get("content", "")}
        tool_calls = []
        for tool_call in message.get("tool_calls", []):
            call_id = tool_call["id"]
            name = tool_call["name"]
            tool_call_names[call_id] = name
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": tool_call["arguments"],
                    },
                }
            )
        if tool_calls:
            projected["tool_calls"] = tool_calls
        return projected

    if role == "tool":
        tool_call_id = message["tool_call_id"]
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_call_names[tool_call_id],
            "content": message.get("content", ""),
        }

    return {"role": role, "content": message.get("content", "")}


def project_record_to_openai_tools(record: dict[str, Any]) -> dict[str, Any]:
    tool_call_names: dict[str, str] = {}
    projected = {
        "messages": [
            _project_message(message, tool_call_names)
            for message in record["messages"]
        ],
        "tools": [_project_tool(tool) for tool in record["tools"]],
    }
    if "metadata" in record:
        projected["metadata"] = record["metadata"]
    return projected


def project_sft_to_openai_tools(
    *,
    input_path: Path,
    output_path: Path,
    expected_env_version: str = EXPECTED_ENV_VERSION,
) -> int:
    validate_sft_jsonl(input_path, expected_env_version=expected_env_version)
    records = _read_jsonl(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_json_dumps(project_record_to_openai_tools(record)))
            handle.write("\n")
    return len(records)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project validated Megaminx flat tool-call SFT JSONL to OpenAI-style "
            "function tool-call records."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-env-version", default=EXPECTED_ENV_VERSION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    count = project_sft_to_openai_tools(
        input_path=args.input,
        output_path=args.output,
        expected_env_version=args.expected_env_version,
    )
    print(f"Wrote {count} OpenAI-style tool-call SFT records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
