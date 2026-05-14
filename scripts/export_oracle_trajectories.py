#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
import tomllib
from typing import Any, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


_LOCAL_PACKAGE_ROOT = _repo_root() / "environments" / "megaminx_solver"
sys.path.insert(0, str(_LOCAL_PACKAGE_ROOT))

from megaminx_solver import load_environment

from megaminx_solver.megaminx_solver import (  # noqa: E402
    action_gated_candidate_path_tail_solve_reward,
    candidate_path_completed,
    first_candidate_relative_flow_count,
    first_candidate_relative_flow_is_candidate_max,
    first_candidate_relative_flow_margin,
    second_candidate_relative_flow_count,
    second_candidate_relative_flow_is_candidate_max,
    second_candidate_relative_flow_margin,
    second_target_candidate_index,
)
from verifiers.types import AssistantMessage  # noqa: E402


EXPECTED_ENV_VERSION = "0.2.57"
DEFAULT_SPLIT = "train_candidate_relative_flow_rule_tail_solve_depth2"
PROMPT_STYLE = "stage_candidate_relative_flow_rule_solve2_native_tool"
REWARD_STYLE = "action_gated_candidate_path_tail_solve"
TOOL_NAME = "select_candidate"
TOOL_CALL_IDS = ("call_1", "call_2")


def _disable_progress_bars() -> None:
    try:
        from datasets.utils.logging import disable_progress_bar
    except Exception:
        return
    disable_progress_bar()


def _installed_env_version() -> str:
    pyproject = tomllib.loads((_LOCAL_PACKAGE_ROOT / "pyproject.toml").read_text())
    return str(pyproject["project"]["version"])


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _decode_moves(raw_moves: str) -> list[dict[str, str]]:
    parsed = json.loads(raw_moves)
    if not isinstance(parsed, list):
        raise ValueError("encoded moves must decode to a list")
    moves: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("encoded move entries must be objects")
        face = item.get("face")
        direction = item.get("direction")
        if not isinstance(face, str) or not isinstance(direction, str):
            raise ValueError("encoded moves must include string face and direction")
        moves.append({"face": face, "direction": direction})
    return moves


def _tool_defs_json(env: Any) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in env.tool_defs:
        if hasattr(tool, "model_dump"):
            tools.append(tool.model_dump(mode="json"))
        else:
            tools.append(dict(tool))
    return tools


def _tool_call(call_id: str, args: dict[str, Any]) -> dict[str, str]:
    return {
        "id": call_id,
        "name": TOOL_NAME,
        "arguments": _json_dumps(args),
    }


def _assistant_message(tool_call: dict[str, str]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [tool_call],
    }


def _tool_message(message: Any) -> dict[str, Any]:
    return {
        "role": getattr(message, "role", "tool"),
        "tool_call_id": getattr(message, "tool_call_id"),
        "content": getattr(message, "content"),
    }


def _as_candidate_faces(raw_faces: Any) -> list[str]:
    if not isinstance(raw_faces, list) or len(raw_faces) != 4:
        raise ValueError("expected four candidate faces")
    if not all(isinstance(face, str) for face in raw_faces):
        raise ValueError("candidate faces must be strings")
    return list(raw_faces)


def _oracle_action(
    *,
    turn: int,
    call_id: str,
    candidate_faces: Sequence[str],
    move: tuple[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    face, direction = move
    try:
        index = list(candidate_faces).index(face) + 1
    except ValueError as exc:
        raise ValueError(f"oracle face {face!r} is absent from turn {turn} candidates") from exc
    args = {"index": index, "direction": direction}
    action = {
        "turn": turn,
        "tool": TOOL_NAME,
        "arguments": args,
        "face": face,
        "candidate_faces": list(candidate_faces),
        "target_index": index,
    }
    return action, _tool_call(call_id, args)


def _validate_row(row: dict[str, Any]) -> None:
    if row.get("scramble_depth") != 2:
        raise ValueError(
            f"split {row.get('split')!r} resolved to depth {row.get('scramble_depth')}; "
            "use a depth-2 tail-solve split"
        )
    if row.get("move_budget") != 2:
        raise ValueError(f"expected move_budget=2, got {row.get('move_budget')!r}")
    if row.get("reward_style") != REWARD_STYLE:
        raise ValueError(f"expected reward_style={REWARD_STYLE!r}")
    if row.get("prompt_style") != PROMPT_STYLE:
        raise ValueError(f"expected prompt_style={PROMPT_STYLE!r}")
    if len(_decode_moves(row["inverse_solution"])) != 2:
        raise ValueError("expected exactly two oracle inverse moves")


async def _generate_record(
    env: Any,
    row_index: int,
    raw_row: Any,
    seed: int,
    env_version: str,
) -> dict[str, Any]:
    row = dict(raw_row)
    _validate_row(row)

    state: dict[str, Any] = {"task": row}
    await env.setup_state(state)
    rollout = state["megaminx"]

    initial_candidate_faces = _as_candidate_faces(row.get("candidate_faces"))
    first_action, first_tool_call = _oracle_action(
        turn=1,
        call_id=TOOL_CALL_IDS[0],
        candidate_faces=initial_candidate_faces,
        move=rollout.inverse_solution[0],
    )
    first_response = await env.env_response(
        [
            AssistantMessage(
                content="",
                tool_calls=[first_tool_call],
            )
        ],
        state,
    )
    if len(first_response) != 1:
        raise RuntimeError(f"expected one first tool response, got {len(first_response)}")
    first_reward = await action_gated_candidate_path_tail_solve_reward(state)

    second_candidate_faces = list(rollout.second_candidate_faces or rollout.candidate_faces)
    second_action, second_tool_call = _oracle_action(
        turn=2,
        call_id=TOOL_CALL_IDS[1],
        candidate_faces=second_candidate_faces,
        move=rollout.inverse_solution[1],
    )
    second_response = await env.env_response(
        [
            AssistantMessage(
                content="",
                tool_calls=[second_tool_call],
            )
        ],
        state,
    )
    if len(second_response) != 1:
        raise RuntimeError(f"expected one second tool response, got {len(second_response)}")

    final_reward = await action_gated_candidate_path_tail_solve_reward(state)
    if final_reward != 1.0 or not rollout.solved():
        raise RuntimeError(
            f"oracle did not solve {row.get('example_id')}: reward={final_reward} solved={rollout.solved()}"
        )

    prompt_messages = list(row["prompt"])
    if not prompt_messages or prompt_messages[0].get("role") != "system":
        prompt_messages.insert(0, {"role": "system", "content": env.system_prompt})
    messages = [
        *prompt_messages,
        _assistant_message(first_tool_call),
        _tool_message(first_response[0]),
        _assistant_message(second_tool_call),
        _tool_message(second_response[0]),
    ]

    return {
        "format": "megaminx-oracle-select-candidate-jsonl/v1",
        "env_id": "megaminx-solver",
        "env_version": env_version,
        "expected_env_version": EXPECTED_ENV_VERSION,
        "example_id": row["example_id"],
        "row_index": row_index,
        "split": row["split"],
        "seed": seed,
        "scramble_depth": row["scramble_depth"],
        "move_budget": row["move_budget"],
        "reward_style": row["reward_style"],
        "prompt_style": row["prompt_style"],
        "scramble": _decode_moves(row["scramble"]),
        "inverse_solution": _decode_moves(row["inverse_solution"]),
        "candidate_seed": row.get("candidate_seed"),
        "candidate_faces_initial": initial_candidate_faces,
        "candidate_faces_after_first": second_candidate_faces,
        "tools": _tool_defs_json(env),
        "messages": messages,
        "actions": [first_action, second_action],
        "metrics": {
            "first_step_reward": first_reward,
            "final_reward": final_reward,
            "solved": rollout.solved(),
            "move_count": rollout.move_count,
            "candidate_path_completed": await candidate_path_completed(state),
            "second_target_candidate_index": await second_target_candidate_index(state),
            "first_candidate_relative_flow_count": await first_candidate_relative_flow_count(state),
            "first_candidate_relative_flow_margin": await first_candidate_relative_flow_margin(state),
            "first_candidate_relative_flow_is_candidate_max": (
                await first_candidate_relative_flow_is_candidate_max(state)
            ),
            "second_candidate_relative_flow_count": await second_candidate_relative_flow_count(state),
            "second_candidate_relative_flow_margin": await second_candidate_relative_flow_margin(state),
            "second_candidate_relative_flow_is_candidate_max": (
                await second_candidate_relative_flow_is_candidate_max(state)
            ),
        },
    }


async def export_oracle_trajectories(
    *,
    num_examples: int,
    seed: int,
    split: str,
    output: Path,
) -> int:
    if num_examples <= 0:
        raise ValueError("--num-examples must be positive")

    _disable_progress_bars()
    env_version = _installed_env_version()
    if env_version != EXPECTED_ENV_VERSION:
        raise RuntimeError(
            f"expected megaminx-solver {EXPECTED_ENV_VERSION}, found {env_version}"
        )

    env = load_environment(
        split=split,
        min_depth=2,
        max_depth=2,
        num_examples=num_examples,
        seed=seed,
        max_turns=4,
        move_budget=2,
        reward_style=REWARD_STYLE,
        prompt_style=PROMPT_STYLE,
        allow_text_tool_actions=False,
    )

    dataset = env.get_dataset()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row_index, raw_row in enumerate(dataset):
            record = await _generate_record(env, row_index, raw_row, seed, env_version)
            handle.write(_json_dumps(record))
            handle.write("\n")
    return len(dataset)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export deterministic v0.2.57 depth-2 Megaminx oracle select_candidate "
            "trajectories as JSONL."
        )
    )
    parser.add_argument("--num-examples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    count = asyncio.run(
        export_oracle_trajectories(
            num_examples=args.num_examples,
            seed=args.seed,
            split=args.split,
            output=args.output,
        )
    )
    print(f"Wrote {count} oracle trajectories to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
