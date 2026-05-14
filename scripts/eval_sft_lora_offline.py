#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _repo_root()
LOCAL_PACKAGE_ROOT = REPO_ROOT / "environments" / "megaminx_solver"
SCRIPT_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from megaminx_solver import load_environment  # noqa: E402
from megaminx_solver.megaminx_solver import (  # noqa: E402
    action_gated_candidate_path_tail_solve_reward,
)
from train_sft_lora import _chat_template_messages, _training_device  # noqa: E402
from verifiers.types import AssistantMessage  # noqa: E402


TOOL_NAME = "select_candidate"
PROMPT_STYLE = "stage_candidate_relative_flow_rule_solve2_native_tool"
REWARD_STYLE = "action_gated_candidate_path_tail_solve"
TOOL_CALL_IDS = ("call_1", "call_2")


def _disable_progress_bars() -> None:
    try:
        from datasets.utils.logging import disable_progress_bar
    except Exception:
        return
    disable_progress_bar()


@dataclass(frozen=True)
class HeldoutSpec:
    name: str
    split: str
    seed: int


HELDOUT_SPECS = {
    "heldout": HeldoutSpec(
        name="heldout",
        split="heldout_candidate_relative_flow_rule_tail_solve_depth2",
        seed=146,
    ),
    "heldout2": HeldoutSpec(
        name="heldout2",
        split="heldout2_candidate_relative_flow_rule_tail_solve_depth2",
        seed=246,
    ),
}


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _tool_call(call_id: str, args: dict[str, Any]) -> dict[str, str]:
    return {
        "id": call_id,
        "name": TOOL_NAME,
        "arguments": _json_dumps(args),
    }


def _tool_defs_json(env: Any) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in env.tool_defs:
        if hasattr(tool, "model_dump"):
            tools.append(tool.model_dump(mode="json"))
        else:
            tools.append(dict(tool))
    return tools


def _base_messages(env: Any, row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = list(row["prompt"])
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": env.system_prompt})
    return messages


def _tool_message(message: Any) -> dict[str, str]:
    return {
        "role": "tool",
        "content": str(getattr(message, "content", "")),
        "tool_call_id": str(getattr(message, "tool_call_id", "")),
        "name": TOOL_NAME,
    }


def parse_qwen_tool_call(completion: str) -> ParsedToolCall:
    text = completion.strip()
    if text.endswith("<|im_end|>"):
        text = text[: -len("<|im_end|>")].strip()
    if "<tool_call>" not in text:
        raise ValueError("missing <tool_call> block")
    if not text.startswith("<tool_call>"):
        raise ValueError("completion contains prose before <tool_call>")
    if text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1:
        raise ValueError("expected exactly one <tool_call> block")
    match = re.fullmatch(
        r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("malformed tool-call function block")
    name = match.group(1)
    body = match.group(2)
    if name != TOOL_NAME:
        raise ValueError(f"expected {TOOL_NAME}, got {name}")
    params = {
        key: value.strip()
        for key, value in re.findall(
            r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
            body,
            flags=re.DOTALL,
        )
    }
    if set(params) != {"index", "direction"}:
        raise ValueError("expected index and direction parameters")
    try:
        index = int(params["index"])
    except ValueError as exc:
        raise ValueError("index parameter must be an integer") from exc
    direction = params["direction"]
    if index < 1 or index > 4:
        raise ValueError("index parameter must be in 1..4")
    if direction not in {"cw", "ccw"}:
        raise ValueError("direction parameter must be cw or ccw")
    return ParsedToolCall(name=name, arguments={"direction": direction, "index": index})


def _expected_args(row: dict[str, Any], rollout: Any, turn: int) -> dict[str, Any]:
    face, direction = rollout.inverse_solution[turn - 1]
    candidate_faces = (
        tuple(row["candidate_faces"])
        if turn == 1
        else tuple(rollout.second_candidate_faces or rollout.candidate_faces)
    )
    if face not in candidate_faces:
        raise ValueError(f"target face {face} is not in candidate faces")
    return {"direction": direction, "index": candidate_faces.index(face) + 1}


ActionProvider = Callable[[int, dict[str, Any], dict[str, Any], list[dict[str, Any]]], dict[str, Any]]


def oracle_action_provider(
    turn: int,
    row: dict[str, Any],
    state: dict[str, Any],
    _messages: list[dict[str, Any]],
) -> dict[str, Any]:
    return _expected_args(row, state["megaminx"], turn)


def constant_bad_action_provider(
    _turn: int,
    _row: dict[str, Any],
    _state: dict[str, Any],
    _messages: list[dict[str, Any]],
) -> dict[str, Any]:
    return {"direction": "cw", "index": 1}


async def evaluate_rows_with_provider(
    env: Any,
    rows: Sequence[dict[str, Any]],
    provider: ActionProvider,
    *,
    output_jsonl: Path | None = None,
) -> dict[str, Any]:
    failure_counts: Counter[str] = Counter()
    row_results: list[dict[str, Any]] = []
    final_rewards: list[float] = []
    solved_count = 0
    first_exact_count = 0
    second_exact_count = 0
    strict_count = 0
    clean_count = 0
    two_call_count = 0
    parse_ok_count = 0

    for row_index, raw_row in enumerate(rows):
        row = dict(raw_row)
        state: dict[str, Any] = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        messages = _base_messages(env, row)
        per_row_failures: list[str] = []
        exact_by_turn: list[bool] = []
        parsed_turns = 0

        for turn in (1, 2):
            try:
                args = provider(turn, row, state, messages)
                expected = _expected_args(row, rollout, turn)
                exact_by_turn.append(args == expected)
                parsed_turns += 1
            except Exception as exc:
                reason = f"provider_error_turn_{turn}:{type(exc).__name__}"
                failure_counts[reason] += 1
                per_row_failures.append(reason)
                break

            tool_call = _tool_call(TOOL_CALL_IDS[turn - 1], args)
            response = await env.env_response(
                [AssistantMessage(content="", tool_calls=[tool_call])],
                state,
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call],
                }
            )
            if response:
                messages.append(_tool_message(response[0]))
            if len(response) != 1:
                reason = f"env_response_count_turn_{turn}:{len(response)}"
                failure_counts[reason] += 1
                per_row_failures.append(reason)
                break

        final_reward = await action_gated_candidate_path_tail_solve_reward(state)
        final_rewards.append(float(final_reward))
        solved = bool(rollout.solved())
        env_clean = (
            rollout.illegal_moves == 0
            and rollout.tool_call_error_count == 0
            and rollout.tool_parse_error_count == 0
            and rollout.protocol_violation_count == 0
        )
        first_exact = bool(exact_by_turn[:1] and exact_by_turn[0])
        second_exact = bool(len(exact_by_turn) > 1 and exact_by_turn[1])
        strict = first_exact and second_exact and solved and env_clean
        two_calls = rollout.candidate_select_call_count == 2

        solved_count += int(solved)
        clean_count += int(env_clean)
        first_exact_count += int(first_exact)
        second_exact_count += int(second_exact)
        strict_count += int(strict)
        two_call_count += int(two_calls)
        parse_ok_count += int(parsed_turns == 2)
        if not solved:
            failure_counts["not_solved"] += 1
        if not env_clean:
            failure_counts["env_not_clean"] += 1
        if not two_calls:
            failure_counts["not_two_select_candidate_calls"] += 1

        row_results.append(
            {
                "row_index": row_index,
                "example_id": row.get("example_id"),
                "solved": solved,
                "final_reward": final_reward,
                "first_exact": first_exact,
                "second_exact": second_exact,
                "env_clean": env_clean,
                "two_select_candidate_calls": two_calls,
                "move_history": rollout.move_history,
                "inverse_solution": rollout.inverse_solution,
                "failures": per_row_failures,
            }
        )

    if output_jsonl is not None:
        output_jsonl.write_text(
            "".join(_json_dumps(result) + "\n" for result in row_results),
            encoding="utf-8",
        )

    row_count = len(rows)
    reward_mean = sum(final_rewards) / row_count if row_count else 0.0
    return {
        "rows": row_count,
        "parse_ok_rate": parse_ok_count / row_count if row_count else 0.0,
        "two_select_candidate_rate": two_call_count / row_count if row_count else 0.0,
        "env_clean_rate": clean_count / row_count if row_count else 0.0,
        "first_exact_rate": first_exact_count / row_count if row_count else 0.0,
        "second_exact_rate": second_exact_count / row_count if row_count else 0.0,
        "inverse_prefix_2_rate": strict_count / row_count if row_count else 0.0,
        "final_reward_mean": reward_mean,
        "solved_rate": solved_count / row_count if row_count else 0.0,
        "strict_two_call_correct_rate": strict_count / row_count if row_count else 0.0,
        "failure_counts": dict(sorted(failure_counts.items())),
    }


def _load_eval_env(spec: HeldoutSpec, *, num_examples: int) -> Any:
    return load_environment(
        split=spec.split,
        min_depth=2,
        max_depth=2,
        num_examples=num_examples,
        seed=spec.seed,
        reward_style=REWARD_STYLE,
        prompt_style=PROMPT_STYLE,
        max_turns=4,
        move_budget=2,
        allow_text_tool_actions=False,
    )


def _resolve_specs(heldout_set: str) -> list[HeldoutSpec]:
    if heldout_set == "both":
        return [HELDOUT_SPECS["heldout"], HELDOUT_SPECS["heldout2"]]
    return [HELDOUT_SPECS[heldout_set]]


def _model_action_provider(
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    tools: list[dict[str, Any]],
    max_new_tokens: int,
) -> ActionProvider:
    def provide(
        _turn: int,
        _row: dict[str, Any],
        _state: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        import torch

        prompt = tokenizer.apply_chat_template(
            _chat_template_messages({"messages": messages, "tools": tools}),
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion_ids = output[0][encoded["input_ids"].shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
        parsed = parse_qwen_tool_call(completion)
        return parsed.arguments

    return provide


def _load_model_provider(args: argparse.Namespace, tools: list[dict[str, Any]]) -> ActionProvider:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_files_only = not args.allow_downloads
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = _training_device(torch)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if args.adapter_dir:
        model = PeftModel.from_pretrained(
            model,
            args.adapter_dir,
            local_files_only=local_files_only,
        )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return _model_action_provider(
        tokenizer=tokenizer,
        model=model,
        device=device,
        tools=tools,
        max_new_tokens=args.max_new_tokens,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline heldout evaluator for local Megaminx SFT LoRA adapters."
    )
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--heldout-set",
        choices=["heldout", "heldout2", "both"],
        default="both",
    )
    parser.add_argument("--num-examples", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--bad-baseline", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    return parser.parse_args(argv)


async def _run_async(args: argparse.Namespace) -> dict[str, Any]:
    specs = _resolve_specs(args.heldout_set)
    payload: dict[str, Any] = {
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "heldout_set": args.heldout_set,
        "num_examples": args.num_examples,
        "prompt_style": PROMPT_STYLE,
        "reward_style": REWARD_STYLE,
        "splits": {},
    }
    if args.dry_run:
        for spec in specs:
            env = _load_eval_env(spec, num_examples=args.num_examples)
            rows = [dict(row) for row in env.get_dataset()]
            payload["splits"][spec.name] = {
                "split": spec.split,
                "seed": spec.seed,
                "rows": len(rows),
                "tool_names": [tool["name"] for tool in _tool_defs_json(env)],
                "first_example_id": rows[0]["example_id"] if rows else None,
            }
        return payload

    provider_by_split: dict[str, ActionProvider] = {}
    for spec in specs:
        env = _load_eval_env(spec, num_examples=args.num_examples)
        rows = [dict(row) for row in env.get_dataset()]
        if args.oracle:
            provider = oracle_action_provider
        elif args.bad_baseline:
            provider = constant_bad_action_provider
        else:
            if not provider_by_split:
                provider_by_split[spec.name] = _load_model_provider(
                    args,
                    _tool_defs_json(env),
                )
            provider = next(iter(provider_by_split.values()))
        split_output = None
        if args.output_jsonl:
            suffix = f".{spec.name}.jsonl" if args.heldout_set == "both" else ".jsonl"
            split_output = args.output_jsonl.with_suffix(suffix)
        payload["splits"][spec.name] = await evaluate_rows_with_provider(
            env,
            rows,
            provider,
            output_jsonl=split_output,
        )
        payload["splits"][spec.name]["split"] = spec.split
        payload["splits"][spec.name]["seed"] = spec.seed
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    _disable_progress_bars()
    args = parse_args(argv)
    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive")
    if sum(bool(value) for value in (args.dry_run, args.oracle, args.bad_baseline)) > 1:
        raise ValueError("choose at most one of --dry-run, --oracle, or --bad-baseline")
    payload = asyncio.run(_run_async(args))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
