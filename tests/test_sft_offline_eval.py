from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_sft_lora_offline


def test_parse_qwen_tool_call_accepts_native_candidate_call() -> None:
    parsed = eval_sft_lora_offline.parse_qwen_tool_call(
        """
        <tool_call>
        <function=select_candidate>
        <parameter=direction>
        ccw
        </parameter>
        <parameter=index>
        4
        </parameter>
        </function>
        </tool_call><|im_end|>
        """
    )

    assert parsed.name == "select_candidate"
    assert parsed.arguments == {"direction": "ccw", "index": 4}


def test_parse_qwen_tool_call_rejects_unsafe_shapes() -> None:
    bad_completions = [
        "I will solve it.\n<tool_call></tool_call>",
        "<tool_call></tool_call><tool_call></tool_call>",
        "<tool_call><function=rotate></function></tool_call>",
        "<tool_call><function=select_candidate><parameter=index>5</parameter><parameter=direction>cw</parameter></function></tool_call>",
        "<tool_call><function=select_candidate><parameter=index>1</parameter><parameter=direction>left</parameter></function></tool_call>",
    ]

    for completion in bad_completions:
        try:
            eval_sft_lora_offline.parse_qwen_tool_call(completion)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected parser rejection for {completion!r}")


def test_offline_eval_dry_run_builds_both_heldout_sets() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "eval_sft_lora_offline.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--num-examples",
            "2",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["num_examples"] == 2
    assert set(payload["splits"]) == {"heldout", "heldout2"}
    assert payload["splits"]["heldout"]["seed"] == 146
    assert payload["splits"]["heldout2"]["seed"] == 246
    assert payload["splits"]["heldout"]["rows"] == 2
    assert payload["splits"]["heldout"]["tool_names"] == ["select_candidate"]


def test_offline_eval_oracle_provider_solves_heldout_rows() -> None:
    async def run() -> None:
        spec = eval_sft_lora_offline.HELDOUT_SPECS["heldout"]
        env = eval_sft_lora_offline._load_eval_env(spec, num_examples=2)
        rows = [dict(row) for row in env.get_dataset()]

        summary = await eval_sft_lora_offline.evaluate_rows_with_provider(
            env,
            rows,
            eval_sft_lora_offline.oracle_action_provider,
        )

        assert summary["rows"] == 2
        assert summary["parse_ok_rate"] == 1.0
        assert summary["two_select_candidate_rate"] == 1.0
        assert summary["env_clean_rate"] == 1.0
        assert summary["first_exact_rate"] == 1.0
        assert summary["second_exact_rate"] == 1.0
        assert summary["final_reward_mean"] == 1.0
        assert summary["solved_rate"] == 1.0
        assert summary["strict_two_call_correct_rate"] == 1.0

    asyncio.run(run())


def test_offline_eval_bad_provider_reports_failure_without_crashing() -> None:
    async def run() -> None:
        spec = eval_sft_lora_offline.HELDOUT_SPECS["heldout"]
        env = eval_sft_lora_offline._load_eval_env(spec, num_examples=2)
        rows = [dict(row) for row in env.get_dataset()]

        summary = await eval_sft_lora_offline.evaluate_rows_with_provider(
            env,
            rows,
            eval_sft_lora_offline.constant_bad_action_provider,
        )

        assert summary["rows"] == 2
        assert summary["strict_two_call_correct_rate"] < 1.0
        assert summary["solved_rate"] < 1.0
        assert summary["failure_counts"]["not_solved"] >= 1

    asyncio.run(run())
