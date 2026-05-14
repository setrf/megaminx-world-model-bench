from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_export(output: Path) -> list[dict]:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "export_oracle_trajectories.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--num-examples",
            "2",
            "--seed",
            "64",
            "--split",
            "train_candidate_relative_flow_rule_tail_solve_depth2",
            "--output",
            str(output),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]


def test_export_oracle_trajectories_cli_writes_deterministic_jsonl(tmp_path: Path) -> None:
    first_output = tmp_path / "oracle-first.jsonl"
    second_output = tmp_path / "oracle-second.jsonl"

    first_records = _run_export(first_output)
    second_records = _run_export(second_output)

    assert first_records == second_records
    assert len(first_records) == 2
    for record in first_records:
        assert record["expected_env_version"] == "0.2.56"
        assert record["scramble_depth"] == 2
        assert record["move_budget"] == 2
        assert record["reward_style"] == "action_gated_candidate_path_tail_solve"
        assert record["prompt_style"] == "stage_candidate_relative_flow_rule_solve2_native_tool"
        assert [tool["name"] for tool in record["tools"]] == ["select_candidate"]

        assert len(record["actions"]) == 2
        first_action, second_action = record["actions"]
        assert first_action["tool"] == "select_candidate"
        assert second_action["tool"] == "select_candidate"
        assert first_action["face"] == record["inverse_solution"][0]["face"]
        assert second_action["face"] == record["inverse_solution"][1]["face"]
        assert (
            record["candidate_faces_initial"][first_action["arguments"]["index"] - 1]
            == first_action["face"]
        )
        assert (
            record["candidate_faces_after_first"][second_action["arguments"]["index"] - 1]
            == second_action["face"]
        )

        assert [message["role"] for message in record["messages"]] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
        ]
        assert "Updated candidate relative-flow view" in record["messages"][3]["content"]
        assert record["metrics"]["first_step_reward"] == 0.25
        assert record["metrics"]["final_reward"] == 1.0
        assert record["metrics"]["solved"] is True
        assert record["metrics"]["candidate_path_completed"] == 1.0


def test_summarize_oracle_trajectories_cli_validates_export(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary_script = repo_root / "scripts" / "summarize_oracle_trajectories.py"
    output = tmp_path / "oracle.jsonl"
    records = _run_export(output)

    result = subprocess.run(
        [sys.executable, str(summary_script), str(output)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(result.stdout)

    assert summary["rows"] == len(records) == 2
    assert summary["env_versions"] == {"0.2.56": 2}
    assert summary["action_counts"] == {"2": 2}
    assert summary["final_rewards"] == {"1.0": 2}
    assert summary["solved"] == {"True": 2}
    assert summary["prompt_leak_count"] == 0
    assert set(summary["first_slot_counts"]).issubset({"1", "2", "3", "4"})
    assert set(summary["second_slot_counts"]).issubset({"1", "2", "3", "4"})
