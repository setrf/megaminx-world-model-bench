from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "rl"


def _load_config(name: str) -> dict:
    return tomllib.loads((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_v056_matched_probe_configs_cover_base_and_checkpoint_seeds() -> None:
    expected = {
        "megaminx-v056-qwen9b-tail-solve-depth2-base-heldout-rpe16.toml": {
            "checkpoint_id": None,
            "buffer_seed": 157,
            "env_seed": 146,
            "split": "heldout_candidate_relative_flow_rule_tail_solve_depth2",
        },
        "megaminx-v056-qwen9b-tail-solve-depth2-base-heldout2-rpe16.toml": {
            "checkpoint_id": None,
            "buffer_seed": 257,
            "env_seed": 246,
            "split": "heldout2_candidate_relative_flow_rule_tail_solve_depth2",
        },
        "megaminx-v056-qwen9b-tail-solve-depth2-ckpt2-heldout-rpe16.toml": {
            "checkpoint_id": "o68kzy5up4e65ve6lktmkuat",
            "buffer_seed": 157,
            "env_seed": 146,
            "split": "heldout_candidate_relative_flow_rule_tail_solve_depth2",
        },
        "megaminx-v056-qwen9b-tail-solve-depth2-ckpt2-heldout2-rpe16.toml": {
            "checkpoint_id": "o68kzy5up4e65ve6lktmkuat",
            "buffer_seed": 257,
            "env_seed": 246,
            "split": "heldout2_candidate_relative_flow_rule_tail_solve_depth2",
        },
    }

    for name, spec in expected.items():
        config = _load_config(name)
        env = config["env"][0]
        args = env["args"]

        assert config["model"] == "Qwen/Qwen3.5-9B"
        assert config["batch_size"] == 512
        assert config["rollouts_per_example"] == 16
        if spec["checkpoint_id"]:
            assert config["checkpoint_id"] == spec["checkpoint_id"]
        else:
            assert "checkpoint_id" not in config

        assert config["buffer"]["seed"] == spec["buffer_seed"]
        assert env["id"] == "setrf/megaminx-solver"
        assert env["version"] == "0.2.56"
        assert args["split"] == spec["split"]
        assert args["seed"] == spec["env_seed"]
        assert args["min_depth"] == 2
        assert args["max_depth"] == 2
        assert args["num_examples"] == 1024
        assert args["max_turns"] == 4
        assert args["move_budget"] == 2
        assert args["reward_style"] == "action_gated_candidate_path_tail_solve"
        assert args["prompt_style"] == "stage_candidate_relative_flow_rule_solve2_native_tool"
        assert args["allow_text_tool_actions"] is False
