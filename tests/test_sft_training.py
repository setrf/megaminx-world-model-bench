from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_sft_lora

from test_oracle_export import _run_export


def _build_sft_jsonl(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    convert_script = repo_root / "scripts" / "convert_oracle_to_sft_jsonl.py"
    oracle_output = tmp_path / "oracle.jsonl"
    sft_output = tmp_path / "sft.jsonl"
    _run_export(oracle_output)
    subprocess.run(
        [
            sys.executable,
            str(convert_script),
            str(oracle_output),
            "--output",
            str(sft_output),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return sft_output


def test_sft_lora_dry_run_validates_data_without_ml_dependencies(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "train_sft_lora.py"
    config = repo_root / "configs" / "sft" / "megaminx-v056-qwen08b-tail-solve-smoke.toml"
    sft_output = _build_sft_jsonl(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--train-jsonl",
            str(sft_output),
            "--output-dir",
            str(tmp_path / "adapter"),
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["dry_run"] is True
    assert payload["model"] == "Qwen/Qwen3.5-0.8B"
    assert payload["records"] == 2
    assert payload["max_optimizer_steps"] == 10
    assert payload["gradient_accumulation_steps"] == 4
    assert payload["max_seq_length"] == 4096
    assert payload["output_dir"] == str(tmp_path / "adapter")
    assert isinstance(payload["missing_packages"], list)


def test_sft_training_text_keeps_tool_calls_as_supervised_targets(tmp_path: Path) -> None:
    sft_output = _build_sft_jsonl(tmp_path)
    record = json.loads(sft_output.read_text(encoding="utf-8").splitlines()[0])

    text = train_sft_lora._format_training_text(record)

    assert text.startswith("<|tools|>")
    assert text.count("<|assistant_tool_call|>") == 2
    assert text.count("<|tool|>") == 2
    assert "call_1" in text
    assert "call_2" in text
    assert "select_candidate" in text
    assert "oracle-" not in text
