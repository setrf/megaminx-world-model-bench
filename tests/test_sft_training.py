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
    assert payload["max_seq_length"] == 2048
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


def test_sft_chat_template_conversion_parses_tool_arguments(tmp_path: Path) -> None:
    sft_output = _build_sft_jsonl(tmp_path)
    record = json.loads(sft_output.read_text(encoding="utf-8").splitlines()[0])

    messages = train_sft_lora._chat_template_messages(record)

    assistant_messages = [
        message for message in messages if message["role"] == "assistant"
    ]
    tool_messages = [message for message in messages if message["role"] == "tool"]
    assert assistant_messages[0]["tool_calls"][0]["type"] == "function"
    assert assistant_messages[0]["tool_calls"][0]["function"]["name"] == "select_candidate"
    assert assistant_messages[0]["tool_calls"][0]["function"]["arguments"] == json.loads(
        record["messages"][2]["tool_calls"][0]["arguments"]
    )
    assert [message["name"] for message in tool_messages] == [
        "select_candidate",
        "select_candidate",
    ]


def test_sft_target_char_spans_mask_only_assistant_tool_calls() -> None:
    text = "\n".join(
        [
            "<|system|>Do not train me.<|end|>",
            "<|assistant_tool_call|>[{\"name\":\"select_candidate\"}]<|end|>",
            "<|tool|>{\"content\":\"observation\"}<|end|>",
            "<|assistant_tool_call|>[{\"name\":\"select_candidate\",\"arguments\":\"{}\"}]<|end|>",
        ]
    )

    spans = train_sft_lora._target_char_spans(text)

    assert len(spans) == 2
    covered = "".join(text[start:end] for start, end in spans)
    assert "Do not train me" not in covered
    assert "observation" not in covered
    assert covered.count("<|assistant_tool_call|>") == 2
    assert train_sft_lora._token_in_spans(
        text.index("select_candidate"),
        text.index("select_candidate") + len("select_candidate"),
        spans,
    )
    assert not train_sft_lora._token_in_spans(
        text.index("observation"),
        text.index("observation") + len("observation"),
        spans,
    )


def test_sft_assistant_char_spans_ignore_system_tool_examples() -> None:
    text = "\n".join(
        [
            "<|im_start|>system",
            "<tool_call>",
            "<function=example_function_name>",
            "</function>",
            "</tool_call>",
            "<|im_end|>",
            "<|im_start|>assistant",
            "<think>",
            "",
            "</think>",
            "",
            "<tool_call>",
            "<function=select_candidate>",
            "<parameter=index>",
            "3",
            "</parameter>",
            "</function>",
            "</tool_call><|im_end|>",
        ]
    )

    spans = train_sft_lora._assistant_char_spans(text)

    assert len(spans) == 1
    covered = text[spans[0][0] : spans[0][1]]
    assert "example_function_name" not in covered
    assert "select_candidate" in covered
    assert covered.endswith("<|im_end|>")
