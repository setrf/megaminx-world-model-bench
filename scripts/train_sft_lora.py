#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from summarize_oracle_trajectories import EXPECTED_ENV_VERSION, _read_jsonl  # noqa: E402
from validate_sft_jsonl import validate_sft_jsonl  # noqa: E402


REQUIRED_PACKAGES = ("torch", "transformers", "peft")


@dataclass(frozen=True)
class SFTConfig:
    model: str
    train_jsonl: Path
    output_dir: Path
    max_samples: int
    max_steps: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    max_seq_length: int
    seed: int
    format_style: str
    gradient_checkpointing: bool


def _require_packages() -> None:
    missing = _missing_packages()
    if missing:
        raise RuntimeError(
            "Missing SFT dependencies: "
            + ", ".join(missing)
            + ". Install an ML extra such as torch, transformers, and peft on a GPU machine before running."
        )


def _require_tokenizer_package() -> None:
    try:
        __import__("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "Missing tokenizer dependency: transformers. Install transformers or use "
            'format_style = "plain_markers" for a dependency-light dry run.'
        ) from exc


def _load_config(path: Path) -> SFTConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    config = SFTConfig(
        model=str(data["model"]),
        train_jsonl=Path(data["train_jsonl"]),
        output_dir=Path(data["output_dir"]),
        max_samples=int(data.get("max_samples", 0)),
        max_steps=int(data["max_steps"]),
        learning_rate=float(data["learning_rate"]),
        batch_size=int(data["batch_size"]),
        gradient_accumulation_steps=int(data["gradient_accumulation_steps"]),
        lora_rank=int(data["lora_rank"]),
        lora_alpha=int(data["lora_alpha"]),
        lora_dropout=float(data["lora_dropout"]),
        max_seq_length=int(data["max_seq_length"]),
        seed=int(data.get("seed", 64)),
        format_style=str(data.get("format_style", "chat_template")),
        gradient_checkpointing=bool(data.get("gradient_checkpointing", True)),
    )
    _validate_config(config)
    return config


def _validate_config(config: SFTConfig) -> None:
    checks = {
        "max_steps": config.max_steps,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "max_seq_length": config.max_seq_length,
    }
    invalid = [name for name, value in checks.items() if value <= 0]
    if config.learning_rate <= 0:
        invalid.append("learning_rate")
    if config.lora_dropout < 0:
        invalid.append("lora_dropout")
    if config.max_samples < 0:
        invalid.append("max_samples")
    if config.format_style not in {"chat_template", "plain_markers"}:
        invalid.append("format_style")
    if invalid:
        raise ValueError("invalid SFT config values: " + ", ".join(invalid))


def _load_training_records(path: Path, *, max_samples: int) -> list[dict[str, Any]]:
    validate_sft_jsonl(path, expected_env_version=EXPECTED_ENV_VERSION)
    records = _read_jsonl(path)
    return records[:max_samples] if max_samples > 0 else records


def _format_tool_block(tools: list[dict[str, Any]]) -> str:
    return json.dumps(tools, sort_keys=True, separators=(",", ":"))


def _format_message(message: dict[str, Any]) -> str:
    role = message["role"]
    if role == "assistant" and message.get("tool_calls"):
        tool_calls = json.dumps(message["tool_calls"], sort_keys=True, separators=(",", ":"))
        return f"<|assistant_tool_call|>{tool_calls}<|end|>"
    if role == "tool":
        payload = {
            "tool_call_id": message["tool_call_id"],
            "name": message.get("name"),
            "content": message.get("content", ""),
        }
        return f"<|tool|>{json.dumps(payload, sort_keys=True, separators=(',', ':'))}<|end|>"
    return f"<|{role}|>{message.get('content', '')}<|end|>"


def _format_training_text(record: dict[str, Any]) -> str:
    parts = ["<|tools|>" + _format_tool_block(record["tools"]) + "<|end|>"]
    parts.extend(_format_message(message) for message in record["messages"])
    return "\n".join(parts)


def _build_dataset(records: list[dict[str, Any]]) -> list[str]:
    return [_format_training_text(record) for record in records]


def _build_chat_template_dataset(records: list[dict[str, Any]], tokenizer: Any) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            _chat_template_messages(record),
            tools=record["tools"],
            tokenize=False,
            add_generation_prompt=False,
        )
        for record in records
    ]


def _chat_template_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    tool_names = {tool["name"] for tool in record["tools"]}
    default_tool_name = next(iter(tool_names)) if len(tool_names) == 1 else None
    for message in record["messages"]:
        item = dict(message)
        if item["role"] == "assistant" and item.get("tool_calls"):
            calls = []
            for call in item["tool_calls"]:
                calls.append(
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.loads(call["arguments"]),
                        },
                    }
                )
            item["tool_calls"] = calls
        if item["role"] == "tool" and "name" not in item and default_tool_name:
            item["name"] = default_tool_name
        messages.append(item)
    return messages


def _target_char_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    marker = "<|assistant_tool_call|>"
    end_marker = "<|end|>"
    start = 0
    while True:
        marker_start = text.find(marker, start)
        if marker_start < 0:
            break
        end = text.find(end_marker, marker_start + len(marker))
        if end < 0:
            break
        spans.append((marker_start, end + len(end_marker)))
        start = end + len(end_marker)
    return spans


def _assistant_char_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    marker = "<|im_start|>assistant"
    end_marker = "<|im_end|>"
    start = 0
    while True:
        marker_start = text.find(marker, start)
        if marker_start < 0:
            break
        content_start = text.find("\n", marker_start)
        if content_start < 0:
            break
        content_start += 1
        end = text.find(end_marker, content_start)
        if end < 0:
            break
        spans.append((content_start, end + len(end_marker)))
        start = end + len(end_marker)
    return spans


def _token_in_spans(
    token_start: int,
    token_end: int,
    spans: Sequence[tuple[int, int]],
) -> bool:
    return token_start != token_end and any(
        token_start < span_end and token_end > span_start
        for span_start, span_end in spans
    )


def _missing_packages() -> list[str]:
    missing: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    return missing


def _training_device(torch_module: Any) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if (
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def _train(config: SFTConfig, *, dry_run: bool) -> dict[str, Any]:
    records = _load_training_records(config.train_jsonl, max_samples=config.max_samples)
    tokenizer = None
    if config.format_style == "chat_template":
        _require_tokenizer_package()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
        texts = _build_chat_template_dataset(records, tokenizer)
    else:
        texts = _build_dataset(records)
    if not texts:
        raise ValueError("no training records found")

    preview = {
        "model": config.model,
        "train_jsonl": str(config.train_jsonl),
        "output_dir": str(config.output_dir),
        "records": len(records),
        "max_optimizer_steps": config.max_steps,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_seq_length": config.max_seq_length,
        "format_style": config.format_style,
        "gradient_checkpointing": config.gradient_checkpointing,
        "first_text_chars": len(texts[0]),
    }
    if dry_run:
        return {"dry_run": True, **preview, "missing_packages": _missing_packages()}

    _require_packages()
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = _training_device(torch)
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model.train()

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.max_seq_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    for row_index, text in enumerate(texts):
        spans = (
            _assistant_char_spans(text)
            if config.format_style == "chat_template"
            else _target_char_spans(text)
        )
        offsets = encoded["offset_mapping"][row_index]
        for token_index, (token_start, token_end) in enumerate(offsets.tolist()):
            if not _token_in_spans(token_start, token_end, spans):
                labels[row_index, token_index] = -100
        if torch.all(labels[row_index] == -100):
            raise ValueError("training record has no assistant tool-call target tokens")
    del encoded["offset_mapping"]
    dataset = [
        {
            "input_ids": encoded["input_ids"][index],
            "attention_mask": encoded["attention_mask"][index],
            "labels": labels[index],
        }
        for index in range(len(texts))
    ]
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    parameter_device = next(model.parameters()).device

    micro_step = 0
    optimizer_step = 0
    losses: list[float] = []
    while optimizer_step < config.max_steps:
        for batch in loader:
            batch = {key: value.to(parameter_device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss / config.gradient_accumulation_steps
            loss.backward()
            micro_step += 1
            if micro_step % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            losses.append(float(loss.detach().cpu()) * config.gradient_accumulation_steps)
            if optimizer_step >= config.max_steps:
                break

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return {
        "dry_run": False,
        **preview,
        "micro_steps": micro_step,
        "optimizer_steps": optimizer_step,
        "device": str(parameter_device),
        "final_loss": losses[-1] if losses else None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a local LoRA SFT warm-start adapter from Megaminx oracle trajectories."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/data and print the training plan without importing ML packages.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_config(args.config)
    overrides = {
        key: value
        for key, value in {
            "train_jsonl": args.train_jsonl,
            "output_dir": args.output_dir,
            "max_samples": args.max_samples,
            "max_steps": args.max_steps,
            "max_seq_length": args.max_seq_length,
        }.items()
        if value is not None
    }
    if overrides:
        config = SFTConfig(**{**config.__dict__, **overrides})
        _validate_config(config)
    try:
        result = _train(config, dry_run=args.dry_run)
    except RuntimeError as exc:
        message = str(exc)
        if "MPS backend out of memory" in message:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "mps_out_of_memory",
                        "hint": (
                            "Reduce --max-seq-length/--max-samples, keep "
                            "gradient_checkpointing enabled, or run on a CUDA GPU."
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
