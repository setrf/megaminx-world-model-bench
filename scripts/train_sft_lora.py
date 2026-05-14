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


def _require_packages() -> None:
    missing = _missing_packages()
    if missing:
        raise RuntimeError(
            "Missing SFT dependencies: "
            + ", ".join(missing)
            + ". Install an ML extra such as torch, transformers, and peft on a GPU machine before running."
        )


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
    if invalid:
        raise ValueError("invalid non-positive SFT config values: " + ", ".join(invalid))


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


def _missing_packages() -> list[str]:
    missing: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    return missing


def _train(config: SFTConfig, *, dry_run: bool) -> dict[str, Any]:
    records = _load_training_records(config.train_jsonl, max_samples=config.max_samples)
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

    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
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
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
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
    device = next(model.parameters()).device

    micro_step = 0
    optimizer_step = 0
    losses: list[float] = []
    while optimizer_step < config.max_steps:
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
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
        "final_loss": losses[-1] if losses else None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a local LoRA SFT warm-start adapter from Megaminx oracle trajectories."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/data and print the training plan without importing ML packages.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_config(args.config)
    if args.train_jsonl:
        config = SFTConfig(**{**config.__dict__, "train_jsonl": args.train_jsonl})
    if args.output_dir:
        config = SFTConfig(**{**config.__dict__, "output_dir": args.output_dir})
    result = _train(config, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
