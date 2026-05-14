#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
CONFIG_ROOT = REPO_ROOT / "configs" / "rl"

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from summarize_oracle_trajectories import (  # noqa: E402
    EXPECTED_ENV_VERSION,
    summarize_oracle_trajectories,
)
from validate_sft_jsonl import validate_sft_jsonl  # noqa: E402


DEFAULT_ORACLE_JSONL = Path("/tmp/megaminx-oracle-v056-1024.jsonl")
DEFAULT_SFT_JSONL = Path("/tmp/megaminx-oracle-v056-1024-sft.jsonl")
EXPECTED_ORACLE_SHA256 = "1038afa6958030832c028840dafc22fc3724206461608e2ed809c90fa9695e7b"
EXPECTED_SFT_SHA256 = "59b9db129517f3a6f86a868f06179826a032b2e0d07c4393d5a9ae168e8b1ec3"
EXPECTED_CANONICAL_ROWS = 1024
EXPECTED_HUB_HASH = "8a1d0168b96c"
EXPECTED_HUB_ACTION_STATUS = "SUCCESS"

EXPECTED_V056_CONFIGS: dict[str, dict[str, Any]] = {
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


def _check(status: str, name: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": status, "details": details or {}}


def _passed(name: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return _check("passed", name, details)


def _failed(name: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return _check("failed", name, details)


def _skipped(name: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return _check("skipped", name, details)


def _redact(value: str) -> str:
    value = re.sub(r"(api[_-]?key|token|authorization)\s*[:=]\s*\S+", r"\1=<redacted>", value, flags=re.I)
    value = re.sub(r"Bearer\s+\S+", "Bearer <redacted>", value, flags=re.I)
    return value.strip()


def _first_lines(value: str, *, limit: int = 4) -> list[str]:
    lines = [_redact(line)[:240] for line in value.splitlines() if line.strip()]
    return lines[:limit]


def _assert_equal(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def check_v056_configs() -> dict[str, Any]:
    errors: list[str] = []
    summaries: dict[str, dict[str, Any]] = {}

    for name, spec in EXPECTED_V056_CONFIGS.items():
        path = CONFIG_ROOT / name
        if not path.exists():
            errors.append(f"{name}: missing config")
            continue

        try:
            config = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            errors.append(f"{name}: TOML parse failed: {exc}")
            continue

        env = config.get("env", [{}])[0]
        args = env.get("args", {})

        _assert_equal(errors, f"{name} model", config.get("model"), "Qwen/Qwen3.5-9B")
        _assert_equal(errors, f"{name} batch_size", config.get("batch_size"), 512)
        _assert_equal(errors, f"{name} rollouts_per_example", config.get("rollouts_per_example"), 16)
        if spec["checkpoint_id"]:
            _assert_equal(errors, f"{name} checkpoint_id", config.get("checkpoint_id"), spec["checkpoint_id"])
        elif "checkpoint_id" in config:
            errors.append(f"{name}: base config unexpectedly has checkpoint_id={config['checkpoint_id']!r}")

        _assert_equal(errors, f"{name} buffer seed", config.get("buffer", {}).get("seed"), spec["buffer_seed"])
        _assert_equal(errors, f"{name} env id", env.get("id"), "setrf/megaminx-solver")
        _assert_equal(errors, f"{name} env version", env.get("version"), EXPECTED_ENV_VERSION)
        _assert_equal(errors, f"{name} split", args.get("split"), spec["split"])
        _assert_equal(errors, f"{name} seed", args.get("seed"), spec["env_seed"])
        _assert_equal(errors, f"{name} min_depth", args.get("min_depth"), 2)
        _assert_equal(errors, f"{name} max_depth", args.get("max_depth"), 2)
        _assert_equal(errors, f"{name} num_examples", args.get("num_examples"), EXPECTED_CANONICAL_ROWS)
        _assert_equal(errors, f"{name} max_turns", args.get("max_turns"), 4)
        _assert_equal(errors, f"{name} move_budget", args.get("move_budget"), 2)
        _assert_equal(
            errors,
            f"{name} reward_style",
            args.get("reward_style"),
            "action_gated_candidate_path_tail_solve",
        )
        _assert_equal(
            errors,
            f"{name} prompt_style",
            args.get("prompt_style"),
            "stage_candidate_relative_flow_rule_solve2_native_tool",
        )
        _assert_equal(errors, f"{name} allow_text_tool_actions", args.get("allow_text_tool_actions"), False)

        summaries[name] = {
            "checkpoint_id": config.get("checkpoint_id"),
            "buffer_seed": config.get("buffer", {}).get("seed"),
            "env_seed": args.get("seed"),
            "split": args.get("split"),
        }

    if errors:
        return _failed("v0.2.56 matched probe configs", {"errors": errors})
    return _passed("v0.2.56 matched probe configs", {"configs": summaries})


def _canonical_artifact_errors(summary: dict[str, Any], expected_sha256: str) -> list[str]:
    errors: list[str] = []
    if summary.get("rows") != EXPECTED_CANONICAL_ROWS:
        errors.append(f"rows: expected {EXPECTED_CANONICAL_ROWS}, got {summary.get('rows')!r}")
    if summary.get("sha256") != expected_sha256:
        errors.append(f"sha256: expected {expected_sha256}, got {summary.get('sha256')!r}")
    return errors


def check_oracle_artifact(path: Path, *, require: bool, allow_noncanonical: bool) -> dict[str, Any]:
    if not path.exists():
        details = {
            "path": str(path),
            "regenerate": (
                "uv run python scripts/export_oracle_trajectories.py --num-examples 1024 "
                "--seed 64 --split train_candidate_relative_flow_rule_tail_solve_depth2 "
                f"--output {path}"
            ),
        }
        return _failed("oracle JSONL", details) if require else _skipped("oracle JSONL", details)

    try:
        summary = summarize_oracle_trajectories(path)
    except Exception as exc:
        return _failed("oracle JSONL", {"path": str(path), "error": str(exc)})

    errors = [] if allow_noncanonical else _canonical_artifact_errors(summary, EXPECTED_ORACLE_SHA256)
    details = {
        "path": summary["path"],
        "rows": summary["rows"],
        "bytes": summary["bytes"],
        "sha256": summary["sha256"],
        "env_versions": summary["env_versions"],
        "final_rewards": summary["final_rewards"],
        "solved": summary["solved"],
        "prompt_leak_count": summary["prompt_leak_count"],
    }
    if errors:
        details["errors"] = errors
        return _failed("oracle JSONL", details)
    return _passed("oracle JSONL", details)


def check_sft_artifact(path: Path, *, require: bool, allow_noncanonical: bool) -> dict[str, Any]:
    if not path.exists():
        details = {
            "path": str(path),
            "regenerate": (
                "uv run python scripts/convert_oracle_to_sft_jsonl.py "
                f"{DEFAULT_ORACLE_JSONL} --output {path}"
            ),
        }
        return _failed("SFT JSONL", details) if require else _skipped("SFT JSONL", details)

    try:
        summary = validate_sft_jsonl(path)
    except Exception as exc:
        return _failed("SFT JSONL", {"path": str(path), "error": str(exc)})

    errors = [] if allow_noncanonical else _canonical_artifact_errors(summary, EXPECTED_SFT_SHA256)
    details = {
        "path": summary["path"],
        "rows": summary["rows"],
        "bytes": summary["bytes"],
        "sha256": summary["sha256"],
        "env_versions": summary["env_versions"],
        "tool_names": summary["tool_names"],
        "prompt_leak_count": summary["prompt_leak_count"],
        "forbidden_payload_fields": summary["forbidden_payload_fields"],
    }
    if errors:
        details["errors"] = errors
        return _failed("SFT JSONL", details)
    return _passed("SFT JSONL", details)


def _run_prime_command(args: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return {"ok": False, "error": "prime CLI not found"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout}s"}

    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": _first_lines(result.stdout),
        "stderr": _first_lines(result.stderr),
    }


def _wallet_balance(details: dict[str, Any]) -> float | None:
    text = _combined_output(details)
    match = re.search(r"\$\s*(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1))


def _combined_output(details: dict[str, Any]) -> str:
    return "\n".join(details.get("stdout", []) + details.get("stderr", []))


def check_prime_access(timeout: int) -> dict[str, Any]:
    checks = {
        "whoami": _run_prime_command(["prime", "whoami", "--plain"], timeout=timeout),
        "wallet": _run_prime_command(["prime", "wallet", "--plain"], timeout=timeout),
        "env_status": _run_prime_command(
            ["prime", "env", "status", "setrf/megaminx-solver", "--plain"],
            timeout=timeout,
        ),
    }

    errors: list[str] = []
    for name, details in checks.items():
        if not details.get("ok"):
            errors.append(f"{name}: command failed")

    balance = _wallet_balance(checks["wallet"])
    if checks["wallet"].get("ok") and balance is not None and balance <= 0:
        errors.append(f"wallet: balance must be positive for hosted run creation, got ${balance:.2f}")

    env_status_output = _combined_output(checks["env_status"])
    if checks["env_status"].get("ok"):
        for expected in (EXPECTED_ENV_VERSION, EXPECTED_HUB_HASH, EXPECTED_HUB_ACTION_STATUS):
            if expected not in env_status_output:
                errors.append(f"env_status: expected {expected!r} in output")

    details: dict[str, Any] = {"commands": checks, "wallet_balance_usd": balance}
    if errors:
        details["errors"] = errors
        return _failed("Prime access and wallet", details)
    return _passed("Prime access and wallet", details)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the Megaminx v0.2.56 next-run path is ready."
    )
    parser.add_argument("--oracle-jsonl", type=Path, default=DEFAULT_ORACLE_JSONL)
    parser.add_argument("--sft-jsonl", type=Path, default=DEFAULT_SFT_JSONL)
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument(
        "--allow-noncanonical-artifacts",
        action="store_true",
        help="Validate artifact structure without enforcing the canonical 1024-row SHA256 values.",
    )
    parser.add_argument(
        "--check-prime",
        action="store_true",
        help="Also run prime whoami/wallet/env status. Off by default to avoid auth side effects.",
    )
    parser.add_argument("--prime-timeout", type=int, default=20)
    return parser.parse_args(argv)


def readiness_payload(args: argparse.Namespace) -> dict[str, Any]:
    checks = [check_v056_configs()]

    if args.skip_oracle:
        checks.append(_skipped("oracle JSONL", {"reason": "--skip-oracle"}))
    else:
        checks.append(
            check_oracle_artifact(
                args.oracle_jsonl,
                require=args.require_artifacts,
                allow_noncanonical=args.allow_noncanonical_artifacts,
            )
        )

    if args.skip_sft:
        checks.append(_skipped("SFT JSONL", {"reason": "--skip-sft"}))
    else:
        checks.append(
            check_sft_artifact(
                args.sft_jsonl,
                require=args.require_artifacts,
                allow_noncanonical=args.allow_noncanonical_artifacts,
            )
        )

    if args.check_prime:
        checks.append(check_prime_access(args.prime_timeout))
    else:
        checks.append(
            _skipped(
                "Prime access and wallet",
                {"reason": "run with --check-prime after refreshing auth/billing"},
            )
        )

    failures = [check for check in checks if check["status"] == "failed"]
    local_failures = [
        check
        for check in checks
        if check["status"] == "failed" and check["name"] != "Prime access and wallet"
    ]
    artifact_statuses = {
        check["name"]: check["status"]
        for check in checks
        if check["name"] in {"oracle JSONL", "SFT JSONL"}
    }
    prime_status = next(check["status"] for check in checks if check["name"] == "Prime access and wallet")

    return {
        "repo": "setrf/megaminx-world-model-bench",
        "prime_environment": "setrf/megaminx-solver",
        "expected_env_version": EXPECTED_ENV_VERSION,
        "ready_for_local_validation": not local_failures,
        "ready_for_hosted_prime_runs": not failures
        and artifact_statuses == {"oracle JSONL": "passed", "SFT JSONL": "passed"}
        and prime_status == "passed",
        "checks": checks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = readiness_payload(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if any(check["status"] == "failed" for check in payload["checks"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
