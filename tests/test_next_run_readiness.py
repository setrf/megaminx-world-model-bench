from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_next_run_readiness

from test_oracle_export import _run_export


def test_next_run_readiness_cli_validates_configs_without_prime() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "check_next_run_readiness.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--skip-oracle",
            "--skip-sft",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["repo"] == "setrf/megaminx-world-model-bench"
    assert payload["prime_environment"] == "setrf/megaminx-solver"
    assert payload["expected_env_version"] == "0.2.57"
    assert payload["ready_for_local_validation"] is True
    assert payload["ready_for_hosted_prime_runs"] is False

    statuses = {check["name"]: check["status"] for check in payload["checks"]}
    assert statuses["v0.2.56 matched probe configs"] == "passed"
    assert statuses["oracle JSONL"] == "skipped"
    assert statuses["SFT JSONL"] == "skipped"
    assert statuses["Prime access and wallet"] == "skipped"


def test_next_run_readiness_cli_validates_generated_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "check_next_run_readiness.py"
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

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--oracle-jsonl",
            str(oracle_output),
            "--sft-jsonl",
            str(sft_output),
            "--allow-noncanonical-artifacts",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    statuses = {check["name"]: check["status"] for check in payload["checks"]}
    assert statuses["v0.2.56 matched probe configs"] == "passed"
    assert statuses["oracle JSONL"] == "passed"
    assert statuses["SFT JSONL"] == "passed"
    assert statuses["Prime access and wallet"] == "skipped"
    assert payload["ready_for_local_validation"] is True
    assert payload["ready_for_hosted_prime_runs"] is False


def test_next_run_readiness_keeps_local_ready_when_prime_check_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    oracle_output = tmp_path / "oracle.jsonl"
    sft_output = tmp_path / "sft.jsonl"
    repo_root = Path(__file__).resolve().parents[1]
    convert_script = repo_root / "scripts" / "convert_oracle_to_sft_jsonl.py"

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

    monkeypatch.setattr(
        check_next_run_readiness,
        "check_prime_access",
        lambda _timeout: {
            "name": "Prime access and wallet",
            "status": "failed",
            "details": {"errors": ["whoami: command failed"]},
        },
    )

    payload = check_next_run_readiness.readiness_payload(
        Namespace(
            oracle_jsonl=oracle_output,
            sft_jsonl=sft_output,
            skip_oracle=False,
            skip_sft=False,
            require_artifacts=False,
            allow_noncanonical_artifacts=True,
            check_prime=True,
            prime_timeout=1,
        )
    )

    assert payload["ready_for_local_validation"] is True
    assert payload["ready_for_hosted_prime_runs"] is False


def test_prime_access_check_requires_expected_hub_status(monkeypatch) -> None:
    def fake_run_prime_command(args: list[str], *, timeout: int, line_limit: int = 12) -> dict:
        assert timeout == 7
        assert line_limit >= 4
        command = " ".join(args)
        if "wallet" in args:
            return {"ok": True, "returncode": 0, "stdout": ["Balance: $10.00"], "stderr": []}
        if "env status" in command:
            return {
                "ok": True,
                "returncode": 0,
                "stdout": ["setrf/megaminx-solver 0.2.55 oldhash SUCCESS"],
                "stderr": [],
            }
        return {"ok": True, "returncode": 0, "stdout": ["setrf"], "stderr": []}

    monkeypatch.setattr(
        check_next_run_readiness,
        "_run_prime_command",
        fake_run_prime_command,
    )

    check = check_next_run_readiness.check_prime_access(7)

    assert check["status"] == "failed"
    assert "env_status: expected '0.2.57' in output" in check["details"]["errors"]
    assert "env_status: expected '35d4bb90de33' in output" in check["details"]["errors"]
