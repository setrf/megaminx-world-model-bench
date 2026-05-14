# Megaminx Next Run Runbook

This runbook is the handoff for the remaining hosted work after v0.2.56.
It assumes the repo is `setrf/megaminx-world-model-bench`, the Prime owner is
`setrf`, and the environment slug is `setrf/megaminx-solver`.

Current local state as of 2026-05-14 07:09 Istanbul:

- `main` contains v0.2.56 plus the next-run readiness-tool follow-up.
- `setrf/megaminx-solver@0.2.56` was pushed and its Hub action
  `kioezfzz4ji4uquyhm0grzwc` reached `SUCCESS`.
- Prime CLI auth currently fails with `API key unauthorized`.
- The last successful wallet check reported balance `$-0.80`, so hosted run
  creation is blocked until billing is restored.
- The Hub visibility field still reports `PRIVATE` despite public push attempts.

Do not paste or commit API keys. The local Prime config contains a saved key,
but `prime whoami --plain` currently rejects it.

## 1. Restore Prime Access

```bash
prime login
prime whoami --plain
prime wallet --plain
prime env status setrf/megaminx-solver --plain
```

Proceed only when:

- `prime whoami --plain` prints the authenticated user instead of
  `API key unauthorized`.
- `prime wallet --plain` shows a positive balance.
- `prime env status setrf/megaminx-solver --plain` reports latest version
  `0.2.56`, hash `8a1d0168b96c`, and action status `SUCCESS`.

If visibility still reports `PRIVATE`, keep running owner-authenticated probes.
Do not block checkpoint validation on public visibility.

The combined readiness command is:

```bash
uv run python scripts/check_next_run_readiness.py --check-prime
```

It validates the four tracked v0.2.56 matched probe configs, the canonical
oracle/SFT artifacts if present under `/tmp`, and redacted Prime auth/wallet/env
status. It should report `ready_for_hosted_prime_runs: true` before launching
the hosted probes below.

## 2. Validate Local Package And Oracle Data

```bash
prime env install setrf/megaminx-solver@0.2.56 --plain
uv run pytest -q
uv run python scripts/check_next_run_readiness.py --skip-oracle --skip-sft

uv run python scripts/export_oracle_trajectories.py \
  --num-examples 1024 \
  --seed 64 \
  --split train_candidate_relative_flow_rule_tail_solve_depth2 \
  --output /tmp/megaminx-oracle-v056-1024.jsonl

uv run python scripts/summarize_oracle_trajectories.py \
  /tmp/megaminx-oracle-v056-1024.jsonl

uv run python scripts/convert_oracle_to_sft_jsonl.py \
  /tmp/megaminx-oracle-v056-1024.jsonl \
  --output /tmp/megaminx-oracle-v056-1024-sft.jsonl

uv run python scripts/validate_sft_jsonl.py \
  /tmp/megaminx-oracle-v056-1024-sft.jsonl

uv run python scripts/project_sft_to_openai_tools.py \
  /tmp/megaminx-oracle-v056-1024-sft.jsonl \
  --output /tmp/megaminx-oracle-v056-1024-sft-openai.jsonl

uv run python scripts/check_next_run_readiness.py
```

Expected oracle JSONL SHA256:

```text
1038afa6958030832c028840dafc22fc3724206461608e2ed809c90fa9695e7b
```

Expected SFT JSONL SHA256:

```text
59b9db129517f3a6f86a868f06179826a032b2e0d07c4393d5a9ae168e8b1ec3
```

Expected OpenAI-style SFT projection SHA256:

```text
2ed51c37e74e32d7944bc7ef14d2bc0d059886698c88d4fbac1e17fbd2604627
```

The summary must show:

- `rows: 1024`
- `env_versions: {"0.2.56": 1024}`
- `final_rewards: {"1.0": 1024}`
- `solved: {"True": 1024}`
- `action_counts: {"2": 1024}`
- `prompt_leak_count: 0`
- stable turn-local tool-call ids (`call_1`, `call_2`) and no legacy
  row-derived `oracle-*` ids

The SFT validation summary must also show:

- `rows: 1024`
- `env_versions: {"0.2.56": 1024}`
- `tool_names: {"select_candidate": 1024}`
- `forbidden_payload_fields: 0`
- `prompt_leak_count: 0`
- metadata contains only non-identifying environment fields; it excludes row
  ids, numeric example ids, seed, split, scramble, inverse solution, and action
  traces

## 3. Run Matched v0.2.56 Heldout Probes

These are the highest-priority hosted commands after billing/auth recovery:

```bash
prime train configs/rl/megaminx-v056-qwen9b-tail-solve-depth2-base-heldout-rpe16.toml --yes --plain
prime train configs/rl/megaminx-v056-qwen9b-tail-solve-depth2-base-heldout2-rpe16.toml --yes --plain
prime train configs/rl/megaminx-v056-qwen9b-tail-solve-depth2-ckpt2-heldout-rpe16.toml --yes --plain
prime train configs/rl/megaminx-v056-qwen9b-tail-solve-depth2-ckpt2-heldout2-rpe16.toml --yes --plain
```

The checkpoint probes use `checkpoint_id = "o68kzy5up4e65ve6lktmkuat"`, the
READY step-2 checkpoint from v0.2.55 continuation run
`gnpet9lrxx16amnytkcb8vju`. The base and checkpoint probes are matched on
model, sampling, environment version, reward style, prompt style, rollout count,
and heldout seeds.

Inspect each run:

```bash
prime train get <run_id> --plain
prime train metrics <run_id> --plain
prime train usage <run_id> --plain
prime train checkpoints <run_id> --plain
```

Record at minimum:

- reward
- solved rate
- first and second candidate correctness
- `native_tool_call_count`
- illegal/protocol/tool errors
- cost
- run id

Acceptance for a material v0.2.56 checkpoint claim:

- no environment, tool-call, protocol, or illegal-move errors
- native tool calls remain near `2.0`
- checkpoint beats base on both heldout seeds
- two-seed solved-rate gain is large enough to be scientifically meaningful
- original stretch target remains `+30pp`; if this is not reached, report the
  exact smaller gain as a partial/negative result

## 4. If v0.2.56 Checkpoint Does Not Improve

Use the SFT JSONL as the next warm-start path. The exported SFT file contains
only `messages`, `tools`, and safe metadata. It intentionally omits scramble,
inverse solution, and action metadata outside the assistant tool calls.

Recommended warm-start gate before more PPO:

- train or initialize from `/tmp/megaminx-oracle-v056-1024-sft.jsonl`
- validate on the same two heldout seeds 146 and 246
- require zero protocol/tool errors
- require exactly two native `select_candidate` actions per rollout
- require the SFT adapter to beat the base model before launching more PPO

Local scaffold commands:

```bash
uv run python scripts/train_sft_lora.py \
  --config configs/sft/megaminx-v056-qwen08b-tail-solve-smoke.toml \
  --dry-run
uv run python scripts/train_sft_lora.py \
  --config configs/sft/megaminx-v056-qwen9b-tail-solve-lora.toml
```

The smoke config validates the data/format path through Qwen's native
chat/tool-call template and reports dependency status. The trainer masks loss to
assistant tool-call spans, enables gradient checkpointing, and treats
`max_steps` as optimizer updates, so gradient accumulation does not silently
shrink the intended run. A one-step local MPS smoke on 0.8B reached
forward/backward but hit the Mac memory ceiling at 4096 tokens; run the full 9B
config on a larger CUDA GPU box with `torch`, `transformers`, and `peft`.

Do not continue long low-learning-rate PPO on the same distribution without
heldout gates. Previous runs repeatedly showed online movement that did not
cleanly generalize.

## 5. Reporting

Update `reports/megaminx-rl-report.md` with:

- all new run ids and commands
- base/checkpoint metric table
- costs from `prime train usage`
- whether v0.2.56 improves over base
- whether the original `+30pp` target is achieved
- if not achieved, the rigorous negative conclusion and the next experiment

Then commit and push:

```bash
git add configs/rl README.md environments/megaminx_solver/README.md reports/megaminx-rl-report.md
git commit -m "Report v0.2.56 heldout probe results"
git push origin main
```
