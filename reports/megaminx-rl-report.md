# Megaminx RL Environment Report

Date: 2026-05-12

## Executive Summary

`setrf/megaminx-solver` is a Prime/Verifiers environment for testing whether LLMs can operate on a compact symbolic Megaminx world model. The environment exposes a stateful tool loop over a deterministic Megaminx simulator, with short-scramble curriculum, dense validation metrics, and an RL-facing action-gated reward.

Current release state:

- GitHub repo: `setrf/megaminx-world-model-bench`
- Working branch: `codex/megaminx-rl-environment`
- Prime Hub slug: `setrf/megaminx-solver`
- Latest pushed Hub version: `0.2.5`
- Hub environment id: `ozde27sytxjkc3wm83zv4e2c`
- Latest version id: `p24gxvpm8pxk7gtll43wwemz`
- Hosted training run: `n1s75wk4cv32g7xhs1ed9bu6`
- Release blocker: Hub visibility still reports `PRIVATE` after successful `prime env push ... --visibility PUBLIC` attempts.

## Motivation

The goal is to build a trainable RL environment that probes physical-world-style reasoning in LLMs through a discrete mechanical puzzle. A Megaminx is useful because it has:

- A persistent world state.
- Local physical actions with global consequences.
- A sparse true objective, but measurable intermediate structure.
- A clear deterministic validator.
- Natural curriculum bands from one-turn scrambles to longer horizons.

v0 is text-first. Rendered/image observations are reserved for later versions after the symbolic simulator is stable under evaluation and training.

## Environment Design

Package path:

```bash
environments/megaminx_solver
```

Entrypoint:

```python
load_environment(...) -> vf.Environment
```

The environment class is `MegaminxEnv(vf.StatefulToolEnv)` because each rollout needs persistent puzzle state across tool calls.

Public API:

```python
load_environment(
    split="train",
    min_depth=1,
    max_depth=8,
    num_examples=200,
    seed=42,
    max_turns=None,
    reward_style="dense",
    prompt_style="default",
)
```

Tools:

- `rotate(face: A-L, direction: cw|ccw)`
- `inspect(face: A-L|all)`
- `finish()`

Observations include a compact facelet net plus solvedness, sticker accuracy, piece accuracy, move budget, and last move.

Simulator properties:

- 12 faces labeled `A` through `L`.
- Each face has 1 center, 5 edge stickers, and 5 corner stickers.
- Total stickers: 132.
- Legal move set: 24 moves, clockwise/counterclockwise for each face.
- Scramble and inverse solution are stored in task metadata only, never in the prompt.

## Curriculum

Named splits:

| Split | Depths | Intended use |
| --- | ---: | --- |
| `depth1` | 1 | Local smoke and baseline eval |
| `train_depth1` | 1 | First hosted RL pass |
| `eval_depth1` | 1 | Held-out depth-1 training eval |
| `easy` | 1-3 | First generalization band |
| `medium` | 4-6 | Mid-horizon eval |
| `hard` | 7-10 | Hard eval |
| `eval` | 1-10 | Broad eval |
| `train` | custom args, default 1-8 | Default curriculum |

Generated scrambles are deterministic by seed and avoid immediate inverse moves.

## Reward Design

Default dense reward is kept for backward compatibility:

```text
0.60 * solved
+ 0.25 * sticker_accuracy
+ 0.10 * piece_accuracy
+ 0.05 * efficiency_if_solved
```

The RL config uses `reward_style="action_gated_dense"`:

- No rollout or no rotate action: `0.0`.
- Solved state: `1.0`.
- Valid non-solving rotations: positive progress-delta reward only, capped at `0.4`.

This fixes the initial reward trap where a model could receive roughly `0.30` dense reward by doing nothing because a one-turn scramble is still mostly solved.

Prompt variant `prompt_style="action_first"` explicitly tells the model that the first assistant turn must be a `rotate` action and that text before rotate receives zero reward.

## Qwen Tool-Action Compatibility

Prime local evals against Qwen/Pinference often emitted JSON action text in `content` or `reasoning_content` rather than OpenAI `tool_calls`. Version `0.2.5` adds a compatibility shim that parses simple JSON action objects/arrays and executes them through the same environment tool path.

Important metric note: Verifiers' built-in `rotate_calls` stays `0.0` for these JSON-text actions because the upstream model did not emit native tool calls. The environment therefore reports custom `rotate_call_count`, `inspect_call_count`, `finish_call_count`, `tool_call_count`, and `action_taken` metrics, which are the correct action metrics for current Qwen evals.

## Validation

Unit and environment tests cover:

- 12 centers, 30 edges, 20 corners, and 132 stickers.
- Move sticker multiset preservation.
- `cw` then `ccw` identity.
- Five clockwise turns on one face identity.
- Generated scramble plus inverse solution solves the puzzle.
- Invalid tool arguments are safe and low reward.
- Named split depth bands.
- `action_gated_dense` no-action behavior and solved reward.
- No answer leakage in prompts.
- Tool schema fields.
- JSON-text and reasoning-content action fallback.

Local command:

```bash
uv run pytest
```

Latest local result:

```text
11 passed in 2.38s
```

## Release History

| Version | Purpose |
| --- | --- |
| `0.2.1` | Dense reward baseline and README/badge update |
| `0.2.2` | Added `reward_style` and `prompt_style` |
| `0.2.3` | Prompt and sampling iteration |
| `0.2.4` | JSON action fallback for message content |
| `0.2.5` | JSON action fallback for reasoning content; active training version |

Latest Hub version check:

```bash
prime env version list setrf/megaminx-solver --plain
```

Latest version: `0.2.5`, content hash `22b601d1`.

## Visibility Blocker

The intended public release command has been run:

```bash
prime env push megaminx-solver --visibility PUBLIC --plain
```

The push succeeds and creates new versions, but `prime env list --search megaminx --output json` and `prime env info setrf/megaminx-solver --plain` still report `PRIVATE`.

Direct checks:

```bash
prime env list --search megaminx --output json
prime env info setrf/megaminx-solver --plain
```

Observed state:

```text
visibility: PRIVATE
version: 0.2.5
```

The app frontend has an owner-only `environments.updateEnvironment` mutation for visibility, but the CLI API token is not accepted by the app tRPC endpoint. This likely needs a logged-in Hub UI action or a Prime-side visibility update.

Do not delete/recreate the environment to force public visibility unless losing version/eval/training continuity is explicitly accepted.

## Baseline Evaluations

All canonical evals were uploaded; no `--skip-upload` was used.

### Dense Reward Baseline

Command:

```bash
prime eval run setrf/megaminx-solver \
  -m Qwen/Qwen3.5-4B \
  -a '{"split":"depth1","num_examples":5}' \
  -n 1 -r 1 -t 768 -A
```

Result:

- URL: https://app.primeintellect.ai/dashboard/evaluations/etr6y2vgx4yw1z7sbn70wjwp
- Version: `0.2.1`
- Reward: `0.3015909091`
- Solved rate: `0.0`
- Move count: `0.0`
- Rotate calls: `0.0`
- Outcome: exposed the no-action dense reward trap.

### Action-Gated Depth-1 Smoke Matrix

Command template:

```bash
prime eval run setrf/megaminx-solver \
  -m <model> \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":"required"}' \
  -a '{"split":"depth1","num_examples":30,"max_turns":2,"reward_style":"action_gated_dense","prompt_style":"action_first"}' \
  -n 10 -r 3 -t 256 -A
```

| Model | Eval | Reward | Solved | Rotate action rate | Illegal moves | First rotate correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.5-0.8B` | https://app.primeintellect.ai/dashboard/evaluations/ollhvhlhjyve8h1ckqbvt4aw | `0.000` | `0.000` | `0.600` | `0.000` | `0.000` |
| `Qwen/Qwen3.5-4B` | https://app.primeintellect.ai/dashboard/evaluations/dq7ajgaiw0vdi6fz284a8ulp | `0.000` | `0.000` | `0.967` | `0.000` | `0.000` |
| `Qwen/Qwen3.5-9B` | https://app.primeintellect.ai/dashboard/evaluations/n1d08t0i1q0i7b3ict2quyn7 | `0.000` | `0.000` | `0.767` | `0.000` | `0.000` |

Gate decision: proceed to hosted RL. Environment errors were zero, illegal moves were zero, and rotate calls were nonzero.

## Hosted RL Run

Launch command:

```bash
prime train configs/rl/megaminx-depth1-qwen-4b.toml --yes --plain --output json
```

Run:

- Run id: `n1s75wk4cv32g7xhs1ed9bu6`
- Status at report draft: `RUNNING`
- Base model: `Qwen/Qwen3.5-4B`
- Environment: `setrf/megaminx-solver@0.2.5`
- Visibility at run launch: `PRIVATE`
- Max steps: `100`
- Batch size: `128`
- Rollouts per example: `8`
- Learning rate: `1e-7`
- LoRA alpha: `16`
- Oversampling factor: `2.0`
- Max async level: `1`
- Sampling max tokens: `256`
- Temperature: `0.7`
- Extra body: `tool_choice="required"`

Monitoring commands:

```bash
prime train get n1s75wk4cv32g7xhs1ed9bu6 --plain --output json
prime train logs n1s75wk4cv32g7xhs1ed9bu6 --plain --tail 120
prime train metrics n1s75wk4cv32g7xhs1ed9bu6 --plain
prime train progress n1s75wk4cv32g7xhs1ed9bu6 --plain
prime train distributions n1s75wk4cv32g7xhs1ed9bu6 --plain
prime train usage n1s75wk4cv32g7xhs1ed9bu6 --plain --output json
prime train checkpoints n1s75wk4cv32g7xhs1ed9bu6 --plain
```

Usage snapshot while step-0 evals were still running:

```text
training tokens: 0
inference tokens: 1,070,251
total cost: $0.1926
```

The run started, installed `setrf/megaminx-solver@0.2.5`, loaded training and eval environments, and began step-0 evals. `eval_depth1` completed with no environment failures:

```text
eval/eval_depth1/avg@1: 0.0
eval/eval_depth1/pass@1: 0.0
eval/eval_depth1/failed_rollouts: 0.0
eval/eval_depth1/is_truncated/mean: 0.0
eval/eval_depth1/completion_len/mean: 1714.2
eval/eval_depth1/time: 911.8007s
```

No checkpoints were available at the time of this report update.

Stop conditions:

- Any environment errors.
- Reward flat for 25 steps.
- Rotate calls near zero.
- Reward exploitation without solving.
- Infrastructure failure or runaway cost.

## Final Evaluation Plan

When checkpoints are available, evaluate base and trained models on identical sets:

```bash
prime train checkpoints n1s75wk4cv32g7xhs1ed9bu6 --plain
```

Then use the checkpoint model identifier returned by Prime in commands of this form:

```bash
prime eval run setrf/megaminx-solver \
  -m <checkpoint-model-id> \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":"required"}' \
  -a '{"split":"depth1","num_examples":120,"max_turns":2,"reward_style":"action_gated_dense","prompt_style":"action_first"}' \
  -n 120 -r 1 -t 256 -A
```

Required final bands:

- `depth1`
- `easy`
- `medium`
- `hard`
- `eval`

Required metrics:

- Reward.
- Solved rate.
- First-rotate correctness.
- Sticker accuracy.
- Piece accuracy.
- Move count.
- Rotate calls.
- Inspect calls.
- Illegal moves.
- Truncation rate.
- Token usage.
- Cost.

Acceptance target:

- Hub environment public and installable.
- Trained checkpoint improves depth-1 solved rate by at least `+30pp`.
- `easy` reward improves over base.
- No environment errors.
- Commands and results are reproducible.

If the checkpoint fails the acceptance target, the result should be reported as an environment release plus negative RL result, with the failure mode preserved for the next experiment.

## Repository Work

Added or updated:

- Environment implementation under `environments/megaminx_solver/`.
- Unit/environment tests under `tests/`.
- RL configs under `configs/rl/`.
- Eval configs under `configs/eval/`.
- GitHub Actions CI under `.github/workflows/ci.yml`.
- MIT license in `LICENSE`.
- Root README quickstart and links.

CI command:

```bash
uv sync --all-extras --dev
uv run pytest
```

Release steps after training/final eval:

```bash
git checkout codex/megaminx-rl-environment
uv run pytest
git add README.md LICENSE .github configs environments tests reports pyproject.toml
git commit -m "Build Megaminx RL environment"
git push origin codex/megaminx-rl-environment
gh pr create --base main --head codex/megaminx-rl-environment \
  --title "Build Megaminx RL environment" \
  --body-file reports/megaminx-rl-report.md
```

After merge:

```bash
git checkout main
git pull origin main
git tag v0.2.5
git push origin v0.2.5
```

Do not force-push or overwrite `main`.

## Next Experiment If Needed

If depth-1 solved rate stays below `20%`:

- Make the first action target more legible in the observation, without leaking the inverse solution.
- Add a `training_hint` prompt style only for depth-1 RL.
- Increase progress reward for correct face or correct direction separately.
- Consider a two-stage curriculum: direction-only, then face+direction.
- Repeat a 100-step Qwen 4B run before scaling to 9B.

If depth-1 solved rate reaches `>=60%`:

- Continue to a 200-step `train_easy` run.
- Keep `max_turns` short for depth-1, but let easy use the default global cap.
- Run the same curriculum on `Qwen/Qwen3.5-9B` for capacity scaling.
