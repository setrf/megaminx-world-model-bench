# Megaminx World Model Bench

[![Prime Hub](https://img.shields.io/badge/Prime%20Hub-setrf%2Fmegaminx--solver-blue)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Latest env](https://img.shields.io/badge/env-v0.2.29-0f766e)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Native probe](https://img.shields.io/badge/native%20step40-0.807-16a34a)](https://app.primeintellect.ai/dashboard/training/b6qh3e36i2poqghq5eum97dd)
[![GitHub branch](https://img.shields.io/badge/branch-codex%2Fmegaminx--direction--breakthrough-black)](https://github.com/setrf/megaminx-world-model-bench/tree/codex/megaminx-direction-breakthrough)
[![License](https://img.shields.io/badge/License-MIT-0f766e)](LICENSE)

Prime Lab workspace for `setrf/megaminx-solver`, a trainable Prime/Verifiers
environment that tests whether LLMs can act in a compact symbolic Megaminx
world. The environment exposes a stateful tool loop: inspect a text facelet net,
call `rotate(face, direction)`, and receive deterministic reward from the
puzzle simulator.

## Current Status

| Item | Value |
| --- | --- |
| GitHub repo | `setrf/megaminx-world-model-bench` |
| Working branch | `codex/megaminx-direction-breakthrough` |
| Prime owner | `setrf` |
| Hub environment | [`setrf/megaminx-solver`](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver) |
| Environment id | `ozde27sytxjkc3wm83zv4e2c` |
| Latest package | `setrf/megaminx-solver@0.2.29` |
| Latest content hash | `d7be3414695e` |
| Latest wheel SHA256 | `f19e1209dabae728af50ad540528ef5c71edb838471886cb0945ca078aae8db8` |
| Latest local tests | `uv run pytest -q` -> `59 passed` |
| Hosted RL run | [`n37avwtg2evd9scmdqmafbdr`](https://app.primeintellect.ai/dashboard/training/n37avwtg2evd9scmdqmafbdr), stopped after step 40/42 |
| Visibility | CLI/API still report `PRIVATE` after `--visibility PUBLIC`; owner-auth install/eval/train work |

## What We Cracked

The important result is two-part and should be read honestly:

1. **Environment/prompt direction breakthrough.** v0.2.21 showed that the model
   could identify the depth-1 face but stayed at chance on `cw` versus `ccw`:
   [`Qwen/Qwen3.5-35B-A3B` solved `0.504`](https://app.primeintellect.ai/dashboard/evaluations/dgu4mymqqk5sy97l3kvusjxu).
   v0.2.22 reframed the same evidence as candidate solving actions and reached
   [`0.929` solved](https://app.primeintellect.ai/dashboard/evaluations/vse6uoo8c9y156svyyv41qll)
   on the identical 240-example heldout set.

2. **Native RL measurement lane.** The credible native-tool lane is Prime Hosted
   Training's Token-In Token-Out path. The v0.2.28 native heldout probes show a
   modest positive RL movement, not the original `+30pp` target:

| Probe | Model/checkpoint | Reward/solved | Native tools | Errors |
| --- | --- | ---: | ---: | ---: |
| [`gwfcyfztn8ahpwceu9n63rzd`](https://app.primeintellect.ai/dashboard/training/gwfcyfztn8ahpwceu9n63rzd) | base `Qwen/Qwen3.5-9B` | `0.7819` | `1.0` | `0` |
| [`eg68wcx3ym12mhvtku4l7g5g`](https://app.primeintellect.ai/dashboard/training/eg68wcx3ym12mhvtku4l7g5g) | step 30 | `0.7700` | `1.0` | `0` |
| [`b6qh3e36i2poqghq5eum97dd`](https://app.primeintellect.ai/dashboard/training/b6qh3e36i2poqghq5eum97dd) | step 40 | `0.8071` | `1.0` | `0` |

Served chat-completions evals are still useful for compatibility history, but
they do not measure the same native tool-call surface for these Qwen adapters:
the served endpoint often emits JSON/private text rather than native
`tool_calls`. See [the report](reports/megaminx-rl-report.md) for the full
native-vs-served accounting.

## Quickstart

```bash
prime env install setrf/megaminx-solver@0.2.29 --plain
uv run pytest -q
```

Smoke the latest package on the served JSON compatibility lane:

```bash
prime eval run setrf/megaminx-solver@0.2.29 -m Qwen/Qwen3.5-9B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"eval_depth1","num_examples":5,"seed":1042,"max_turns":2,"move_budget":1,"reward_style":"action_gated_binary_direction","prompt_style":"stage_solve_direction_flow_json_action","allow_text_tool_actions":true}' \
  -n 5 -r 1 -t 48 -T 0.7 -A --plain
```

Reproduce the main native shaped training run configuration:

```bash
prime train configs/rl/megaminx-depth1-qwen-9b-solve-flow-shaped-v028-native-noeval.toml --yes --plain
```

Reproduce the native heldout probe shape with the step-40 checkpoint:

```bash
prime train configs/rl/megaminx-v028-native-heldout-probe-step40.toml --yes --plain
```

## Environment Shape

- Entry point: `load_environment(...) -> vf.Environment`
- Implementation: `MegaminxEnv(vf.StatefulToolEnv)`
- Tools: `rotate(face: A-L, direction: cw|ccw)`, `inspect(face: A-L|all)`,
  `finish()`
- Simulator: 12 faces, 30 edge pieces, 20 corner pieces, 132 stickers, 24 legal
  face turns, deterministic scrambles and inverse metadata hidden from prompts
- Curriculum splits: `depth1`, `easy`, `medium`, `hard`, `eval`, plus `train_*`
  and `eval_*` variants

The v0.2.29 package keeps the v0.2.28 native RL behavior and adds robustness for
structured private reasoning metadata so non-action private fields are ignored
instead of being counted as malformed tool attempts.

## Protocol Lanes

Prime's renderer/TITO direction matters here because a one-action puzzle task is
very sensitive to whether the sampled assistant/tool-call tokens are preserved
exactly. The project follows this split:

- **Native training/probe lane:** `stage_solve_direction_flow_native_tool`,
  `allow_text_tool_actions=false`, Hosted Training token-native tool calls,
  metrics show `native_tool_call_count=1.0`.
- **Served eval compatibility lane:** `stage_solve_direction_flow_json_action`,
  `allow_text_tool_actions=true`, chat-completions JSON/private text fallback,
  metrics show `private_text_action_count` or `text_tool_action_count`.

The report cites the Prime Environments, Verifiers, and Renderers docs used to
set this boundary.

## Files

- Environment package: `environments/megaminx_solver/`
- Main implementation: `environments/megaminx_solver/megaminx_solver/megaminx_solver.py`
- Tests: `tests/test_megaminx_solver.py`
- Hosted RL configs: `configs/rl/`
- Eval configs: `configs/eval/`
- Final report: `reports/megaminx-rl-report.md`
