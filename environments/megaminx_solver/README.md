# megaminx-solver

[![Prime Hub](https://img.shields.io/badge/Prime%20Hub-setrf%2Fmegaminx--solver-blue)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Latest env](https://img.shields.io/badge/env-v0.2.29-0f766e)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Native heldout](https://img.shields.io/badge/native%20step40-0.807-16a34a)](https://app.primeintellect.ai/dashboard/training/b6qh3e36i2poqghq5eum97dd)
[![Breakthrough eval](https://img.shields.io/badge/v0.2.22%2035B-0.929-2563eb)](https://app.primeintellect.ai/dashboard/evaluations/vse6uoo8c9y156svyyv41qll)

Trainable multi-turn Megaminx environment for Prime Intellect Verifiers. The
model sees a compact text facelet net, acts through tools, and receives a
deterministic reward from a persistent puzzle simulator.

## Overview

| Item | Value |
| --- | --- |
| Environment id | `megaminx-solver` |
| Hub target | `setrf/megaminx-solver` |
| Current package | `setrf/megaminx-solver@0.2.29` |
| Type | `vf.StatefulToolEnv` |
| Tags | `multi-turn`, `tool-env`, `spatial-reasoning`, `megaminx`, `rl`, `eval` |

## Task

Each rollout starts from a deterministic generated scramble. Faces are labeled
`A` through `L`; a solved face contains only its own label. The model can call:

| Tool | Arguments | Meaning |
| --- | --- | --- |
| `rotate` | `face: A-L`, `direction: cw|ccw` | Apply one face turn |
| `inspect` | `face: A-L|all` | Observe a face or full net |
| `finish` | none | End the rollout |

The scramble and inverse solution are stored in task metadata for tests and
metrics, but they are not printed directly in the prompt.

## Public API

```python
load_environment(
    split="train",
    min_depth=1,
    max_depth=8,
    num_examples=200,
    seed=42,
    max_turns=None,
    move_budget=None,
    reward_style="dense",
    prompt_style="default",
    allow_text_tool_actions=None,
)
```

Named splits:

| Split | Depths | Intended use |
| --- | ---: | --- |
| `depth1`, `train_depth1`, `eval_depth1` | 1 | First RL and one-turn heldout |
| `easy`, `train_easy`, `eval_easy` | 1-3 | Short-scramble generalization |
| `medium`, `train_medium`, `eval_medium` | 4-6 | Mid-horizon eval |
| `hard`, `train_hard`, `eval_hard` | 7-10 | Hard eval |
| `eval` | 1-10 | Broad eval |
| `train` | custom, default 1-8 | General curriculum |

Default move budget is `2 * scramble_depth + 4`, capped at `32`, unless
`max_turns` or `move_budget` is supplied. For the staged depth-1 experiments we
use `max_turns=2` and `move_budget=1`.

## Rewards

Default dense reward:

```text
0.60 * solved
+ 0.25 * sticker_accuracy
+ 0.10 * piece_accuracy
+ 0.05 * efficiency_if_solved
```

Supported reward styles:

- `dense`
- `action_gated_dense`
- `action_gated_curriculum`
- `action_gated_overlap`
- `action_gated_direction`
- `action_gated_exact_direction`
- `action_gated_binary_direction`
- `action_gated_strict_shaped_direction`

The strict eval gate is `action_gated_binary_direction`: reward is `1.0` only
for one clean first `rotate` equal to the inverse move, and `0.0` otherwise.
Clean means exactly one action attempt, exactly one rotate, no inspect/finish,
no illegal move, no parse/call error, and no protocol violation.

The current hosted RL on-ramp uses
`action_gated_strict_shaped_direction`: solved exact action gives `1.0`, a
clean wrong-direction same-face action gives partial credit around `0.40`, and
other valid rotations receive smaller direction/progress credit. This keeps
rollouts learnable while the heldout acceptance metric remains binary.

`reward_style` is exported as a numeric metric: binary direction is `6.0`,
strict shaped direction is `7.0`.

## Prompt Styles

The main current styles are:

- `stage_solve_direction_flow_native_tool`: native tool-call training/probe
  lane. Use `allow_text_tool_actions=false`.
- `stage_solve_direction_flow_json_action`: served chat-completions eval lane.
  Use `allow_text_tool_actions=true` for Qwen JSON/private text fallback.
- `default`, `action_first`, topology/sensor JSON styles, and older staged
  prompts remain available for ablations and history.

Depth-1 staged prompts intentionally reveal the affected face and isolate the
binary direction bit. They do not print the scramble, inverse solution, answer
candidate, winner, overlap count, or precomputed best action.

## Metrics

Tracked metrics include:

```text
illegal_move_count
move_count
solved_rate
scramble_depth
tool_call_count
native_tool_call_count
text_tool_action_count
private_text_action_count
tool_parse_error_count
tool_call_error_count
protocol_violation_count
rotate_call_count
inspect_call_count
finish_call_count
action_taken
first_rotate_correct
first_rotate_face_correct
first_rotate_direction_correct
first_rotate_neighbor_overlap
reward_style
initial_sticker_accuracy
initial_piece_accuracy
```

v0.2.29 hardens the private reasoning path: structured non-action private
metadata such as reasoning summaries is ignored, while malformed private JSON
action attempts are counted as private parse errors.

## Validation

Install the current Hub package:

```bash
prime env install setrf/megaminx-solver@0.2.29 --plain
```

Run local tests:

```bash
uv run pytest -q
```

Latest result: `59 passed`.

Quick served eval smoke:

```bash
prime eval run setrf/megaminx-solver@0.2.29 -m Qwen/Qwen3.5-9B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"eval_depth1","num_examples":5,"seed":1042,"max_turns":2,"move_budget":1,"reward_style":"action_gated_binary_direction","prompt_style":"stage_solve_direction_flow_json_action","allow_text_tool_actions":true}' \
  -n 5 -r 1 -t 48 -T 0.7 -A --plain
```

Run the native hosted RL config used for the step-40 result:

```bash
prime train configs/rl/megaminx-depth1-qwen-9b-solve-flow-shaped-v028-native-noeval.toml --yes --plain
```

Run the native heldout probe shape:

```bash
prime train configs/rl/megaminx-v028-native-heldout-probe-step40.toml --yes --plain
```

## Current Evidence

The v0.2.22 environment/prompt breakthrough moved 35B depth-1 direction solving
from [`0.504`](https://app.primeintellect.ai/dashboard/evaluations/dgu4mymqqk5sy97l3kvusjxu)
to [`0.929`](https://app.primeintellect.ai/dashboard/evaluations/vse6uoo8c9y156svyyv41qll).

The v0.2.28/v0.2.29 RL state is more modest. Native TITO heldout probes on the
same 240-example depth-1 set show:

| Probe | Checkpoint | Reward/solved | Note |
| --- | --- | ---: | --- |
| [`gwfcyfztn8ahpwceu9n63rzd`](https://app.primeintellect.ai/dashboard/training/gwfcyfztn8ahpwceu9n63rzd) | base | `0.7819` | native calls clean |
| [`eg68wcx3ym12mhvtku4l7g5g`](https://app.primeintellect.ai/dashboard/training/eg68wcx3ym12mhvtku4l7g5g) | step 30 | `0.7700` | regressed |
| [`b6qh3e36i2poqghq5eum97dd`](https://app.primeintellect.ai/dashboard/training/b6qh3e36i2poqghq5eum97dd) | step 40 | `0.8071` | modest positive |

The original `+30pp` trained-checkpoint acceptance target was not met. The
honest current result is: environment released and hardened, native-tool
measurement path established, modest positive step-40 RL signal, and a clear
next experiment needed for larger gains.

## Hub Caveat

`prime env push megaminx-solver --path ./environments --owner setrf --visibility PUBLIC --plain`
successfully pushes versions, but the existing Hub record still reports
`PRIVATE` through `prime env status` and `prime env list`. Flip visibility in the
owner web UI for environment id `ozde27sytxjkc3wm83zv4e2c`; do not delete and
recreate the environment unless losing version/eval/training continuity is
acceptable.
