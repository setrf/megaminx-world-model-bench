# megaminx-solver

[![Prime Hub](https://img.shields.io/badge/Prime%20Hub-setrf%2Fmegaminx--solver-blue)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Smoke eval](https://img.shields.io/badge/Smoke%20eval-v0.2.17%20match%20table-16a34a)](https://app.primeintellect.ai/dashboard/evaluations/k16letr5pyhopqwu0e5seshs)

Trainable multi-turn Megaminx environment for Prime Intellect Verifiers. The
agent receives a compact text facelet net, calls tools to rotate or inspect the
puzzle, and is rewarded for solving short deterministic scrambles efficiently.

## Overview

- **Environment ID**: `megaminx-solver`
- **Hub target**: `setrf/megaminx-solver`
- **Type**: multi-turn, stateful tool environment
- **Tags**: `multi-turn`, `tool-env`, `spatial-reasoning`, `megaminx`, `rl`, `train`, `eval`

## Task

Each rollout starts from a generated Megaminx scramble. Faces are labeled `A`
through `L`; a solved face contains only its own label. The model can call:

- `rotate(face, direction)`: rotate a face clockwise or counterclockwise.
- `inspect(face)`: inspect one face or `all`.
- `finish()`: end the rollout.

The inverse scramble is stored in task metadata for verification and is not
shown in the prompt.

## Environment Arguments

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `split` | str | `"train"` | `depth1`, `train_depth1`, `easy`, `medium`, `hard`, `eval`, or `eval_*` bands. |
| `min_depth` | int | `1` | Minimum scramble depth for custom splits. |
| `max_depth` | int | `8` | Maximum scramble depth for custom splits. |
| `num_examples` | int | `200` | Number of generated examples. |
| `seed` | int | `42` | Deterministic scramble seed. |
| `max_turns` | int or null | `null` | Override per-task move budget and global turn cap. |
| `reward_style` | str | `"dense"` | `dense`, `action_gated_dense`, `action_gated_curriculum`, or `action_gated_overlap`. |
| `prompt_style` | str | `"default"` | `default`, `action_first`, JSON action styles, or native tool styles. |
| `allow_text_tool_actions` | bool or null | `null` | Auto-enables JSON text fallback for JSON prompt styles and disables it for native tool styles unless overridden. |

Default move budget is `2 * scramble_depth + 4`, capped at `32`. The
environment-level turn cap is also reduced for shallow curricula so first RL
runs do not spend most tokens inspecting.

## Reward

The main scalar reward is:

```text
0.60 * solved
+ 0.25 * sticker_accuracy
+ 0.10 * piece_accuracy
+ 0.05 * efficiency_if_solved
```

For the current hosted RL path, use `reward_style="action_gated_curriculum"`
with `prompt_style="sensor_match_json_action"` and `max_turns=2` for depth-1
training. This keeps no-rotate rollouts at `0.0`, gives solved states `1.0`,
and gives valid non-solving rotations shaped first-move credit weighted toward
correct face selection before direction. The sensor-match prompt appends all 24
legal rotate JSON actions, derived affected faces, affected face summaries,
static neighbor rings, and static sorted candidate neighbor sets for all faces.
It does not print precomputed overlap counts or answer candidates; the model
must match affected faces to the static table and infer direction from sticker
flow.
For a smoother first on-ramp, `reward_style="action_gated_overlap"` rewards how
much the chosen face's neighbor ring overlaps the initially affected faces,
then adds smaller credit for exact face/direction correctness.
`action_gated_dense` remains available for stricter evals where only positive
state progress should score.

JSON action prompt styles intentionally allow a text JSON fallback because
current Qwen chat-completions evals often place tool calls in text or reasoning
fields. Native prompt styles (`native_action`, `topology_native_tool`,
`sensor_native_tool`, `sensor_match_native_tool`) keep that fallback disabled by
default for renderer/TITO-compatible trajectories.

Tracked metrics include `illegal_move_count`, `move_count`, `solved_rate`,
`scramble_depth`, `tool_call_count`, `action_taken`, `first_rotate_correct`,
`first_rotate_face_correct`, `first_rotate_direction_correct`,
`first_rotate_neighbor_overlap`, `reward_style`, `initial_sticker_accuracy`,
`initial_piece_accuracy`, and Verifiers' built-in tool metrics.

## Quickstart

```bash
prime env install setrf/megaminx-solver@0.2.17
prime eval run setrf/megaminx-solver -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"depth1","num_examples":120,"max_turns":2,"reward_style":"action_gated_curriculum","prompt_style":"sensor_match_json_action"}' \
  -n 120 -r 1 -t 48 -T 0.7 -A
```

Evaluate an explicit band:

```bash
prime eval run setrf/megaminx-solver -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"easy","num_examples":30,"max_turns":8,"reward_style":"action_gated_curriculum","prompt_style":"sensor_match_json_action"}' \
  -n 30 -r 1 -t 64 -T 0.7 -A
```

Run the depth-1 on-ramp intended for the first RL pass:

```bash
prime eval run setrf/megaminx-solver -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"depth1","num_examples":120,"max_turns":2,"reward_style":"action_gated_curriculum","prompt_style":"sensor_match_json_action"}' \
  -n 120 -r 1 -t 48 -T 0.7 -A
```

Run the action-gated RL smoke eval:

```bash
prime eval run setrf/megaminx-solver -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"depth1","num_examples":120,"max_turns":2,"reward_style":"action_gated_curriculum","prompt_style":"sensor_match_json_action"}' \
  -n 120 -r 1 -t 48 -T 0.7 -A
```

Push publicly after validation:

```bash
prime env push megaminx-solver --visibility PUBLIC
```
