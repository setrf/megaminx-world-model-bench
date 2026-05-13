# megaminx-solver

[![Prime Hub](https://img.shields.io/badge/Prime%20Hub-setrf%2Fmegaminx--solver-blue)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Latest eval](https://img.shields.io/badge/Latest%20eval-v0.2.20%20stage%20direction-16a34a)](https://app.primeintellect.ai/dashboard/evaluations/y78em945ey7fb8ijmxbej5mg)
[![Best baseline](https://img.shields.io/badge/Best%20baseline-v0.2.17%20match%20table-0f766e)](https://app.primeintellect.ai/dashboard/evaluations/k16letr5pyhopqwu0e5seshs)
[![Indexed eval](https://img.shields.io/badge/Indexed%20eval-v0.2.18-2563eb)](https://app.primeintellect.ai/dashboard/evaluations/gnudftur8j76ndbokfpas7jq)

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
| `max_turns` | int or null | `null` | Override the Verifiers turn cap. Also sets move budget unless `move_budget` is provided. |
| `move_budget` | int or null | `null` | Optional puzzle-rotation budget shown in the prompt and enforced by the environment. |
| `reward_style` | str | `"dense"` | `dense`, `action_gated_dense`, `action_gated_curriculum`, `action_gated_overlap`, `action_gated_direction`, or `action_gated_exact_direction`. |
| `prompt_style` | str | `"default"` | `default`, `action_first`, JSON action styles, staged JSON styles, or native tool styles. |
| `allow_text_tool_actions` | bool or null | `null` | Auto-enables JSON text fallback for JSON prompt styles and disables it for native tool styles unless overridden. |

Default move budget is `2 * scramble_depth + 4`, capped at `32`. The
environment-level turn cap is also reduced for shallow curricula so first RL
runs do not spend most tokens inspecting. When overriding both values for custom
configs, keep `move_budget <= max_turns` so the prompt does not advertise more
rotations than the Verifiers turn cap can service.

## Reward

The main scalar reward is:

```text
0.60 * solved
+ 0.25 * sticker_accuracy
+ 0.10 * piece_accuracy
+ 0.05 * efficiency_if_solved
```

For the current v0.2.20 staged hosted RL path, use
`reward_style="action_gated_exact_direction"` with
`prompt_style="stage_face_hint_direction_json_action"`, `max_turns=2`, and
`move_budget=1` for depth-1 training. This keeps no-rotate rollouts at `0.0`,
gives solved states `1.0`, gives small credit for the right face with the wrong
direction, and does not reward right direction on the wrong face. This staged
prompt intentionally reveals the correct face for depth-1 scrambles and limits
the JSON action menu to that face's `cw`/`ccw` choices, so training isolates
direction learning. The latest held-out staged eval scored `0.579` reward with
`0.504` solved/direction accuracy, `1.000` face accuracy, and zero env errors or
illegal moves. It is not a final naturalistic eval prompt and rejects
non-depth-1 splits. The underlying candidate-strip prompt still appends derived
affected faces, affected face summaries, static neighbor rings, static sorted
candidate neighbor sets, an index guide for edge/corner sticker flow, and
candidate-local moved strips. It does not print precomputed overlap counts,
answer candidates, the scramble, or the inverse solution.
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
`initial_piece_accuracy`, and Verifiers' built-in tool metrics. The
`reward_style` metric is encoded as `5.0` for `action_gated_exact_direction`.

## Quickstart

```bash
prime env install setrf/megaminx-solver@0.2.20
prime eval run setrf/megaminx-solver@0.2.20 -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"eval_depth1","num_examples":240,"seed":1042,"max_turns":2,"move_budget":1,"reward_style":"action_gated_exact_direction","prompt_style":"stage_face_hint_direction_json_action","allow_text_tool_actions":true}' \
  -n 240 -r 1 -t 48 -T 0.7 -A
```

Latest validation: [v0.2.20 staged direction eval](https://app.primeintellect.ai/dashboard/evaluations/y78em945ey7fb8ijmxbej5mg)
used `eval_depth1`, 240 examples, seed `1042`, and produced reward `0.579`,
solved/direction `0.504`, face `1.000`, illegal moves `0`, and env errors `0`.

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
prime eval run setrf/megaminx-solver@0.2.20 -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"eval_depth1","num_examples":240,"seed":1042,"max_turns":2,"move_budget":1,"reward_style":"action_gated_exact_direction","prompt_style":"stage_face_hint_direction_json_action","allow_text_tool_actions":true}' \
  -n 240 -r 1 -t 48 -T 0.7 -A
```

Historical v0.2.19 candidate-strip smoke eval:

```bash
prime eval run setrf/megaminx-solver -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"depth1","num_examples":120,"max_turns":2,"move_budget":1,"reward_style":"action_gated_exact_direction","prompt_style":"sensor_candidate_strips_json_action","allow_text_tool_actions":true}' \
  -n 120 -r 1 -t 48 -T 0.7 -A
```

Push publicly after validation:

```bash
prime env push megaminx-solver --visibility PUBLIC
```
