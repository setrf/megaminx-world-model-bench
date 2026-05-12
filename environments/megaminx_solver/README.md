# megaminx-solver

[![Prime Hub](https://img.shields.io/badge/Prime%20Hub-setrf%2Fmegaminx--solver-blue)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Smoke eval](https://img.shields.io/badge/Smoke%20eval-action--gated%204B-7c3aed)](https://app.primeintellect.ai/dashboard/evaluations/dq7ajgaiw0vdi6fz284a8ulp)

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
| `reward_style` | str | `"dense"` | `dense` or `action_gated_dense`. |
| `prompt_style` | str | `"default"` | `default` or `action_first`. |

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

For RL runs, use `reward_style="action_gated_dense"` with
`prompt_style="action_first"`. This keeps no-rotate rollouts at `0.0`, gives
solved states `1.0`, and gives valid non-solving rotations only positive
progress-delta reward.

Tracked metrics include `illegal_move_count`, `move_count`, `solved_rate`,
`scramble_depth`, `tool_call_count`, `action_taken`, `first_rotate_correct`,
`first_rotate_face_correct`, `first_rotate_direction_correct`, `reward_style`,
`initial_sticker_accuracy`, `initial_piece_accuracy`, and Verifiers' built-in
tool metrics.

## Quickstart

```bash
prime env install setrf/megaminx-solver
prime eval run setrf/megaminx-solver -m Qwen/Qwen3.5-4B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":"required"}' \
  -a '{"split":"depth1","num_examples":30,"max_turns":2,"reward_style":"action_gated_dense","prompt_style":"action_first"}' \
  -n 10 -r 3 -t 256 -A
```

Evaluate an explicit band:

```bash
prime eval run setrf/megaminx-solver -m Qwen/Qwen3.5-4B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":"required"}' \
  -a '{"split":"easy","num_examples":30,"reward_style":"action_gated_dense","prompt_style":"action_first"}' \
  -n 30 -r 3 -t 256 -A
```

Run the depth-1 on-ramp intended for the first RL pass:

```bash
prime eval run setrf/megaminx-solver -m Qwen/Qwen3.5-4B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":"required"}' \
  -a '{"split":"depth1","num_examples":30,"max_turns":2,"reward_style":"action_gated_dense","prompt_style":"action_first"}' \
  -n 10 -r 3 -t 256 -A
```

Run the action-gated RL smoke eval:

```bash
prime eval run setrf/megaminx-solver -m Qwen/Qwen3.5-4B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":"required"}' \
  -a '{"split":"depth1","num_examples":30,"max_turns":2,"reward_style":"action_gated_dense","prompt_style":"action_first"}' \
  -n 10 -r 3 -t 256 -A
```

Push publicly after validation:

```bash
prime env push megaminx-solver --visibility PUBLIC
```
