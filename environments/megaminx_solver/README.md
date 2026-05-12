# megaminx-solver

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
| `split` | str | `"train"` | `train`, `eval`, `easy`, `medium`, `hard`, or `eval_*` bands. |
| `min_depth` | int | `1` | Minimum scramble depth for custom splits. |
| `max_depth` | int | `8` | Maximum scramble depth for custom splits. |
| `num_examples` | int | `200` | Number of generated examples. |
| `seed` | int | `42` | Deterministic scramble seed. |
| `max_turns` | int or null | `null` | Override per-task move budget and global turn cap. |

Default move budget is `2 * scramble_depth + 4`, capped at `32`.

## Reward

The main scalar reward is:

```text
0.60 * solved
+ 0.25 * sticker_accuracy
+ 0.10 * piece_accuracy
+ 0.05 * efficiency_if_solved
```

Tracked metrics include `illegal_move_count`, `move_count`, `solved_rate`,
`scramble_depth`, `tool_call_count`, and Verifiers' built-in tool metrics.

## Quickstart

```bash
prime env install megaminx-solver
prime eval run megaminx-solver -m Qwen/Qwen3.5-4B -n 10 -r 3
```

Evaluate an explicit band:

```bash
prime eval run megaminx-solver -m Qwen/Qwen3.5-4B -a '{"split":"easy","num_examples":30}' -n 30 -r 3
```

Push privately after validation:

```bash
prime env push megaminx-solver --visibility PRIVATE
```
