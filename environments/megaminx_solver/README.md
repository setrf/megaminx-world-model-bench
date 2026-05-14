# megaminx-solver

[![Prime Hub](https://img.shields.io/badge/Prime%20Hub-setrf%2Fmegaminx--solver-blue)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Latest env](https://img.shields.io/badge/env-v0.2.57-0f766e)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Breakthrough eval](https://img.shields.io/badge/v0.2.22%2035B-0.929-2563eb)](https://app.primeintellect.ai/dashboard/evaluations/vse6uoo8c9y156svyyv41qll)
[![GitHub](https://img.shields.io/badge/GitHub-setrf%2Fmegaminx--world--model--bench-black)](https://github.com/setrf/megaminx-world-model-bench)

`megaminx-solver` is a trainable Prime Intellect / Verifiers environment for
studying whether language models can learn a physical puzzle world through
state, tool use, reward, and repeated rollouts.

The environment simulates a symbolic Megaminx, a twelve-faced twisty puzzle.
Models observe compact facelet or sensor text, act through native tools, and
receive deterministic rewards from a persistent simulator. The current release
focuses on a hardened depth-2 candidate-path curriculum for native Hosted
Training and an oracle trajectory exporter for SFT warm starts.

## What This Environment Tests

This is not a trivia benchmark. The model must interact with a world whose
state changes after every action.

The environment tests:

- **Physical state tracking:** stickers move when a face turns, and future
  observations depend on earlier actions.
- **Local geometry:** each face turn affects one pentagonal face and a ring of
  adjacent strips.
- **Tool grounding:** the answer is not a text string; the model must call the
  right tool with valid arguments.
- **Planning horizon:** depth-1 tasks test first-action grounding; depth-2
  tasks require acting, reading the refreshed world state, and acting again.
- **Protocol discipline:** native tool calls, text fallback actions, malformed
  calls, and protocol violations are measured separately.
- **Shortcut resistance:** hidden scramble metadata is never printed, and later
  releases remove visible masks, row ids, fixed slots, and other accidental
  hints found during audits.

## Rollout Contract

Every rollout follows this contract:

1. A deterministic scramble is generated from `split`, `seed`, and example
   index.
2. The simulator applies the scramble to a solved Megaminx.
3. The inverse solution is stored in task metadata for scoring, tests, and
   metrics, but not shown to the model.
4. The prompt prints a compact observation or staged sensor/candidate view.
5. The model acts through native tools or, only for configured JSON prompt
   styles, a text action fallback.
6. The environment mutates the same puzzle state after each action.
7. The rubric scores the final trajectory and emits diagnostic metrics.

The implementation class is `MegaminxEnv(vf.StatefulToolEnv)`, so rollout state
is persistent rather than reconstructed from text.

## Observations

The base observation includes:

- solved/not-solved status
- sticker accuracy
- piece accuracy
- move budget
- last move
- compact face lines for faces `A` through `L`

A solved face contains only its own label. Centers are fixed and identify the
target color for a face. Edge and corner strings show the sticker labels
currently visible on that face.

Staged prompt styles may replace or augment the full net with sensor tables,
direction-flow evidence, candidate faces, candidate-relative flow tokens, or a
refreshed candidate table after the first action.

Prompts intentionally do not expose:

- scramble moves
- inverse solution
- hidden answer
- row-derived ids in the current v0.2.56 candidate-path lane
- hidden candidate-slot seeds

## Tools

General puzzle tools:

| Tool | Arguments | Meaning |
| --- | --- | --- |
| `rotate` | `face: A-L`, `direction: cw|ccw` | Apply one legal Megaminx face turn. |
| `inspect` | `face: A-L|all` | Return one face or the full compact net without changing state. |
| `finish` | none | End the rollout and report the final state. |

Staged tools used by research lanes:

| Tool | Arguments | Meaning |
| --- | --- | --- |
| `select_candidate` | `index: 1-4`, `direction: cw|ccw` | Rotate the face shown in a visible candidate slot. Current depth-2 lanes call this twice. |
| `select_candidate_index` | `index: 1-4` | Select a candidate face without choosing direction. Used for face-discovery ablations. |
| `predict_rotate` | `face`, `direction`, `predicted_after` | Predict five local post-move strips, then apply the move. |

Candidate prompt styles expose only their matching candidate tool by default.
For example, `stage_candidate_relative_flow_rule_solve2_native_tool` exposes
`select_candidate` rather than the full general tool set.

Important schema detail: `MegaminxEnv` implements all six tools. With
`exposed_tool_names=None`, ordinary non-candidate prompt styles expose all six
implemented tools even though the prompt asks the model to use the general
`rotate`/`inspect`/`finish` loop. Candidate prompt styles automatically narrow
the schema to `select_candidate` or `select_candidate_index`.

## `load_environment` API

```python
from megaminx_solver import load_environment

env = load_environment(
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
    exposed_tool_names=None,
)
```

| Argument | Default | Description |
| --- | --- | --- |
| `split` | `"train"` | Named curriculum split. |
| `min_depth` | `1` | Minimum scramble depth for custom splits. |
| `max_depth` | `8` | Maximum scramble depth for custom splits. |
| `num_examples` | `200` | Deterministic dataset size. |
| `seed` | `42` | Dataset seed. |
| `max_turns` | `None` | Global Verifiers turn cap. |
| `move_budget` | `None` | Puzzle move budget shown to the model. |
| `reward_style` | `"dense"` | Rubric/reward family. |
| `prompt_style` | `"default"` | Observation and tool protocol family. |
| `allow_text_tool_actions` | `None` | Enables JSON/private-text action parsing for JSON prompt styles. |
| `exposed_tool_names` | `None` | Optional explicit tool subset. |

By default, `allow_text_tool_actions` is true only for JSON prompt styles and
false for native Hosted Training styles.

Public package exports:

```text
MegaminxEnv
build_dataset
load_environment
MegaminxPuzzle
MegaminxTopology
FACES
POSITIONS_PER_FACE
STICKERS_PER_PUZZLE
EDGE_COUNT
CORNER_COUNT
generate_scramble
inverse_moves
```

## Splits And Curricula

| Split | Depths | Purpose |
| --- | ---: | --- |
| `depth1`, `train_depth1`, `eval_depth1` | 1 | One-turn action grounding. |
| `easy`, `train_easy`, `eval_easy` | 1-3 | Short-scramble generalization. |
| `medium`, `train_medium`, `eval_medium` | 4-6 | Mid-horizon evaluation. |
| `hard`, `train_hard`, `eval_hard` | 7-10 | Hard evaluation. |
| `eval` | 1-10 | Broad mixed-depth evaluation. |
| `train` | custom, default 1-8 | General curriculum. |
| `train_candidate_relative_flow_rule_tail_solve_depth2` | 2 | Current v0.2.56 oracle/SFT lane. |

For ordinary rollouts, the default move budget is `min(32, 2 * depth + 4)`.
Staged one-action and two-action curricula often override this with a smaller
`move_budget` to make the protocol exact.

## Simulator Guarantees

The simulator tracks the Megaminx as facelet state:

- 12 faces: `A` through `L`
- 132 visible stickers
- 12 fixed centers
- 30 edge pieces
- 20 corner pieces
- 24 legal moves: each face clockwise or counterclockwise
- five clockwise turns of any face return to identity
- clockwise followed by counterclockwise returns to identity
- every move preserves the sticker multiset
- generated scrambles plus their inverse solve the puzzle

The dodecahedron topology is represented through face-neighbor rings and
programmatic side strips. The tests assert the topology and move invariants so
reward bugs cannot silently become puzzle-physics bugs.

## Reward Styles

The default dense reward is:

```text
0.60 * solved
+ 0.25 * sticker_accuracy
+ 0.10 * piece_accuracy
+ 0.05 * efficiency_if_solved
```

RL-facing rewards are action-gated so tool-free reasoning does not score well.
The important families are:

| Reward style | Use |
| --- | --- |
| `dense` | General multi-turn simulator reward. |
| `action_gated_dense` | Dense reward capped at zero until an action is taken. |
| `action_gated_binary_direction` | Strict depth-1 eval: exactly one clean inverse `rotate` scores `1.0`. |
| `action_gated_strict_shaped_direction` | Depth-1 training reward with partial credit for learnable wrong-but-informative actions. |
| `action_gated_candidate_geometry_frontier` | Candidate selection using visible affected geometry. |
| `action_gated_candidate_strict_frontier` | Candidate-relative flow reward for one-turn-frontier progress. |
| `action_gated_candidate_path_solve` | Two-call candidate path that rewards solving after refreshed observation. |
| `action_gated_candidate_path_tail_solve` | Current v0.2.56 hardened two-call reward. |

v0.2.56 caps non-solving second-step tail reward below `0.50`, so a rollout
must actually solve to receive a high score.

Full implemented reward-style registry:

```text
dense
action_gated_dense
action_gated_curriculum
action_gated_overlap
action_gated_direction
action_gated_exact_direction
action_gated_binary_direction
action_gated_strict_shaped_direction
action_gated_overlap_strict_shaped_direction
action_gated_mask_overlap_strict_shaped_direction
action_gated_counterfactual_frontier_strict
action_gated_counterfactual_frontier_value_strict
action_gated_predict_rotate_value_strict
action_gated_predict_rotate_transition
action_gated_face_discovery
action_gated_face_tournament
action_gated_candidate_tournament
action_gated_candidate_index
action_gated_candidate_mask_index_rank
action_gated_candidate_mask_frontier_equivalence
action_gated_candidate_geometry_frontier
action_gated_candidate_strict_frontier
action_gated_candidate_path_solve
action_gated_candidate_path_tail_solve
```

## Prompt Styles

Representative prompt styles:

| Prompt style | Tool surface | Description |
| --- | --- | --- |
| `default` | `rotate`, `inspect`, `finish` | General text Megaminx puzzle. |
| `action_first` | `rotate`, `inspect`, `finish` | General puzzle with stronger instruction to act before explanation. |
| `stage_solve_direction_flow_json_action` | JSON/text `rotate` fallback | Served eval compatibility lane. |
| `stage_solve_direction_flow_native_tool` | native `rotate` | Depth-1 Hosted Training native tool lane. |
| `stage_solve_direction_flow_native_tool_v2` | native `rotate` | Depth 1-2 all-candidate solve-flow lane. |
| `stage_candidate_geometry_frontier_native_tool` | native `select_candidate` | Clean candidate lane built from visible affected geometry. |
| `stage_candidate_relative_flow_rule_frontier_native_tool` | native `select_candidate` | Candidate-relative flow with explicit counting rule. |
| `stage_candidate_relative_flow_rule_solve2_native_tool` | native `select_candidate` twice | Current depth-2 path: choose candidate, observe refreshed table, choose again. |
| `stage_predict_rotate_native_tool` | native `predict_rotate` | Predict local strips before rotating. |

Older prompt styles remain available for ablations and reproducibility, but the
current release should be evaluated through the v0.2.56 candidate-path lane.

Full implemented prompt-style registry:

```text
default
action_first
direct_json_action
choice_json_action
topology_choice_json_action
sensor_choice_json_action
sensor_match_json_action
sensor_indexed_match_json_action
sensor_candidate_strips_json_action
stage_face_hint_direction_json_action
stage_direction_flow_json_action
stage_direction_flow_reasoned_json_action
stage_solve_direction_flow_json_action
native_action
topology_native_tool
sensor_native_tool
sensor_match_native_tool
stage_direction_flow_native_tool
stage_solve_direction_flow_native_tool
stage_solve_direction_flow_native_tool_v2
stage_solve_action_table_native_tool
stage_solve_action_mask_native_tool
stage_frontier_sensor_native_tool
stage_frontier_sensor_compact_native_tool
stage_predict_rotate_native_tool
stage_predict_transition_native_tool
stage_face_discovery_native_tool
stage_face_tournament_native_tool
stage_candidate_tournament_native_tool
stage_candidate_index_native_tool
stage_candidate_scorecard_native_tool
stage_candidate_scorecard_no_frontier_native_tool
stage_candidate_scorecard_mask_native_tool
stage_candidate_scorecard_mask_index_native_tool
stage_candidate_scorecard_mask_frontier_equivalence_native_tool
stage_candidate_geometry_frontier_native_tool
stage_candidate_relative_flow_frontier_native_tool
stage_candidate_relative_flow_rule_frontier_native_tool
stage_candidate_relative_flow_rule_solve2_native_tool
```

## Protocol Lanes

There are two distinct protocol lanes:

| Lane | Use | Tool handling |
| --- | --- | --- |
| Native Hosted Training | Credible RL measurement | Model emits native `tool_calls`; text before a tool call is a protocol violation. |
| Served eval compatibility | Historical/scaling evals | JSON/private-text fallback can be parsed into tool actions when native calls are unreliable. |

These lanes should not be mixed when making claims. Native tool-call metrics
are the trusted RL surface.

## Metrics

The environment reports outcome, action, protocol, and diagnostic metrics:

| Group | Examples |
| --- | --- |
| Outcome | `solved_rate`, `sticker_accuracy`, `piece_accuracy`, `move_count`, `scramble_depth` |
| Protocol | `native_tool_call_count`, `text_tool_action_count`, `private_text_action_count`, `tool_parse_error_count`, `tool_call_error_count`, `protocol_violation_count` |
| Tool counts | `rotate_call_count`, `candidate_select_call_count`, `predict_rotate_call_count`, `inspect_call_count`, `finish_call_count` |
| First action | `first_rotate_correct`, `first_rotate_face_correct`, `first_rotate_direction_correct`, `first_rotate_neighbor_overlap` |
| Candidate path | `target_face_in_candidate_set`, `target_candidate_index`, `second_target_candidate_index`, `candidate_path_completed` |
| Relative flow | `first_candidate_relative_flow_count`, `first_candidate_relative_flow_margin`, `first_candidate_relative_flow_is_candidate_max`, second-step variants |
| Prediction | `first_prediction_strip_accuracy`, `first_prediction_exact_strip_count`, `first_prediction_char_accuracy`, `first_prediction_valid` |

These metrics make failures debuggable. For example, a run can have nonzero
reward but fail because the model called text JSON instead of native tools,
chose the right face with the wrong direction, selected a visible shortcut, or
improved online reward without improving heldout solves.

The rubric currently includes the primary reward plus sticker, piece, and
efficiency functions, followed by 70 zero-weight diagnostic metric functions.
The exact metric registry lives in `_build_rubric(...)` in
`megaminx_solver.py`; the groups above are the stable public reading guide.

## Determinism And Hidden Metadata

Task metadata stores:

- scramble
- inverse solution
- split
- example id
- reward style
- prompt style
- hidden candidate seeds

This metadata is used by tests, rewards, oracle export, and metrics. It is not
printed in the current prompts. v0.2.56 specifically removes row-id leakage
from the visible prompt and derives refreshed second-step candidate slots from
hidden scramble/action metadata.

The v0.2.57 1,024-row oracle export is byte-stable when re-run with the same
arguments. Recorded SHA256:

```text
729d03b69c56d2c9e1775d452cf20abed05fd5e142b0eb4db737ff11187247f5
```

## Leakage And Shortcut Controls

This environment went through multiple shortcut audits. Important fixes:

- removed direct answer, scramble, inverse solution, winner, and scalar support
  fields from prompts
- separated native Hosted Training from served JSON fallback evals
- removed public `cw_mask` / `ccw_mask` reward proxies from the clean geometry
  lane
- stopped seeding candidate slots directly from hidden solution metadata in the
  clean lane
- balanced second-step candidate target slots
- removed visible example-id dependence from refreshed second-step slots
- capped high reward for non-solving second-step tail actions
- added oracle and SFT JSONL validators to check for forbidden payload fields

Historical scaffolded runs are kept for reproducibility, but v0.2.56 is the
current hardened baseline.

## Current Evidence

| Claim | Evidence |
| --- | --- |
| Package pushed | `setrf/megaminx-solver@0.2.57`, Hub hash `35d4bb90de33`, wheel SHA256 `744af86fdf6e0effc8ae54f1f3f08c7aa8227c06bd9bb7cd58cd56c8df48b40a` |
| Local tests | `uv run pytest -q` recorded `129 passed in 45.50s` |
| Prompt breakthrough | v0.2.22 35B eval solved `0.929` on 240 heldout examples: <https://app.primeintellect.ai/dashboard/evaluations/vse6uoo8c9y156svyyv41qll> |
| Clean native RL lane | v0.2.54 two-call candidate-path run reached online reward `0.7335`, solved `0.6615`, two native calls, zero env errors |
| Best matched hosted heldout gain | v0.2.54 checkpoint improved solved from `0.5625` to `0.6048` on one heldout seed |
| Local SFT warm-start signal | `Qwen/Qwen3.5-0.8B` local adapter improved from `0/32` to `18/32` heldout solves |
| Not yet solved | Original `+30pp` hosted RL target was not reached before Prime billing/auth blocked further runs |

The full evidence trail is in the GitHub report:
<https://github.com/setrf/megaminx-world-model-bench/blob/main/reports/megaminx-rl-report.md>.

## Usage Recipes

Install the environment:

```bash
prime env install setrf/megaminx-solver@0.2.57 --plain
```

No environment variables are required to import or load the environment. Hosted
evals and training still require the normal Prime/model-provider credentials.

From a checkout of the GitHub repository, smoke local tests:

```bash
uv run pytest -q
```

From a checkout of the GitHub repository, create the current oracle dataset:

```bash
uv run python scripts/export_oracle_trajectories.py \
  --num-examples 1024 \
  --seed 64 \
  --split train_candidate_relative_flow_rule_tail_solve_depth2 \
  --output /tmp/megaminx-oracle-v057-1024.jsonl
```

Convert and validate SFT data:

```bash
uv run python scripts/convert_oracle_to_sft_jsonl.py \
  /tmp/megaminx-oracle-v057-1024.jsonl \
  --output /tmp/megaminx-oracle-v057-1024-sft.jsonl

uv run python scripts/validate_sft_jsonl.py \
  /tmp/megaminx-oracle-v057-1024-sft.jsonl
```

Check readiness before hosted probes:

```bash
uv run python scripts/check_next_run_readiness.py
```

The next hosted probe sequence is documented at:
<https://github.com/setrf/megaminx-world-model-bench/blob/main/reports/megaminx-next-run-runbook.md>.

## Known Caveats

- The environment is text-first. It models physical structure symbolically; it
  does not yet render images or vision observations.
- Some historical runs used scaffolded prompts to discover a learnable
  curriculum. Use v0.2.56 for the current hardened lane.
- Native Hosted Training and served evals measure different protocol surfaces.
- Prime now reports the Hub environment as `PUBLIC`.
- Hosted run creation later hit billing/auth limits, so the prepared v0.2.56
  matched heldout probes remain to be run.

## Version Notes

| Version | Main change |
| --- | --- |
| v0.2.21 | Direction-flow depth-1 eval exposed the face/direction inversion trap. |
| v0.2.22 | Candidate solving-action framing reached `0.929` solved with 35B on heldout depth-1. |
| v0.2.28 | Native Hosted Training/TITO lane produced clean native tool-call probes. |
| v0.2.47 | Candidate mask frontier-equivalence lane showed native checkpoint movement but exposed scaffold risk. |
| v0.2.50 | Clean geometry candidate lane removed public mask reward proxies. |
| v0.2.52 | Candidate-relative flow tokens replaced heavier scorecards. |
| v0.2.53 | Rule-flow prompt added an explicit fair counting rule. |
| v0.2.54 | Two-call depth-2 candidate path solved after refreshed observations. |
| v0.2.55 | Tail-solve reward hardening and balanced second-step targets. |
| v0.2.56 | Visible-id shortcut fix, second-step reward cap, deterministic oracle export, SFT warm-start path. |
| v0.2.57 | Documentation and package metadata refresh for GitHub and Prime Hub. |
