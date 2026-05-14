# megaminx-solver

[![Prime Hub](https://img.shields.io/badge/Prime%20Hub-setrf%2Fmegaminx--solver-blue)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Latest env](https://img.shields.io/badge/env-v0.2.54-0f766e)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Clean geometry](https://img.shields.io/badge/clean%20geometry-0.303-16a34a)](https://app.primeintellect.ai/dashboard/training/dbup76z9d460x4fbcfxw8yql)
[![Breakthrough eval](https://img.shields.io/badge/v0.2.22%2035B-0.929-2563eb)](https://app.primeintellect.ai/dashboard/evaluations/vse6uoo8c9y156svyyv41qll)

Trainable multi-turn Megaminx environment for Prime Intellect Verifiers. The
model sees a compact text facelet net, acts through tools, and receives a
deterministic reward from a persistent puzzle simulator.

## Overview

| Item | Value |
| --- | --- |
| Environment id | `megaminx-solver` |
| Hub target | `setrf/megaminx-solver` |
| Current package | `setrf/megaminx-solver@0.2.54` |
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
- `action_gated_candidate_mask_frontier_equivalence`
- `action_gated_candidate_geometry_frontier`
- `action_gated_candidate_strict_frontier`

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
- `stage_solve_direction_flow_native_tool_v2`: native all-candidate
  solve-flow lane. It removes the explicit face hint, supports depths 1-2, and
  is the current continuation path for larger RL gains.
- `stage_solve_direction_flow_json_action`: served chat-completions eval lane.
  Use `allow_text_tool_actions=true` for Qwen JSON/private text fallback.
- `stage_candidate_scorecard_mask_frontier_equivalence_native_tool`: v0.2.47
  native candidate-mask lane. It uses the same public scorecard columns as the
  v0.2.46 mask scorecard and rewards clean solved actions or depth-2
  one-turn-frontier-equivalent actions.
- `stage_candidate_geometry_frontier_native_tool`: v0.2.50 clean native
  candidate lane. It removes printed `cw_mask`/`ccw_mask` reward proxies and
  builds the four candidate slots from visible affected geometry instead of
  injecting the hidden solution face.
- `stage_candidate_relative_flow_frontier_native_tool`: v0.2.52 native
  candidate lane. It prints a coordinate transform of visible strips into
  candidate-relative flow tokens and pairs with the stricter frontier reward.
- `stage_candidate_relative_flow_rule_frontier_native_tool`: v0.2.53 native
  candidate lane. It keeps the same visible flow tokens and adds a fair
  counting rule for selecting the largest `+1` or `-1` evidence.
- `stage_candidate_relative_flow_rule_solve2_native_tool`: v0.2.54 native
  two-call candidate-path lane. After each `select_candidate` call, the
  environment refreshes the visible candidate relative-flow table so a depth-2
  rollout can solve instead of only choosing a one-turn-frontier first move.
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
target_face_in_candidate_set
target_candidate_index
first_candidate_relative_flow_count
first_candidate_relative_flow_margin
first_candidate_relative_flow_is_candidate_max
candidate_relative_flow_oracle_unique
first_rotate_direction_correct
first_rotate_face_id
first_rotate_direction_id
first_rotate_neighbor_overlap
reward_style
initial_sticker_accuracy
initial_piece_accuracy
```

v0.2.29 hardens the private reasoning path: structured non-action private
metadata such as reasoning summaries is ignored, while malformed private JSON
action attempts are counted as private parse errors.

v0.2.30 adds the native-v2 all-candidate prompt and action-id metrics. The
native-v2 prompt is restricted to depth 1-2 because it is intended as a
first-move curriculum before broader `easy` training.

v0.2.46 adds `stage_candidate_scorecard_mask_native_tool`, the current
least-scaffolded scorecard lane: four candidate faces, affected-neighbor counts,
and direction masks, with scalar `support`, `frontier`, inverse solution, and
answer fields removed from the prompt.

v0.2.47 adds `stage_candidate_scorecard_mask_frontier_equivalence_native_tool`.
It preserves the v0.2.46 public scorecard columns and `select_candidate`
surface, but `action_gated_candidate_mask_frontier_equivalence` gives high
credit to any clean depth-2 first move that leaves the puzzle one turn from
solved.

v0.2.50 added `stage_candidate_geometry_frontier_native_tool`, the first clean
candidate hosted-RL lane. It keeps native `select_candidate(index, direction)` calls, but
removes the public mask scorecard and avoids seeding candidate slots from hidden
solution metadata.

v0.2.51 adds `target_face_in_candidate_set`, a diagnostic metric for the clean
geometry-frontier lane. The diagnostic Qwen 9B baseline
[`g8egyymgds47wtb4rbieyd20`](https://app.primeintellect.ai/dashboard/training/g8egyymgds47wtb4rbieyd20)
reported target availability `1.0`, reward `0.3241`, solved `0.0622`, native
tool calls `1.0`, and zero protocol/illegal-move errors.

v0.2.52 adds `stage_candidate_relative_flow_frontier_native_tool`, the
`action_gated_candidate_strict_frontier` reward, compact final tool outputs for
one-shot candidate rollouts, and immutable native tool argument handling. It was
pushed with wheel SHA
`2178fc527952570a84d28e7d4917f64a795b0bb58a40c4cc606fddba16bf156a`.

v0.2.53 adds `stage_candidate_relative_flow_rule_frontier_native_tool`, a fair
counting-rule prompt over the same visible `+1/-1` flow tokens. It was pushed
with wheel SHA
`3e2815bb5fa1e28dfeb3501ed65b67de2dfcf3c2fc1b2e14278eef590eb43d95`.

v0.2.54 adds `stage_candidate_relative_flow_rule_solve2_native_tool` and
`action_gated_candidate_path_solve`, the first clean two-call candidate path
lane. It was pushed with wheel SHA
`9a1edd7195f08a516596efd772ef8729149b1332d7bc885e090eb52613289748`.

## Validation

Install the current Hub package:

```bash
prime env install setrf/megaminx-solver@0.2.54 --plain
```

Run local tests:

```bash
uv run pytest -q
```

Latest result: `102 passed in 7.65s`.

Install and test the latest package:

```bash
prime env install setrf/megaminx-solver@0.2.54 --plain
uv run pytest -q
```

Run the current clean rule-flow baseline and training config:

```bash
prime train configs/rl/megaminx-v049-native-candidate-geometry-frontier-depth12-base-qwen9b-rpe2-fast.toml --yes --plain
prime train configs/rl/megaminx-v049-qwen9b-candidate-geometry-frontier-depth12-rpe16-noeval.toml --yes --plain
prime train configs/rl/megaminx-v051-native-candidate-geometry-frontier-depth12-base-qwen9b-rpe2-fast.toml --yes --plain
prime train configs/rl/megaminx-v051-qwen9b-candidate-geometry-frontier-depth12-lr1e8-b1024-rpe16-temp07.toml --yes --plain
prime train configs/rl/megaminx-v052-native-candidate-relative-flow-strict-frontier-depth12-base-qwen9b-rpe2-fast.toml --yes --plain
prime train configs/rl/megaminx-v052-qwen9b-candidate-relative-flow-strict-frontier-depth12-lr1e8-b1024-rpe16-temp07.toml --yes --plain
prime train configs/rl/megaminx-v053-native-candidate-relative-flow-rule-strict-frontier-depth12-base-qwen9b-rpe2-fast.toml --yes --plain
prime train configs/rl/megaminx-v053-qwen9b-candidate-relative-flow-rule-strict-frontier-depth12-lr5e9-b1024-rpe16-noeval.toml --yes --plain
prime train configs/rl/megaminx-v053-qwen9b-candidate-relative-flow-rule-strict-frontier-depth12-lr5e9-b1024-rpe16-complete2.toml --yes --plain
```

## Current Evidence

The v0.2.22 environment/prompt breakthrough moved 35B depth-1 direction solving
from [`0.504`](https://app.primeintellect.ai/dashboard/evaluations/dgu4mymqqk5sy97l3kvusjxu)
to [`0.929`](https://app.primeintellect.ai/dashboard/evaluations/vse6uoo8c9y156svyyv41qll).

The v0.2.44-v0.2.46 candidate-scorecard ablation shows the current bridge from
raw stickers toward learnable action selection:

| Probe | Scaffold | Reward | Face | Note |
| --- | --- | ---: | ---: | --- |
| [`bf37hb72kaqha4fp1cein28q`](https://app.primeintellect.ai/dashboard/training/bf37hb72kaqha4fp1cein28q) | support + frontier | `0.9277` | `0.9145` | cracked scaffold |
| [`qtkzn3gbi9alj12lbe7q6ijw`](https://app.primeintellect.ai/dashboard/training/qtkzn3gbi9alj12lbe7q6ijw) | support, no frontier | `0.8708` | `0.8498` | still strong |
| [`etecohz0kxjx0hwpj06aoevq`](https://app.primeintellect.ai/dashboard/training/etecohz0kxjx0hwpj06aoevq) | affected-neighbor counts + masks only | `0.7336` | `0.7034` | train gate passed |

An earlier v0.2.47 follow-up was
[`pirt9kurev8d0okydmbo309d`](https://app.primeintellect.ai/dashboard/training/pirt9kurev8d0okydmbo309d),
a stopped v0.2.47 frontier-equivalence train run. By step 4 it reached reward
`0.5693` and direction `0.6582` with zero errors, producing checkpoint
`ginimgkdx6okinz84klf0m98`, then drifted to `0.4865` by step 8. A heldout probe
[`cto44tv6sqbkynjp2g01ggpw`](https://app.primeintellect.ai/dashboard/training/cto44tv6sqbkynjp2g01ggpw)
measured the checkpoint at reward `0.5312`. The v0.2.47 baseline
[`v6p7exy9p8h4vbek7ujvj86c`](https://app.primeintellect.ai/dashboard/training/v6p7exy9p8h4vbek7ujvj86c)
reached reward `0.4620`, face `0.7817`, and action-frontier `0.1878`.
The v0.2.46 hosted RL run
[`hv6ljq5jlc8w391a0q38373l`](https://app.primeintellect.ai/dashboard/training/hv6ljq5jlc8w391a0q38373l)
was stopped at step 4 after an unstable train signal.

Leak audit: v0.2.47 is retained as scaffolded training history, not as a clean
Megaminx-understanding claim. v0.2.50 is the clean lane. Its base run
[`dbup76z9d460x4fbcfxw8yql`](https://app.primeintellect.ai/dashboard/training/dbup76z9d460x4fbcfxw8yql)
reached reward `0.3031`, solved `0.0508`, native tool calls `1.0`, and zero
errors. The clean hosted train run is
[`dbs7pcyih846945xubanvdjr`](https://app.primeintellect.ai/dashboard/training/dbs7pcyih846945xubanvdjr).
The v0.2.51 diagnostic baseline
[`g8egyymgds47wtb4rbieyd20`](https://app.primeintellect.ai/dashboard/training/g8egyymgds47wtb4rbieyd20)
confirmed the clean candidate set contains the inverse target face on every
sample (`target_face_in_candidate_set=1.0`). The conservative v0.2.51 Qwen 9B
training run is
[`byzwnn49pt9ztm6xiw0jumkx`](https://app.primeintellect.ai/dashboard/training/byzwnn49pt9ztm6xiw0jumkx).
It was stopped at step 4 because every row stayed below the v0.2.51 baseline
and the old reward could be beaten by a fixed slot/direction policy. The next
run is the v0.2.52 strict relative-flow baseline
[`iwozt5azyroqtwkfztd47e6s`](https://app.primeintellect.ai/dashboard/training/iwozt5azyroqtwkfztd47e6s),
which reached reward `0.5676`, solved `0.3048`, face `0.7677`, native tool
calls `1.0`, and zero errors. The main v0.2.52 training run is
[`hg24yiykjdrvounjsg6bd6si`](https://app.primeintellect.ai/dashboard/training/hg24yiykjdrvounjsg6bd6si).
It was stopped at step 3 after online rewards `0.5557`, `0.5231`, `0.5459`,
and `0.5325` stayed below the baseline; total cost was `$4.06`. Its checkpoint
`lle93v210pcgw6kawfr2mr3e` is queued for a heldout probe when Prime marks it
`READY`.
The current clean comparison point is the v0.2.53 rule-flow baseline
[`vxrolc00tg1h1yc8f96udnu9`](https://app.primeintellect.ai/dashboard/training/vxrolc00tg1h1yc8f96udnu9),
which reached reward `0.6726`, solved `0.4033`, face `0.7644`, action-frontier
`0.3178`, native tool calls `1.0`, and zero errors.
The pure v0.2.53 no-eval training run
[`vychxoaksf66c7pto4wz7rez`](https://app.primeintellect.ai/dashboard/training/vychxoaksf66c7pto4wz7rez)
reached step-1 online reward `0.7133`, solved `0.4456`, face `0.8065`, and
zero errors, then fell back to reward `0.6299` at step 2. It was stopped after
step 2 with cost `$3.38`; no checkpoint was exposed through the CLI.
The mixed v0.2.47/v0.2.53 teacher run
[`vtg65yvig5fstip74awggpgn`](https://app.primeintellect.ai/dashboard/training/vtg65yvig5fstip74awggpgn)
reached clean-lane reward `0.6820` at step 0, fell to `0.6338` at step 1, and
was stopped with cost `$3.03`.
The completed short v0.2.53 run
[`f6ajk2cyut5om7qfb1qisz5e`](https://app.primeintellect.ai/dashboard/training/f6ajk2cyut5om7qfb1qisz5e)
finished two steps naturally. Step 1 reached reward `0.7055`, solved `0.4083`,
direction `0.7434`, and zero env/protocol/tool errors, with cost `$2.23`; no
cloud checkpoint was exposed through `prime train checkpoints`.
The depth-2-only Qwen 9B diagnostic
[`ycx5aoe58lko3gvqyyto7794`](https://app.primeintellect.ai/dashboard/training/ycx5aoe58lko3gvqyyto7794)
completed four steps with rewards `0.5483`, `0.4986`, `0.5019`, and `0.5412`.
It kept native tool calls at `1.0` and env/protocol/tool errors at zero, but did
not improve the depth-2 one-turn-frontier signal (`0.6973` at step 0 versus
`0.6953` at step 3). Checkpoint `pxum4a0nnvfuq5jssm28qgwa` was still
`UPLOADING` at the latest CLI poll; total cost was `$2.16`.
v0.2.54 switches depth-2 from a one-shot frontier proxy to a two-call
candidate-path solve. The Qwen 9B run
[`bg0vbir6u6d521qcr8kghvvv`](https://app.primeintellect.ai/dashboard/training/bg0vbir6u6d521qcr8kghvvv)
completed six steps and reached final online reward `0.7335`, solved `0.6615`,
two native tool calls, and zero errors; cost was `$7.70`. Its READY step-3
checkpoint `ce4skj0ockwhx4zq7ztutsap` was probed on the heldout depth-2 split:
base run
[`xx4pql9agtfl2se7046brmio`](https://app.primeintellect.ai/dashboard/training/xx4pql9agtfl2se7046brmio)
scored reward `0.6335`, solved `0.5625`, while checkpoint probe
[`k2j8tjj7gra5ukmzmrff9epu`](https://app.primeintellect.ai/dashboard/training/k2j8tjj7gra5ukmzmrff9epu)
scored reward `0.6282`, solved `0.5746`. This is a partial heldout solved-rate
gain, not yet a clean reward win.
The checkpoint-tail-room run
[`junxsn2n4rz3uvkcl88ru2in`](https://app.primeintellect.ai/dashboard/training/junxsn2n4rz3uvkcl88ru2in)
then produced READY checkpoint `lbujflb1zyzv764lh9dhzu3s` at step 2. Its
heldout probe
[`j4y1xf2i6cg3lw40m8e3yom9`](https://app.primeintellect.ai/dashboard/training/j4y1xf2i6cg3lw40m8e3yom9)
scored reward `0.6631`, solved `0.6048`, two native tool calls, and zero
errors, a clean `+2.96pp` reward / `+4.23pp` solved-rate improvement over the
heldout base.
A second heldout seed was also positive but much smaller: base
[`lj8g10rkj4haatuhq1iqxqzn`](https://app.primeintellect.ai/dashboard/training/lj8g10rkj4haatuhq1iqxqzn)
scored reward `0.6707`, solved `0.5723`; checkpoint probe
[`jiwkc1h58uwyeiyc8z7pnka9`](https://app.primeintellect.ai/dashboard/training/jiwkc1h58uwyeiyc8z7pnka9)
scored reward `0.6712`, solved `0.5801`, with zero errors.

## Hub Caveat

`prime env push megaminx-solver --path ./environments --owner setrf --visibility PUBLIC --plain`
successfully pushes versions, but the existing Hub record still reports
`PRIVATE` through `prime env status` and `prime env list`. Flip visibility in the
owner web UI for environment id `ozde27sytxjkc3wm83zv4e2c`; do not delete and
recreate the environment unless losing version/eval/training continuity is
acceptable.
