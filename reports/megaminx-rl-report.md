# Megaminx RL Environment Report

Date: 2026-05-13

## Executive Summary

`setrf/megaminx-solver` is a Prime/Verifiers environment for testing whether LLMs can operate on a compact symbolic Megaminx world model. It exposes `load_environment(...) -> vf.Environment`, installs through `prime env install`, evaluates through `prime eval run`, and is packaged for the Prime Environments Hub.

The environment is now a stateful tool environment with a deterministic Megaminx simulator, balanced depth-1 curriculum, action-gated rewards, native tool schemas, JSON-action compatibility for Qwen-style outputs, and no-answer-leak sensor/match/candidate-strip curricula. v0.2.20 adds an explicitly staged depth-1 direction curriculum that reveals the affected face and restricts the action menu to `cw`/`ccw`, so training can isolate whether the model can learn the inverse turn direction from visible sticker order. Prime's current Lab framing is the right fit for this project: hosted training, environments, evals, verifiers, and prime-rl are all part of one train/evaluate/deploy loop: https://docs.primeintellect.ai/introduction.

Current state:

| Item | Value |
| --- | --- |
| GitHub repo | `setrf/megaminx-world-model-bench` |
| Working branch | `codex/megaminx-rl-environment` |
| Pull request | https://github.com/setrf/megaminx-world-model-bench/pull/1 |
| Prime owner | `setrf` |
| Hub environment | `setrf/megaminx-solver` |
| Environment id | `ozde27sytxjkc3wm83zv4e2c` |
| Latest pushed version | `0.2.20` |
| Latest content hash | `536dd250` |
| Latest wheel SHA256 | `fba4eb8658c266acceadfa387f2d503b32e977d5c75cc4983c3c22ea8f5458c2` |
| Visibility | Still reports `PRIVATE` after `--visibility PUBLIC`; see blocker below |
| Best clean base eval | v0.2.17 static match-table prompt, `Qwen/Qwen3.5-35B-A3B`, solved rate `0.442` |
| Latest published eval | v0.2.20 staged face-hint direction eval, solved rate `0.504` |
| Latest hosted run | `lkf8asi6jqam6l0l7op3g6tp`, v0.2.20 staged direction run, stopped |

Finish status: environment release and validation are substantially complete, but the full finish criterion is not met yet because Hub visibility still reports private and hosted RL has not produced the requested `+30pp` depth-1 solved-rate gain. The v0.2.17 train-first adapter regressed on served eval (`0.517` reward, `0.358` solved). The v0.2.18 indexed-direction retry was stopped after step 11 because it failed the improvement gate. The v0.2.19 candidate-strip retry was also stopped after step 10; it remained mechanically clean but did not beat the v0.2.19 base or the v0.2.17 best clean baseline. v0.2.20 is a narrower diagnostic: face choice is solved by a staged hint, while direction remains close to chance in the base model. The v0.2.20 hosted run showed a brief early bump but failed the step-5 gate and was stopped.

## Why We Were Stuck

The simulator itself was not the blocker. The blockers were the model/environment contract and release plumbing:

- The original dense reward allowed no-action rollouts to receive nonzero reward, so Qwen could reason without using tools.
- Qwen chat-completions runs often emitted JSON in text or reasoning fields instead of native OpenAI `tool_calls`.
- A v0.2.12 sensor prompt produced a promising result but leaked answer-adjacent top-overlap candidates, so it was rejected.
- Pre-v0.2.14 hosted training produced illegal/null tool arguments; enum-constrained schemas fixed that.
- Native-tool-only prompts are renderer-clean, but current Qwen OpenAI-chat evals did not actually call tools in that mode.
- `prime env push --visibility PUBLIC` succeeds and creates versions, but the existing Hub environment still lists as `PRIVATE`.

v0.2.17 was the first crack in the wall: it used static candidate neighbor sets for all faces, plus current-state affected faces. That turned depth-1 face selection into a clean set-matching curriculum without printing the hidden inverse solution or top-ranked answer candidates. v0.2.18 keeps that scaffold and adds indexed corner/edge evidence plus a reward that reduces right-face/wrong-direction credit. v0.2.19 goes one step more explicit on visible evidence: it prints candidate-local moved strips for every face, while using a reward that only pays direction credit when the face is also correct. v0.2.20 turns that into a staged direction-isolation curriculum by deriving the correct face from visible affected faces and showing only the two legal actions on that face; it is deliberately not a final naturalistic evaluation prompt.

## Environment Design

Package path:

```bash
environments/megaminx_solver
```

Entrypoint:

```python
load_environment(...) -> vf.Environment
```

Class:

```python
MegaminxEnv(vf.StatefulToolEnv)
```

Public API:

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

Tools:

| Tool | Args | Purpose |
| --- | --- | --- |
| `rotate` | `face: A-L`, `direction: cw\|ccw` | Apply one face turn |
| `inspect` | `face: A-L\|all` | Observe one face or the full net |
| `finish` | none | End the rollout |

Simulator:

- 12 labeled faces `A` through `L`.
- 132 stickers: 12 centers plus 30 edge pieces and 20 corner pieces.
- 24 legal moves: every face clockwise/counterclockwise.
- Dodecahedral neighbor rings generate turn effects.
- Scramble and inverse solution are stored in task metadata and never placed directly in the prompt.
- Depth-1 generation is balanced across all 24 legal moves.

Named splits:

| Split | Depths | Intended use |
| --- | ---: | --- |
| `depth1`, `train_depth1`, `eval_depth1` | 1 | First RL and held-out one-turn eval |
| `easy`, `train_easy`, `eval_easy` | 1-3 | First generalization band |
| `medium`, `train_medium`, `eval_medium` | 4-6 | Mid-horizon eval |
| `hard`, `train_hard`, `eval_hard` | 7-10 | Hard eval |
| `eval` | 1-10 | Broad eval |
| `train` | custom, default 1-8 | General curriculum |

## Reward And Prompts

Default dense reward:

```text
0.60 * solved
+ 0.25 * sticker_accuracy
+ 0.10 * piece_accuracy
+ 0.05 * efficiency_if_solved
```

RL reward styles:

| Style | Purpose |
| --- | --- |
| `action_gated_dense` | No rotate gives `0`; solved gives `1`; non-solved progress delta only |
| `action_gated_curriculum` | No rotate gives `0`; valid first rotate gets shaped face/direction/progress credit; solved gives `1` |
| `action_gated_overlap` | Smooth on-ramp based on chosen face neighbor-ring overlap, plus exact face/direction and progress credit |
| `action_gated_direction` | Direction-focused v0.2.18 reward: solved gives `1`; no rotate gives `0`; valid first rotate gets separate face, direction, and small progress credit capped at `0.35` |
| `action_gated_exact_direction` | v0.2.19 reward: solved gives `1`; no rotate gives `0`; correct face/wrong direction gets `0.15`; direction credit is only paid when the face is correct |

Prompt styles:

| Style family | Styles | Purpose |
| --- | --- | --- |
| Default | `default`, `action_first` | Normal tool-env prompts |
| JSON action | `direct_json_action`, `choice_json_action`, `topology_choice_json_action`, `sensor_choice_json_action`, `sensor_match_json_action`, `sensor_indexed_match_json_action`, `sensor_candidate_strips_json_action` | Compatibility path for models that emit JSON text instead of native tool calls |
| Staged JSON action | `stage_face_hint_direction_json_action` | Depth-1-only curriculum that reveals the derived correct face and restricts the JSON action menu to that face's two directions |
| Native tool | `native_action`, `topology_native_tool`, `sensor_native_tool`, `sensor_match_native_tool` | Renderer/TITO-compatible native tool-call prompts with text fallback disabled by default |

Important compatibility details:

- JSON prompt styles auto-enable `allow_text_tool_actions=True`; native prompt styles leave it disabled unless overridden.
- As of v0.2.12, prose before JSON no longer counts as an action.
- As of v0.2.13, the sensor prompt no longer prints overlap counts or top-overlap answer candidates.
- As of v0.2.14, rotate/inspect tool schemas constrain enum values.
- As of v0.2.17, `sensor_match_json_action` adds static sorted neighbor sets for every face. Tests assert it does not print `Highest-overlap`, `Neighbor-overlap`, `answer`, `scramble`, or `inverse_solution`.
- As of v0.2.18, `sensor_indexed_match_json_action` adds an indexed direction guide and `action_gated_direction` lowers right-face/wrong-direction reward to make direction learnable.
- As of v0.2.19, `sensor_candidate_strips_json_action` adds visible candidate-local moved strips for every face and `action_gated_exact_direction` removes wrong-face direction credit.
- As of v0.2.20, `stage_face_hint_direction_json_action` derives the correct depth-1 face from visible affected faces, rejects non-depth-1 splits, and presents only `cw`/`ccw` choices for that face. This is a staged curriculum, not a clean final eval.

## Renderer Guidance

Prime's May 2026 `renderers` work is token-level RL infrastructure, not a Megaminx image renderer: https://www.primeintellect.ai/blog/renderers. It renders messages to token ids, parses sampled token ids back into structured assistant outputs, attributes tokens for loss masking, and extends multi-turn histories without re-rendering model-sampled assistant text.

That matters here because tool-use rollouts are only trainable cleanly when sampled assistant history remains the source of truth. Near-term Megaminx training should keep native-tool trajectories clean for renderer/TITO runs, avoid silent tool-call repair/history normalization, and record any repair or compaction as an explicit event.

A separate future workstream is a Megaminx visual renderer for physical-world evaluation. That should convert simulator states into deterministic SVG/PNG/HTML observations first, then later add seeded camera pose, lighting, occlusion, sticker variation, and real-image calibration. The visual renderer must remain a state-to-observation layer only; it must not expose scramble, inverse solution, answer candidates, or derived reward hints.

## Validation

Latest local test command:

```bash
uv run pytest tests/test_megaminx_solver.py -q
```

Latest result:

```text
35 passed in 3.31s
```

Coverage includes:

- topology counts: 12 centers, 30 edges, 20 corners, 132 stickers;
- move multiset preservation;
- `cw` then `ccw` identity;
- five turns of one face return identity;
- generated scramble plus inverse solves;
- invalid string and non-string args are safe;
- invalid native tool-call args are safe;
- named split depth bands;
- balanced depth-1 scrambles across all 24 legal moves;
- no direct answer leakage;
- JSON action fallback from content/reasoning fields;
- rejection of prose before JSON fallback;
- JSON/native prompt-style contracts;
- `allow_text_tool_actions` default matrix;
- overlap and curriculum reward behavior;
- indexed direction prompt and reward behavior;
- candidate-strip prompt and exact-direction reward behavior;
- staged face-hint direction prompt and depth-1 guard;
- separate `move_budget` versus Verifiers `max_turns`;
- tool schema enum checks;
- malformed JSON-action args;
- sensor-match prompt no-leak checks.

Prime install validation:

```bash
prime env install setrf/megaminx-solver@0.2.20
```

## Hub Release State

Latest version check:

```bash
prime env version list setrf/megaminx-solver --plain
```

Observed top versions:

| Version | Created UTC | Hash |
| --- | --- | --- |
| `0.2.20` | 2026-05-13 13:46 | `536dd250` |
| `0.2.19` | 2026-05-13 13:29 | `52c31b85` |
| `0.2.18` | 2026-05-13 13:14 | `02aa37af` |
| `0.2.17` | 2026-05-13 12:03 | `a198f03b` |
| `0.2.16` | 2026-05-13 12:02 | `06576caf` |
| `0.2.15` | 2026-05-13 11:59 | `5a470294` |
| `0.2.14` | 2026-05-13 11:45 | `4b944c91` |
| `0.2.13` | 2026-05-13 11:18 | `44082153` |

Public push command used:

```bash
prime env push megaminx-solver --visibility PUBLIC --plain
```

The push succeeds and creates versions, but Hub listing still reports:

```json
{"environment":"setrf/megaminx-solver","visibility":"PRIVATE","version":"0.2.20"}
```

This appears to be a Prime Hub visibility mutation issue for an existing environment, not an environment packaging issue. The safest next action is to ask Prime support to update environment id `ozde27sytxjkc3wm83zv4e2c` to public in place. Do not delete/recreate the environment to force public visibility unless losing version/eval/training continuity is explicitly accepted.

## Key Evaluations

All canonical evals used `prime eval run` without `--skip-upload`.

| Eval | Version | Model | Prompt/reward | Reward | Solved | Face correct | Direction correct | Rotate rate | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| https://app.primeintellect.ai/dashboard/evaluations/etr6y2vgx4yw1z7sbn70wjwp | `0.2.1` | `Qwen/Qwen3.5-4B` | dense/default | `0.302` | `0.000` | n/a | n/a | `0.000` | Exposed dense no-action trap |
| https://app.primeintellect.ai/dashboard/evaluations/i6tqnlapp8czv2rkjdqp8nex | `0.2.7` | `Qwen/Qwen3.5-4B` | direct JSON/curriculum | `0.152` | `0.083` | n/a | `0.583` | `1.000` | First useful Qwen JSON signal |
| https://app.primeintellect.ai/dashboard/evaluations/zveus73j6lma0b4knjkknx6a | `0.2.12` | `Qwen/Qwen3.5-35B-A3B` | leaked sensor JSON/curriculum | `0.205` | `0.167` | `0.267` | `0.200` | `0.333` | Rejected: prompt exposed top-overlap face candidate |
| https://app.primeintellect.ai/dashboard/evaluations/m263wlcigzv21e6xclevtkhf | `0.2.13` | `Qwen/Qwen3.5-35B-A3B` | clean sensor, forced `rotate` | `0.130` | `0.100` | `0.133` | `0.500` | `1.000` | Historical clean baseline |
| https://app.primeintellect.ai/dashboard/evaluations/ap39nu0kwty1awemhdgd77nm | `0.2.14` | `Qwen/Qwen3.5-35B-A3B` | enum schema, forced `rotate` | `0.109` | `0.067` | `0.133` | `0.433` | `1.000` | Illegal moves fixed |
| https://app.primeintellect.ai/dashboard/evaluations/yosp4xsfjqpu0swehmuk36lx | `0.2.15` | `Qwen/Qwen3.5-35B-A3B` | native-only prompt | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` | Current chat path did not emit native tool calls |
| https://app.primeintellect.ai/dashboard/evaluations/k16letr5pyhopqwu0e5seshs | `0.2.17` | `Qwen/Qwen3.5-35B-A3B` | static match table/curriculum | `0.570` | `0.442` | `0.775` | `0.525` | `1.000` | Best clean baseline; illegal moves `0` |
| https://app.primeintellect.ai/dashboard/evaluations/pc5kcutdcfcecfyobjfkwa8c | `0.2.17` | `Qwen/Qwen3.5-35B-A3B:rvwwd6wom5j739h6x0x8aj36` | trained adapter, static match table/curriculum | `0.517` | `0.358` | `0.775` | `0.458` | `1.000` | Served adapter regressed versus base |
| https://app.primeintellect.ai/dashboard/evaluations/gnudftur8j76ndbokfpas7jq | `0.2.18` | `Qwen/Qwen3.5-35B-A3B` | indexed match table/direction reward | `0.479` | `0.383` | `0.742` | `0.525` | `1.000` | Direction reward separates failure classes; illegal moves `0` |
| https://app.primeintellect.ai/dashboard/evaluations/c04ubpjtt27x9f74tc6130u6 | `0.2.19` | `Qwen/Qwen3.5-35B-A3B` | candidate strips/exact-direction reward | `0.451` | `0.392` | `0.758` | `0.525` | `1.000` | Active RL baseline; illegal moves `0`, env errors `0` |
| https://app.primeintellect.ai/dashboard/evaluations/y78em945ey7fb8ijmxbej5mg | `0.2.20` | `Qwen/Qwen3.5-35B-A3B` | staged face hint/exact-direction reward | `0.579` | `0.504` | `1.000` | `0.504` | `1.000` | Direction-isolation baseline; face is intentionally given, illegal moves `0`, env errors `0` |

The v0.2.12 sensor result is retained as failure analysis only. The naturalistic base to beat is the v0.2.17 static match-table baseline. The v0.2.20 result is a staged diagnostic, so it is only comparable to other face-hinted direction-isolation runs.

Sample audit for the v0.2.17 baseline:

- 120 balanced depth-1 samples: each of the 24 legal target moves appears 5 times.
- Exact solve/correct first rotate: `53/120 = 44.17%`.
- Right face, wrong direction: `40/120 = 33.33%`.
- Wrong face: `27/120 = 22.50%`.
- Face accuracy: `93/120 = 77.50%`.
- Direction accuracy conditioned on correct face: `53/93 = 56.99%`.
- Reward clusters are exactly `53` at `1.0`, `40` at `0.37`, and `27` at `0.02`.
- Prompt audit found no raw `answer`, `inverse_solution`, encoded `scramble`, target fields, `last_move`, or tool status text in prompts.

This confirms `sensor_match_json_action` is not a raw hidden-metadata leak. It is, however, a strong visible-derived scaffold: affected faces plus static candidate neighbor sets make the correct face mechanically recoverable for one-turn scrambles. Direction remains the harder and less-scaffolded part.

Reproduction command for the v0.2.17 baseline:

```bash
prime eval run setrf/megaminx-solver@0.2.17 \
  -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"depth1","num_examples":120,"max_turns":2,"reward_style":"action_gated_curriculum","prompt_style":"sensor_match_json_action"}' \
  -n 120 -r 1 -t 48 -T 0.7 -A --plain
```

Reproduction command for the v0.2.18 indexed-direction baseline:

```bash
prime eval run setrf/megaminx-solver@0.2.18 \
  -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"depth1","num_examples":120,"max_turns":2,"move_budget":1,"reward_style":"action_gated_direction","prompt_style":"sensor_indexed_match_json_action","allow_text_tool_actions":true}' \
  -n 120 -r 1 -t 48 -T 0.7 -A --plain
```

Reproduction command for the v0.2.19 candidate-strip baseline:

```bash
prime eval run setrf/megaminx-solver@0.2.19 \
  -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"depth1","num_examples":120,"max_turns":2,"move_budget":1,"reward_style":"action_gated_exact_direction","prompt_style":"sensor_candidate_strips_json_action","allow_text_tool_actions":true}' \
  -n 120 -r 1 -t 48 -T 0.7 -A --plain
```

Reproduction command for the v0.2.20 staged direction baseline:

```bash
prime eval run setrf/megaminx-solver@0.2.20 \
  -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"eval_depth1","num_examples":240,"seed":1042,"max_turns":2,"move_budget":1,"reward_style":"action_gated_exact_direction","prompt_style":"stage_face_hint_direction_json_action","allow_text_tool_actions":true}' \
  -n 240 -r 1 -t 48 -T 0.7 -A --plain
```

## Hosted RL Runs

| Run | Version | Model | Config | Status | Cost | Outcome |
| --- | --- | --- | --- | --- | ---: | --- |
| `n1s75wk4cv32g7xhs1ed9bu6` | `0.2.5` | Qwen 4B | action-gated dense | `FAILED` | `$0.262` | Step 0 all rollouts zero-advantage |
| `fl1s4r2esij2xlj7njddidn2` | `0.2.6` | Qwen 4B | curriculum/action-first | `STOPPED` | `$0.921` | Reached step 10, action calls collapsed |
| `wxrezmp5mlythdnovjrs8mo6` | `0.2.8` | Qwen 35B-A3B | choice JSON signal | `STOPPED` | `$11.64` | Step-5 adapter deployed, but eval failed to improve |
| `swfkmmxm2xchbkh5qrbhj8yl` | `0.2.11` | Qwen 35B-A3B | overlap | `STOPPED` | `$14.19` | Stable overlap proxy, exact face/solve stayed near random |
| `sny20a8rkxfe5nwsdkkb49ld` | `0.2.12` | Qwen 35B-A3B | leaked sensor curriculum | `STOPPED` | `$9.07` | Stopped after leak audit; do not use as evidence |
| `a8dd0c56s2edlhbfsrwni8jn` | `0.2.13` | Qwen 35B-A3B | clean sensor + forced rotate, `T=0.7` | `STOPPED` | `$17.27` | Reward stayed below the v0.2.13 base; high illegal rate before enum patch |
| `qrfsf3jrjjo3l99lvzub6u1q` | `0.2.14` | Qwen 35B-A3B | enum schema + forced rotate, `T=0.7` | `STOPPED` | `$17.33` | Illegal args fixed, but no learning over baseline by step 5 |
| `qn8n2ngda0sbigh1xuoq1aua` | `0.2.17` | Qwen 35B-A3B | match table + forced rotate, `T=0.7`, aggressive batch | `STOPPED` | `$18.44` | Clean rollouts but below base; stopped after off-policy drift reached `3.33` by step 6 |
| `fef0tabjbbe8jz2qnwj3jq3j` | `0.2.17` | Qwen 35B-A3B | match table + forced rotate, small batch, lower LR, oversampling `2.0` | `STOPPED` | `$0.17` | Stopped before training so the config could be relaunched with oversampling `1.0` |
| `fgw7xi5zj7cx6nthnggi72an` | `0.2.17` | Qwen 35B-A3B | match table + forced rotate, small batch, lower LR, oversampling `1.0`, eval block enabled | `STOPPED` | `$0.46` | Stopped before training because step-0 eval was slow; next retry trains first and evaluates checkpoints after |
| `x1y62onvjc1sscmxpwdobdxm` | `0.2.17` | Qwen 35B-A3B | match table + forced rotate, train-first, lower LR, oversampling `1.0` | `COMPLETED` | `$16.81` | Clean run. Final logged step beat base reward slightly (`0.582` vs `0.570`) but solved-rate gain was only `+0.9pp`; final adapter `rvwwd6wom5j739h6x0x8aj36` deployed for served eval |
| `oi3fw0nowunksah9b615a2c1` | `0.2.18` | Qwen 35B-A3B | indexed match table + direction reward, train-first, lower LR, oversampling `1.0` | `STOPPED` | `$4.72` | Failed the gate by step 11; best spike was step 4 solved `0.467`, but the run did not sustain improvement |
| `pr9gkys1jskj8yukyfye6mqv` | `0.2.19` | Qwen 35B-A3B | candidate strips + exact-direction reward, train-first, lower LR, oversampling `1.0` | `STOPPED` | `$6.00` | Stopped at step 10. Best solved was `0.391`, trailing steps 8-10 averaged `0.360`, no checkpoints were produced |
| `lkf8asi6jqam6l0l7op3g6tp` | `0.2.20` | Qwen 35B-A3B | staged face hint + exact-direction reward, train-first, lower LR, oversampling `1.0` | `STOPPED` | `$2.89` | Early steps rose above held-out base, then step 5 fell to solved/direction `0.461`; stopped with no checkpoints |

Stopped aggressive run command:

```bash
prime train configs/rl/megaminx-depth1-qwen-35b-a3b-sensor.toml --yes --plain
```

Stopped aggressive config:

- model `Qwen/Qwen3.5-35B-A3B`
- env `setrf/megaminx-solver@0.2.17`
- split `train_depth1`
- prompt `sensor_match_json_action`
- reward `action_gated_curriculum`
- max steps `100`
- batch size `512`
- rollouts per example `16`
- learning rate `1e-7`
- LoRA alpha `16`
- oversampling factor `4.0`
- max async level `1`
- sampling max tokens `48`
- temperature `0.7`
- `enable_thinking=false`
- `tool_choice={"type":"function","function":{"name":"rotate"}}`

Observed stopped-run metrics:

| Step | Reward | Solved | Face correct | Direction correct | Illegal | Errors | Truncated |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.474` | `0.340` | `0.686` | `0.469` | `0.000` | `0.000` | `0.000` |
| 1 | `0.473` | `0.369` | `0.629` | `0.564` | `0.000` | `0.000` | `0.000` |
| 2 | `0.476` | `0.361` | `0.652` | `0.544` | `0.000` | `0.000` | `0.000` |
| 3 | `0.401` | `0.270` | `0.602` | `0.421` | `0.000` | `0.000` | `0.000` |
| 4 | `0.510` | `0.398` | `0.684` | `0.588` | `0.000` | `0.000` | `0.000` |
| 5 | `0.406` | `0.294` | `0.572` | `0.508` | `0.000` | `0.000` | `0.000` |
| 6 | `0.422` | `0.282` | `0.639` | `0.465` | `0.000` | `0.000` | `0.000` |

Interpretation: the run was healthy at the environment/tooling layer, but it did not improve over the v0.2.17 base eval. Off-policy drift increased from `0` to `3.33`, so the run was stopped at step 6.

Completed train-first retry command:

```bash
prime train configs/rl/megaminx-depth1-qwen-35b-a3b-sensor-sync.toml --yes --plain
```

Completed train-first retry config:

- run `x1y62onvjc1sscmxpwdobdxm`
- model `Qwen/Qwen3.5-35B-A3B`
- env `setrf/megaminx-solver@0.2.17`
- split `train_depth1`
- prompt `sensor_match_json_action`
- reward `action_gated_curriculum`
- max steps `30`
- batch size `128`
- rollouts per example `8`
- learning rate `5e-8`
- LoRA alpha `16`
- oversampling factor `1.0`
- max async level `1` because the Hosted Training backend rejects `0` even though local config parsing accepts it
- sampling max tokens `48`
- temperature `0.7`
- no startup eval block; evaluate depth-1/easy manually after deployment

Observed train-first retry metrics:

| Step | Reward | Solved | Face correct | Direction correct | Illegal | Errors | Truncated |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.462` | `0.328` | `0.672` | `0.461` | `0.000` | `0.000` | `0.000` |
| 7 | `0.543` | `0.422` | `0.734` | `0.547` | `0.000` | `0.000` | `0.000` |
| 13 | `0.562` | `0.453` | `0.734` | `0.617` | `0.000` | `0.000` | `0.000` |
| 20 | `0.478` | `0.328` | `0.719` | `0.484` | `0.000` | `0.000` | `0.000` |
| 26 | `0.564` | `0.438` | `0.766` | `0.555` | `0.000` | `0.000` | `0.000` |
| 29 | `0.582` | `0.451` | `0.795` | `0.531` | `0.000` | `0.000` | `0.000` |

Interpretation: the smaller synchronous run is the first hosted run with a trainer-batch reward above the v0.2.17 base baseline, but the effect is small and noisy. It does not satisfy the original acceptance target of `+30pp` depth-1 solved-rate improvement.

Monitoring commands:

```bash
prime train get x1y62onvjc1sscmxpwdobdxm --plain
prime train logs x1y62onvjc1sscmxpwdobdxm --plain
prime train progress x1y62onvjc1sscmxpwdobdxm --plain
prime train metrics x1y62onvjc1sscmxpwdobdxm --plain
prime train distributions x1y62onvjc1sscmxpwdobdxm --step 29 --plain
prime train rollouts x1y62onvjc1sscmxpwdobdxm --step 29 --plain
prime train usage x1y62onvjc1sscmxpwdobdxm --plain
prime train checkpoints x1y62onvjc1sscmxpwdobdxm --plain
```

Adapter deployment:

```bash
prime deployments delete pesdp7cdv8sbivylxo9tykct --plain
prime deployments create rvwwd6wom5j739h6x0x8aj36 --yes --plain
prime deployments list --plain -o json -n 100
```

The first command unloaded an older failed Megaminx adapter to free one of the five active deployment slots. It did not delete model files.

Served v0.2.17 adapter eval:

```bash
prime eval run setrf/megaminx-solver@0.2.17 \
  -m Qwen/Qwen3.5-35B-A3B:rvwwd6wom5j739h6x0x8aj36 \
  -p prime \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"depth1","num_examples":120,"max_turns":2,"reward_style":"action_gated_curriculum","prompt_style":"sensor_match_json_action","allow_text_tool_actions":true}' \
  -n 120 -r 1 -t 48 -T 0.7 -A --plain
```

Served adapter result:

| Eval | Reward | Solved | Face correct | Direction correct | Illegal | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| https://app.primeintellect.ai/dashboard/evaluations/pc5kcutdcfcecfyobjfkwa8c | `0.517` | `0.358` | `0.775` | `0.458` | `0.000` | `0.000` |

Interpretation: the v0.2.17 trainer-batch bump did not survive deployment. It regressed solved rate from the base `0.442` to `0.358`, mostly by worsening direction.

v0.2.18 indexed-direction retry command:

```bash
prime train configs/rl/megaminx-depth1-qwen-35b-a3b-indexed-direction.toml --yes --plain
```

v0.2.18 indexed-direction config:

- run `oi3fw0nowunksah9b615a2c1`
- model `Qwen/Qwen3.5-35B-A3B`
- env `setrf/megaminx-solver@0.2.18`
- split `train_depth1`
- prompt `sensor_indexed_match_json_action`
- reward `action_gated_direction`
- max turns `2`
- move budget `1`
- max steps `30`
- batch size `128`
- rollouts per example `8`
- learning rate `5e-8`
- LoRA alpha `16`
- oversampling factor `1.0`
- max async level `1`
- sampling max tokens `48`
- temperature `0.7`

Observed v0.2.18 indexed-direction metrics:

| Step | Reward | Solved | Face correct | Direction correct | Illegal | Errors | Truncated |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.426` | `0.336` | `0.641` | `0.531` | `0.000` | `0.000` | `0.000` |
| 4 | `0.540` | `0.467` | `0.733` | `0.588` | `0.000` | `0.000` | `0.000` |
| 7 | `0.502` | `0.414` | `0.742` | `0.547` | `0.000` | `0.000` | `0.000` |
| 10 | `0.455` | `0.354` | `0.733` | `0.508` | `0.000` | `0.000` | `0.000` |
| 11 | `0.398` | `0.305` | `0.594` | `0.578` | `0.000` | `0.000` | `0.000` |

Interpretation: the reward shape worked mechanically but did not sustain an improvement over the v0.2.17 base. The run was stopped at step 11; step-5 checkpoint `b0nz5kwq6lrhdfpn7que4qt9` exists but was not deployed because the metrics did not clear the gate.

v0.2.19 candidate-strip retry command:

```bash
prime train configs/rl/megaminx-depth1-qwen-35b-a3b-candidate-strips.toml --yes --plain
```

v0.2.19 candidate-strip config:

- run `pr9gkys1jskj8yukyfye6mqv`
- model `Qwen/Qwen3.5-35B-A3B`
- env `setrf/megaminx-solver@0.2.19`
- split `train_depth1`
- prompt `sensor_candidate_strips_json_action`
- reward `action_gated_exact_direction`
- max turns `2`
- move budget `1`
- max steps `30`
- batch size `128`
- rollouts per example `8`
- learning rate `5e-8`
- LoRA alpha `16`
- oversampling factor `1.0`
- max async level `1`
- sampling max tokens `48`
- temperature `0.7`

Observed v0.2.19 candidate-strip metrics:

| Step | Reward | Solved | Face correct | Direction correct | Illegal | Errors | Truncated | Off-policy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.371` | `0.336` | `0.500` | `0.609` | `0.000` | `0.000` | `0.000` | `0.000` |
| 2 | `0.398` | `0.336` | `0.711` | `0.438` | `0.000` | `0.000` | `0.000` | `0.065` |
| 5 | `0.417` | `0.375` | `0.602` | `0.578` | `0.000` | `0.000` | `0.000` | `0.000` |
| 8 | `0.442` | `0.391` | `0.695` | `0.523` | `0.000` | `0.000` | `0.000` | `0.008` |
| 9 | `0.433` | `0.391` | `0.625` | `0.617` | `0.000` | `0.000` | `0.000` | `0.000` |
| 10 | `0.346` | `0.298` | `0.562` | `0.510` | `0.000` | `0.000` | `0.000` | `0.000` |

Interpretation: the run was healthy at the environment/tooling layer, but the reward curve did not clear the step-10 gate. It was stopped at `$6.00` total usage before checkpoint/adapters were created.

v0.2.20 staged direction run command:

```bash
prime train configs/rl/megaminx-depth1-qwen-35b-a3b-face-hint-direction.toml --yes --plain
```

v0.2.20 staged direction config:

- run `lkf8asi6jqam6l0l7op3g6tp`
- model `Qwen/Qwen3.5-35B-A3B`
- env `setrf/megaminx-solver@0.2.20`
- split `train_depth1`
- prompt `stage_face_hint_direction_json_action`
- reward `action_gated_exact_direction`
- max turns `2`
- move budget `1`
- max steps `30`
- batch size `128`
- rollouts per example `8`
- learning rate `5e-8`
- LoRA alpha `16`
- oversampling factor `1.0`
- max async level `1`
- sampling max tokens `48`
- temperature `0.7`

Observed v0.2.20 staged direction metrics:

| Step | Reward | Solved | Face correct | Direction correct | Illegal | Errors | Truncated | Off-policy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.632` | `0.567` | `1.000` | `0.567` | `0.000` | `0.000` | `0.000` | `0.000` |
| 1 | `0.641` | `0.578` | `1.000` | `0.578` | `0.000` | `0.000` | `0.000` | `0.000` |
| 2 | `0.568` | `0.492` | `1.000` | `0.492` | `0.000` | `0.000` | `0.000` | `0.095` |
| 3 | `0.555` | `0.477` | `1.000` | `0.477` | `0.000` | `0.000` | `0.000` | `0.000` |
| 4 | `0.635` | `0.570` | `1.000` | `0.570` | `0.000` | `0.000` | `0.000` | `0.033` |
| 5 | `0.542` | `0.461` | `1.000` | `0.461` | `0.000` | `0.000` | `0.000` | `0.000` |

Interpretation: the environment/tool contract was clean, and the face hint did exactly what it was supposed to do. Direction learning did not stick. The run briefly exceeded the held-out staged baseline (`0.504` solved/direction), then fell below it by step 5. It was stopped at `$2.89`; no checkpoints were created.

## Finish Criteria

Original acceptance target:

- public Hub environment works;
- trained checkpoint improves depth-1 solved rate by at least `+30pp`;
- `easy` reward improves over base;
- no environment errors;
- all commands/results are reproducible.

Current acceptance status:

| Criterion | Status |
| --- | --- |
| Environment install/eval works | Passed |
| Tests pass | Passed |
| Hub public | Blocked by visibility mutation |
| Base depth-1 signal | Passed: v0.2.17 solved `0.442` |
| RL checkpoint improves base | Failed so far: v0.2.17 served adapter regressed; v0.2.18, v0.2.19, and v0.2.20 stopped without deployable improvement |
| Easy/medium/hard final evals | Pending |
| Final PR/release/tag | Pending |

If the next staged curriculum fails to beat v0.2.17 clearly, finish should be documented as “environment released plus negative hosted RL result.” That is still a useful result: the prompt/curriculum made the base model much better, but LoRA RL did not yet improve it meaningfully.

## Repro Commands

Install:

```bash
prime env install setrf/megaminx-solver@0.2.20
```

Best clean baseline eval:

```bash
prime eval run setrf/megaminx-solver@0.2.17 \
  -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"depth1","num_examples":120,"max_turns":2,"reward_style":"action_gated_curriculum","prompt_style":"sensor_match_json_action"}' \
  -n 120 -r 1 -t 48 -T 0.7 -A --plain
```

Current v0.2.20 staged direction eval:

```bash
prime eval run setrf/megaminx-solver@0.2.20 \
  -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"eval_depth1","num_examples":240,"seed":1042,"max_turns":2,"move_budget":1,"reward_style":"action_gated_exact_direction","prompt_style":"stage_face_hint_direction_json_action","allow_text_tool_actions":true}' \
  -n 240 -r 1 -t 48 -T 0.7 -A --plain
```

Training:

```bash
prime train configs/rl/megaminx-depth1-qwen-35b-a3b-face-hint-direction.toml --yes --plain
```

Historical served adapter eval:

```bash
prime eval run setrf/megaminx-solver@0.2.17 \
  -m Qwen/Qwen3.5-35B-A3B:rvwwd6wom5j739h6x0x8aj36 \
  -p prime \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"depth1","num_examples":120,"max_turns":2,"reward_style":"action_gated_curriculum","prompt_style":"sensor_match_json_action","allow_text_tool_actions":true}' \
  -n 120 -r 1 -t 48 -T 0.7 -A --plain
```

Local validation:

```bash
uv run pytest
prime env install megaminx-solver --plain
```

Publish attempt:

```bash
prime env push megaminx-solver --visibility PUBLIC --plain
```

## Limitations And Next Work

- Hub visibility remains `PRIVATE` despite public push commands.
- v0 is text-first; visual observations are reserved for a later deterministic renderer milestone.
- `sensor_match_json_action` is a curriculum prompt: it exposes current-state affected faces and static candidate neighbor sets, but not the hidden inverse solution or ranked answer candidates.
- `sensor_indexed_match_json_action` exposes indexed visible sticker evidence for direction, but still does not expose the inverse solution or answer candidates.
- `sensor_candidate_strips_json_action` exposes candidate-local visible strips for every face; it improves direction observability but remains a shaped depth-1 curriculum rather than a naturalistic observation.
- `stage_face_hint_direction_json_action` is a deliberately staged depth-1 curriculum that reveals the derived face and leaves only direction to infer. It should not be reported as a clean Megaminx-solving benchmark.
- Without JSON fallback, current Qwen OpenAI-chat evals often fail to produce tool calls.
- Direction correctness remains the harder half of depth-1 solving.
- If v0.2.20 direction-isolation RL fails to improve, the next experiment should target a renderer-native tool path or a richer observation renderer rather than more training on the same JSON prompt.
