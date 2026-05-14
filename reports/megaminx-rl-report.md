# Megaminx RL Environment Report

Date: 2026-05-14

## Executive Summary

`setrf/megaminx-solver` is a Prime/Verifiers environment for studying whether
LLMs can act in a compact symbolic model of a physical puzzle. It defines a
Megaminx world, legal tool actions, persistent rollout state, deterministic
reward, and reproducible train/eval curricula.

The project is not at the original aggressive finish target, but it now has a
cleaner experimental path. The environment is hardened and pushed, the native
Token-In Token-Out training lane is working, and v0.2.50 removes the two main
v0.2.47 scaffolding leaks: printed `cw_mask`/`ccw_mask` reward proxies and
hidden-target candidate insertion. The requested `+30pp` solved-checkpoint gain
was not achieved. v0.2.51 adds a diagnostic guard for the clean
geometry-frontier lane; the Qwen 9B baseline confirms the inverse target face is
available in the visible candidate set on every sample. v0.2.52 then patches
the next failure mode: it replaces the long raw strip table with a fair
candidate-relative flow transform and adds a stricter frontier reward after an
audit found the old scalar reward could be beaten by a fixed slot/direction
policy. v0.2.53 keeps that clean representation but adds an explicit fair
counting rule for the visible `+1/-1` flow tokens; this became the strongest
clean Qwen 9B baseline so far. v0.2.54 extends that representation into a
two-call path: after the first candidate action, the environment refreshes the
candidate relative-flow table so depth-2 examples can actually be solved. The
first v0.2.54 Qwen 9B run reached online reward `0.7335` and solved rate
`0.6615` with two native tool calls and zero environment/tool errors. The first
READY checkpoint probe was neutral rather than a clean reward win, but a
checkpoint-tail-room rerun produced a heldout win: checkpoint
`lbujflb1zyzv764lh9dhzu3s` improved reward from `0.6335` to `0.6631` and solved
rate from `0.5625` to `0.6048`. A second heldout seed was also positive but
much smaller: reward `0.6707` to `0.6712`, solved `0.5723` to `0.5801`.

| Item | Value |
| --- | --- |
| GitHub repo | `setrf/megaminx-world-model-bench` |
| Working branch | `codex/megaminx-rl-crack-v2` |
| Prime owner | `setrf` |
| Hub environment | [`setrf/megaminx-solver`](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver) |
| Environment id | `ozde27sytxjkc3wm83zv4e2c` |
| Latest pushed version | `0.2.54` |
| Latest wheel SHA256 | `9a1edd7195f08a516596efd772ef8729149b1332d7bc885e090eb52613289748` |
| Latest install check | `prime env install megaminx-solver --plain` and Hub push succeeded |
| Latest local tests | `uv run pytest -q` -> `102 passed in 7.65s` |
| Visibility | CLI/API still report `PRIVATE` after public pushes |

Prime framing used:

- [Environments Hub blog](https://www.primeintellect.ai/blog/environments):
  environments define the world, rules, state/action/reward loop, and shared RL
  boundary.
- [Prime docs introduction](https://docs.primeintellect.ai/introduction): Lab
  connects Hosted Training, Environments Hub, Evaluations, Verifiers, sandboxes,
  prime-rl, and inference.
- [Verifiers overview](https://docs.primeintellect.ai/verifiers/overview):
  environments package datasets, harnesses/tools/context, and reward functions,
  and expose `load_environment(...)`.
- [Renderers blog](https://www.primeintellect.ai/blog/renderers): agentic RL
  should preserve sampled token identity through Token-In Token-Out and renderer
  parsing, especially for multi-turn tool trajectories.

## Environment Design

Package path:

```bash
environments/megaminx_solver
```

Entrypoint:

```python
load_environment(...) -> vf.Environment
```

Environment class:

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
| `rotate` | `face: A-L`, `direction: cw|ccw` | Apply one face turn |
| `inspect` | `face: A-L|all` | Observe one face or the full net |
| `finish` | none | End the rollout |

Simulator properties:

- 12 labeled faces `A` through `L`.
- 132 stickers: 12 centers, 30 edge pieces, and 20 corner pieces.
- 24 legal moves: every face clockwise/counterclockwise.
- Dodecahedral neighbor rings generate turn effects.
- Scramble and inverse solution are stored in task metadata and withheld from
  prompts.
- Depth-1 data is balanced across all 24 legal moves.

## Reward And Prompt Surface

Default dense reward:

```text
0.60 * solved
+ 0.25 * sticker_accuracy
+ 0.10 * piece_accuracy
+ 0.05 * efficiency_if_solved
```

Strict binary heldout reward:

```text
reward_style = "action_gated_binary_direction"
```

This returns `1.0` only when the first clean rotate is exactly the inverse move.
The strict gate requires exactly one action attempt, exactly one rotate, no
inspect/finish, no illegal moves, no parse/call errors, and no protocol
violation.

Training reward:

```text
reward_style = "action_gated_strict_shaped_direction"
prompt_style = "stage_solve_direction_flow_native_tool"
allow_text_tool_actions = false
```

This keeps the same clean-action gate but adds partial credit for valid actions:
exact solve gives `1.0`, wrong direction on the correct face gives around
`0.40`, wrong face with correct direction gives around `0.10`, and positive
state-progress deltas can add small credit. Heldout acceptance still uses the
strict binary reward.

The staged depth-1 solve-direction prompts reveal the affected face and ask only
for the direction decision. They do not print the scramble, inverse solution,
answer candidate, winner, overlap count, or precomputed best action. This is a
curriculum scaffold, not the final naturalistic Megaminx benchmark.

## Why We Were Stuck

The core simulator was not the main blocker. The blockers were the action
contract and the direction representation:

- Dense reward let no-action rollouts receive nonzero credit.
- Qwen chat-completions evals often produced JSON or private reasoning text
  instead of native `tool_calls`.
- Early prompts keyed the direction table to the scramble direction, so models
  often chose the scramble direction rather than the inverse solving direction.
- Served chat evals and Hosted Training did not measure the same protocol lane.
- Native malformed/private metadata accounting needed extra hardening before
  strict metrics could be trusted.

The v0.2.22 prompt breakthrough changed the table from hidden scramble-direction
framing to candidate solving-action framing:

```text
choose cw  only if current strips match expected_current_if_solve_cw
choose ccw only if current strips match expected_current_if_solve_ccw
```

That moved `Qwen/Qwen3.5-35B-A3B` from `0.504` solved to `0.929` solved on the
same 240-example depth-1 heldout set.

## Renderer/TITO Boundary

Prime's renderers post is directly relevant to this environment. A depth-1
Megaminx rollout often hinges on one exact structured tool call:

```json
{"tool":"rotate","args":{"face":"A","direction":"cw"}}
```

If the system parses, repairs, re-renders, or rewrites that sampled assistant
turn, the trainer may no longer be learning from the exact token stream that
produced the action. For that reason, the final evidence is separated into two
lanes:

| Lane | Prompt | Tool parsing | What it means |
| --- | --- | --- | --- |
| Native TITO training/probe | `stage_solve_direction_flow_native_tool` | Native `tool_calls`, no text fallback | Credible RL trajectory lane |
| Served chat eval compatibility | `stage_solve_direction_flow_json_action` | JSON/private text fallback | Useful eval history, not native tool measurement |

Hosted eval attempts with the token client failed on backend parameter support:

| Eval | Failure |
| --- | --- |
| [`kp2y6tcx8zi7kb8va00r8yg5`](https://app.primeintellect.ai/dashboard/evaluations/kp2y6tcx8zi7kb8va00r8yg5) | `enable_thinking` unsupported |
| [`drthkkm81rx02v5okhr9ck9x`](https://app.primeintellect.ai/dashboard/evaluations/drthkkm81rx02v5okhr9ck9x) | `return_token_ids` unsupported |

The practical workaround was to use Hosted Training itself as the native probe
runner with `rollouts_per_example=4`, `skip_verification=true`, one heldout
batch, and very small learning rate. This produced clean native tool metrics
without relying on the served chat endpoint.

## Validation

Latest local test command:

```bash
uv run pytest -q
```

Result:

```text
102 passed in 7.65s
```

Coverage includes:

- topology counts: 12 centers, 30 edges, 20 corners, 132 stickers
- move multiset preservation
- `cw + ccw` identity and five-turn face identity
- generated scramble plus inverse solve
- split depth bands
- invalid tool safety
- strict binary and strict shaped reward behavior
- clean geometry-frontier reward behavior
- strict relative-flow frontier reward behavior
- printed relative-flow rule policy beats all fixed slot/direction shortcuts
- renderer-friendly one-shot candidate tool handling
- native/text/private tool accounting
- no direct answer leakage in staged prompts
- package metadata matching the pushed v0.2.54 environment
- v0.2.54 two-call candidate-path refresh and scripted depth-2 solve

Latest package lifecycle commands:

```bash
prime env push megaminx-solver --path ./environments --owner setrf --visibility PUBLIC --plain
prime env install setrf/megaminx-solver@0.2.54 --plain
prime env status setrf/megaminx-solver --plain
```

Observed pushed package:

| Field | Value |
| --- | --- |
| Version | `0.2.54` |
| Wheel SHA256 | `9a1edd7195f08a516596efd772ef8729149b1332d7bc885e090eb52613289748` |

Visibility blocker: public pushes succeed, but the existing Hub record still
reports `PRIVATE`. The owner should flip environment id
`ozde27sytxjkc3wm83zv4e2c` in the web UI rather than deleting/recreating it.

## Key Evaluations

All canonical evals below were run with `prime eval run` without an explicit
`--skip-upload`.

| Eval | Version | Model | Prompt/reward | Samples | Solved/reward | Face correct | Direction correct | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| [`dgu4mymqqk5sy97l3kvusjxu`](https://app.primeintellect.ai/dashboard/evaluations/dgu4mymqqk5sy97l3kvusjxu) | `0.2.21` | `Qwen/Qwen3.5-35B-A3B` | direction-flow / binary | 240 | `0.504` | `1.000` | `0.504` | inversion trap |
| [`vse6uoo8c9y156svyyv41qll`](https://app.primeintellect.ai/dashboard/evaluations/vse6uoo8c9y156svyyv41qll) | `0.2.22` | `Qwen/Qwen3.5-35B-A3B` | solve-direction / binary | 240 | `0.929` | `1.000` | `0.929` | prompt breakthrough |
| [`koesfnz4tt3362o6uuo2lcmw`](https://app.primeintellect.ai/dashboard/evaluations/koesfnz4tt3362o6uuo2lcmw) | `0.2.22` | `Qwen/Qwen3.5-0.8B` | solve-direction / binary | 240 | `0.483` | `0.996` | `0.487` | chance direction target |
| [`o6wtoz32jf4k2vgoelskascb`](https://app.primeintellect.ai/dashboard/evaluations/o6wtoz32jf4k2vgoelskascb) | `0.2.22` | `Qwen/Qwen3.5-9B` | solve-direction / binary | 240 | `0.613` | `0.958` | `0.638` | capacity target |

Served JSON adapter evals after native shaped training:

| Eval | Adapter | Samples | Solved/reward | Face correct | Direction correct | Tool lane |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [`famcyb5lhltq34315fwh2s01`](https://app.primeintellect.ai/dashboard/evaluations/famcyb5lhltq34315fwh2s01) | step 10 `ucxz9y...` | 240 | `0.6250` | `0.9708` | `0.6375` | JSON/private text |
| [`qxyw1q2mca8gpnro6nfwnf99`](https://app.primeintellect.ai/dashboard/evaluations/qxyw1q2mca8gpnro6nfwnf99) | step 20 `sotz5v...` | 240 | `0.5750` | `0.9540` | `0.6080` | JSON/private text |
| [`udljtwz3bambq89p95dbp6oa`](https://app.primeintellect.ai/dashboard/evaluations/udljtwz3bambq89p95dbp6oa) | step 30 `a4vfj...` | 240 | `0.6208` | `0.9500` | `0.6417` | JSON/private text |

Served JSON did not produce a reliable trained-adapter improvement. The best
historical served adapter remains [`n0pl7ezrnyner8b57zdtx30k`](https://app.primeintellect.ai/dashboard/evaluations/n0pl7ezrnyner8b57zdtx30k)
at `0.6708`, from an earlier adapter.

## Hosted RL

Main native shaped run:

| Field | Value |
| --- | --- |
| Run | [`n37avwtg2evd9scmdqmafbdr`](https://app.primeintellect.ai/dashboard/training/n37avwtg2evd9scmdqmafbdr) |
| Model | `Qwen/Qwen3.5-9B` |
| Env | `setrf/megaminx-solver@0.2.28` |
| Config | `configs/rl/megaminx-depth1-qwen-9b-solve-flow-shaped-v028-native-noeval.toml` |
| Status | Stopped with `prime train stop ... --force --plain` |
| Stopped at | step 40/42 area after step-40 checkpoint |
| Total usage | 30.98M tokens |
| Total cost | `$12.40` |

Key hyperparameters:

```toml
max_steps = 140
batch_size = 256
rollouts_per_example = 16
learning_rate = 3e-7
lora_alpha = 16
oversampling_factor = 1.0
max_async_level = 1

[sampling]
max_tokens = 48
temperature = 0.7
enable_thinking = false

[env.args]
split = "train_depth1"
num_examples = 960
seed = 42
max_turns = 2
move_budget = 1
reward_style = "action_gated_strict_shaped_direction"
prompt_style = "stage_solve_direction_flow_native_tool"
allow_text_tool_actions = false
```

Representative online metrics:

| Step | Reward | Solved | Native tool calls | Text tool actions | Errors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.8547` | `0.7578` | `1.0` | `0.0` | `0` |
| 11 | `0.9063` | `0.8438` | `1.0` | `0.0` | `0` |
| 26 | `0.8922` | `0.8203` | `1.0` | `0.0` | `0` |
| 37 | `0.9000` | `0.8333` | `1.0` | `0.0` | `0` |
| 40 | `0.8406` | `0.7344` | `1.0` | `0.0` | `0` |
| 42 | `0.8525` | `0.7542` | `1.0` | `0.0` | `0` |

Checkpoints:

| Step | Checkpoint | Adapter |
| ---: | --- | --- |
| 10 | `vmrxckndxzo6t63yqrtdwnmy` | `ucxz9y00162bomqjno9ekjfb` |
| 20 | `i34jd5vb3jqz01cm1muxz4vw` | `sotz5v6zrnorx1l6xbkdiby9` |
| 30 | `wvubeopn9vy9syf8n8gnr4c2` | `a4vfj08x2lhsuqgnsbxceprd` |
| 40 | `tyrf3h6ts6non3z7imb87wgs` | `mzr7c9iwwyxuo8z903nik1id` |

Native heldout probes:

| Probe run | Checkpoint | Reward/solved | Face correct | Direction correct | Native tools | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [`gwfcyfztn8ahpwceu9n63rzd`](https://app.primeintellect.ai/dashboard/training/gwfcyfztn8ahpwceu9n63rzd) | base | `0.781893` | `0.998920` | `0.782202` | `1.0` | `$1.16` |
| [`eg68wcx3ym12mhvtku4l7g5g`](https://app.primeintellect.ai/dashboard/training/eg68wcx3ym12mhvtku4l7g5g) | step 30 | `0.769978` | `1.000000` | `0.769978` | `1.0` | `$1.11` |
| [`b6qh3e36i2poqghq5eum97dd`](https://app.primeintellect.ai/dashboard/training/b6qh3e36i2poqghq5eum97dd) | step 40 | `0.807119` | `0.997517` | `0.809603` | `1.0` | `$1.08` |

Probe settings:

```toml
batch_size = 960
rollouts_per_example = 4
learning_rate = 1e-9

[env.args]
split = "eval_depth1"
num_examples = 240
seed = 1042
max_turns = 2
move_budget = 1
reward_style = "action_gated_binary_direction"
prompt_style = "stage_solve_direction_flow_native_tool"
allow_text_tool_actions = false
```

Single-rollout probe attempts failed because the trainer's zero-advantage
filter needs multiple rollouts per example. The RPE=4 probe shape fixed that.

### v0.2.54 Depth-2 Candidate-Path Run

v0.2.54 is the first clean two-turn Megaminx lane. The model must call
`select_candidate` twice. After the first call, the environment recomputes the
visible candidate set and prints a refreshed relative-flow table for the new
state. The hidden scramble and inverse solution stay in metadata only.

Training config:

```toml
name = "megaminx-v054-qwen9b-rule-flow-solve2-depth2-rpe16-complete6"
model = "Qwen/Qwen3.5-9B"
max_steps = 6
batch_size = 512
rollouts_per_example = 16
learning_rate = 3e-9
lora_alpha = 16
oversampling_factor = 1.0
max_async_level = 1

[sampling]
max_tokens = 64
temperature = 0.8
enable_thinking = false

[sampling.extra_body]
tool_choice = { type = "function", function = { name = "select_candidate" } }
parallel_tool_calls = false

[env.args]
split = "train_candidate_relative_flow_rule_solve2_depth2"
min_depth = 2
max_depth = 2
num_examples = 2048
seed = 46
max_turns = 4
move_budget = 2
reward_style = "action_gated_candidate_path_solve"
prompt_style = "stage_candidate_relative_flow_rule_solve2_native_tool"
allow_text_tool_actions = false
```

Online training results:

| Step | Reward | Solved | First exact | Second exact | Frontier | Native tools | Errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.6159` | `0.5254` | `0.4531` | `0.4922` | `0.5996` | `2.0` | `0` |
| 1 | `0.6272` | `0.5323` | `0.4496` | `0.4980` | `0.6058` | `2.0` | `0` |
| 2 | `0.6027` | `0.5176` | `0.4258` | `0.4570` | `0.5820` | `2.0` | `0` |
| 3 | `0.5958` | `0.5121` | `0.4456` | `0.4919` | `0.5625` | `2.0` | `0` |
| 4 | `0.6284` | `0.5488` | `0.4727` | `0.5566` | `0.5996` | `2.0` | `0` |
| 5 | `0.7335` | `0.6615` | `0.4188` | `0.4396` | `0.7146` | `2.0` | `0` |

Run summary:

| Field | Value |
| --- | --- |
| Train run | [`bg0vbir6u6d521qcr8kghvvv`](https://app.primeintellect.ai/dashboard/training/bg0vbir6u6d521qcr8kghvvv) |
| Status | `COMPLETED` |
| Cost | `$7.70` |
| READY checkpoint | `ce4skj0ockwhx4zq7ztutsap` at step 3 |
| Uploading checkpoint | `gr3gpjstbttjy7xudpwnuyxf` at step 4 |

Heldout probes on the same heldout seed/config:

| Probe | Checkpoint | Reward | Solved | First exact | Second exact | Frontier | Native tools | Errors | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [`xx4pql9agtfl2se7046brmio`](https://app.primeintellect.ai/dashboard/training/xx4pql9agtfl2se7046brmio) | base | `0.6335` | `0.5625` | `0.3780` | `0.4587` | `0.5887` | `2.0` | `0` | `$1.33` |
| [`k2j8tjj7gra5ukmzmrff9epu`](https://app.primeintellect.ai/dashboard/training/k2j8tjj7gra5ukmzmrff9epu) | `ce4skj0ockwhx4zq7ztutsap` | `0.6282` | `0.5746` | `0.3599` | `0.4073` | `0.5968` | `2.0` | `0` | `$1.33` |
| [`j4y1xf2i6cg3lw40m8e3yom9`](https://app.primeintellect.ai/dashboard/training/j4y1xf2i6cg3lw40m8e3yom9) | `lbujflb1zyzv764lh9dhzu3s` | `0.6631` | `0.6048` | `0.3891` | `0.4304` | `0.6331` | `2.0` | `0` | `$1.32` |
| [`lj8g10rkj4haatuhq1iqxqzn`](https://app.primeintellect.ai/dashboard/training/lj8g10rkj4haatuhq1iqxqzn) | base, seed 246 | `0.6707` | `0.5723` | `0.4902` | `0.5078` | `0.6836` | `2.0` | `0` | `$1.59` |
| [`jiwkc1h58uwyeiyc8z7pnka9`](https://app.primeintellect.ai/dashboard/training/jiwkc1h58uwyeiyc8z7pnka9) | `lbujflb1zyzv764lh9dhzu3s`, seed 246 | `0.6712` | `0.5801` | `0.4648` | `0.4805` | `0.6738` | `2.0` | `0` | `$1.32` |

The second probe is the cleanest v0.2.54 result: reward improves by `+2.96pp`
and solved rate by `+4.23pp` over the heldout base, with the same model,
heldout seed, environment version, prompt, reward, batch size, sampling settings,
and native two-call tool path. This does not meet the original aggressive
`+30pp` acceptance target, but it is a reproducible positive RL checkpoint on
the harder two-turn depth-2 environment. Across the two heldout seeds, reward
improved by about `+1.50pp` on average and solved rate by about `+2.50pp`.

A later lower-LR/batch-1024 diagnostic run
[`vm2servi2dsv54ynuo2kxebl`](https://app.primeintellect.ai/dashboard/training/vm2servi2dsv54ynuo2kxebl)
reached step-1 reward `0.6818` and solved `0.5813` before any weight update,
then was stopped while waiting on the first checkpoint. It is not counted as
trained-checkpoint evidence.

## Reproducibility Commands

Install and test:

```bash
prime env install setrf/megaminx-solver@0.2.54 --plain
uv run pytest -q
```

Push the environment:

```bash
prime env push megaminx-solver --path ./environments --owner setrf --visibility PUBLIC --plain
prime env status setrf/megaminx-solver --plain
```

Run the served JSON compatibility heldout:

```bash
prime eval run setrf/megaminx-solver@0.2.29 -m Qwen/Qwen3.5-9B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"eval_depth1","num_examples":240,"seed":1042,"max_turns":2,"move_budget":1,"reward_style":"action_gated_binary_direction","prompt_style":"stage_solve_direction_flow_json_action","allow_text_tool_actions":true}' \
  -n 240 -r 1 -t 48 -T 0.7 -A --plain
```

Run native shaped training:

```bash
prime train configs/rl/megaminx-depth1-qwen-9b-solve-flow-shaped-v028-native-noeval.toml --yes --plain
```

Monitor training:

```bash
prime train progress n37avwtg2evd9scmdqmafbdr --plain
prime train metrics n37avwtg2evd9scmdqmafbdr --plain -n 40
prime train logs n37avwtg2evd9scmdqmafbdr --plain --tail 160
prime train distributions n37avwtg2evd9scmdqmafbdr --plain
prime train usage n37avwtg2evd9scmdqmafbdr --plain
prime train checkpoints n37avwtg2evd9scmdqmafbdr --plain
```

Run native heldout probes:

```bash
prime train configs/rl/megaminx-v028-native-heldout-probe-base.toml --yes --plain
prime train configs/rl/megaminx-v028-native-heldout-probe-step30.toml --yes --plain
prime train configs/rl/megaminx-v028-native-heldout-probe-step40.toml --yes --plain
```

## Completion Audit

Concrete success criteria from the active goal and finish plan:

| Requirement | Evidence | Status |
| --- | --- | --- |
| Prime Lab/Verifiers environment exposes `load_environment(...)` | `megaminx_solver.py` exports `load_environment`, tests import it repeatedly, and `prime env install setrf/megaminx-solver@0.2.54 --plain` succeeds | Passed |
| Stateful Megaminx tool environment | `MegaminxEnv(vf.StatefulToolEnv)` with persistent rollout state, `rotate`, `inspect`, `finish`, and candidate tools | Passed |
| v0.2.54 two-turn depth-2 task | `stage_candidate_relative_flow_rule_solve2_native_tool` refreshes candidate table after the first `select_candidate`; scripted depth-2 solve test passes | Passed |
| Hidden answer not leaked in prompts | Prompt tests assert no direct answer leakage; scramble/inverse stay in metadata | Passed |
| Unit/environment tests cover simulator and RL reward behavior | `uv run pytest -q` -> `102 passed in 7.65s` | Passed |
| Hub package pushed and installable | `prime env status setrf/megaminx-solver --plain` reports latest version `0.2.54`, action `SUCCESS`; install command succeeds | Passed |
| Hub environment public | CLI still reports visibility `PRIVATE` after public pushes; no separate CLI visibility update exists | Blocked |
| Hosted RL run completed | `bg0vbir6u6d521qcr8kghvvv` completed with final online reward `0.7335`, solved `0.6615`, zero tool/protocol/env errors | Passed |
| Probeable trained checkpoint exists | `lbujflb1zyzv764lh9dhzu3s` from `junxsn2n4rz3uvkcl88ru2in` is READY and was probed | Passed |
| Heldout checkpoint improves base | Seed 146: reward `0.6335` -> `0.6631`, solved `0.5625` -> `0.6048`; seed 246: reward `0.6707` -> `0.6712`, solved `0.5723` -> `0.5801` | Passed, modest |
| Original `+30pp` solved target | Best v0.2.54 two-seed average solved gain is about `+2.50pp` | Failed |
| Final report with commands, costs, limitations | This report includes design, rewards, runs, heldout table, costs, reproduction commands, and limitations | Passed |
| CI with `uv run pytest` on PRs | `.github/workflows/ci.yml` runs `uv run pytest` on `pull_request` | Passed |
| License | `LICENSE` is MIT; root and env `pyproject.toml` declare MIT | Passed |
| PR/merge/tag | PR [`#3`](https://github.com/setrf/megaminx-world-model-bench/pull/3) is open from `codex/megaminx-rl-crack-v2`; merge/tag remain pending until CI/review | Partial |

Audit conclusion: the technical RL objective is partially achieved with a real
but modest positive checkpoint on a harder two-turn depth-2 task. The original
aggressive `+30pp` acceptance target and public Hub visibility are not achieved.
The repository release steps are partially complete: changes are committed and
pushed, and PR `#3` is open. Merge and release tag remain pending until CI and
review are complete.

## Acceptance Status

| Criterion | Status |
| --- | --- |
| Public Hub env works | Partially blocked: owner-auth works, visibility still reports `PRIVATE` |
| Latest Hub package installs | Passed at `0.2.54` |
| Local tests pass | Passed: `102 passed in 7.65s` |
| Env errors in native probes | Passed: zero errors |
| Native tool calls nonzero | Passed: depth-1 probes use `1.0`; v0.2.54 depth-2 probes use `2.0` |
| Trained checkpoint improves depth-1 solved by `+30pp` | Failed: best native gain is about `+2.52pp` |
| v0.2.54 depth-2 checkpoint improves heldout reward | Passed: tail-room checkpoint reward `0.6631` vs base `0.6335` |
| v0.2.54 depth-2 checkpoint improves heldout solved | Passed: tail-room checkpoint solved `0.6048` vs base `0.5625` |
| Easy reward improves over base | Not completed after depth-1 failed large-gain gate |
| Commands/results reproducible | Passed for package, train config, and native probe shape |

Finished state so far: environment released and hardened, native-tool RL
measurement working, scaffolded positive native result documented, clean
v0.2.50 checkpoint probe negative, v0.2.51 conservative training stopped,
v0.2.52 strict relative-flow training stayed below its baseline, and v0.2.53
rule-flow established the strongest clean Qwen 9B one-shot baseline. v0.2.54
then cracked the two-turn depth-2 environment shape: online reward reached
`0.7335` and solved `0.6615`, and a READY tail-room checkpoint improved heldout
reward from `0.6335` to `0.6631` and solved rate from `0.5625` to `0.6048`.
The original `+30pp` goal remains unmet, but the current finish state is a real
positive RL result on a harder two-turn environment plus a reproducible report.

## Costs

| Run | Cost |
| --- | ---: |
| Main native shaped run `n37avwtg2evd9scmdqmafbdr` | `$12.40` |
| Native base probe `gwfcyfztn8ahpwceu9n63rzd` | `$1.16` |
| Native step-30 probe `eg68wcx3ym12mhvtku4l7g5g` | `$1.11` |
| Native step-40 probe `b6qh3e36i2poqghq5eum97dd` | `$1.08` |
| v0.2.54 solve-2 train `bg0vbir6u6d521qcr8kghvvv` | `$7.70` |
| v0.2.54 heldout base `xx4pql9agtfl2se7046brmio` | `$1.33` |
| v0.2.54 ckpt3 heldout `k2j8tjj7gra5ukmzmrff9epu` | `$1.33` |
| v0.2.54 tail-room train `junxsn2n4rz3uvkcl88ru2in` | `$5.51` at stop poll |
| v0.2.54 tail-room ckpt2 heldout `j4y1xf2i6cg3lw40m8e3yom9` | `$1.32` |
| v0.2.54 heldout2 base `lj8g10rkj4haatuhq1iqxqzn` | `$1.59` |
| v0.2.54 heldout2 ckpt2 `jiwkc1h58uwyeiyc8z7pnka9` | `$1.32` |
| v0.2.54 lower-LR batch-1024 run `vm2servi2dsv54ynuo2kxebl` | `$5.82`; stopped after step 1 because checkpoint wait blocked post-update evidence |

## Limitations And Next Experiments

- The older staged depth-1 prompt reveals the face and isolates direction. It
  remains useful as a curriculum probe, not as a naturalistic full Megaminx
  benchmark.
- The clean geometry-frontier prompt removes the mask proxy and hidden-target
  candidate insertion, but the first clean checkpoint probe regressed on
  heldout reward.
- Served chat-completions adapter evals do not measure native tool-call behavior
  for these Qwen adapters.
- Hosted eval could not currently run the token-client native lane because the
  backend rejected required token-return parameters.
- The next training attempt should be gated by heldout probes, not online reward
  alone: the v0.2.50 run showed online movement that did not generalize, and
  the v0.2.53 run produced one positive online step but no cloud-visible
  checkpoint for heldout probing.
- The depth-2-only Qwen 9B run did not crack the harder transition: reward
  ended at `0.5412` after starting at `0.5483`, and one-turn frontier stayed
  essentially flat (`0.6973` to `0.6953`).

Recommended next experiment:

1. Keep v0.2.53 as the current clean package baseline.
2. Probe checkpoint `lle93v210pcgw6kawfr2mr3e` from run
   `hg24yiykjdrvounjsg6bd6si` once Prime marks it `READY`; it was still
   `UPLOADING` at the latest poll.
3. Probe v0.2.53 checkpoint `pxum4a0nnvfuq5jssm28qgwa` from
   `ycx5aoe58lko3gvqyyto7794` if it becomes `READY`, but do not treat it as a
   likely breakthrough; the online curve was flat to negative.
4. If neither checkpoint becomes cloud-visible, treat the completed short run
   as an online-only positive signal and switch away from more low-LR PPO on the
   same distribution: generate supervised warm-start data from the printed
   relative-flow rule, filter/oversample unique depth-2 rows, or resolve hosted
   startup for the larger/model-family-flip configs before another long run.

## May 13 Candidate-Curriculum Addendum

After the original direction-flow work, we ran a rapid sequence of native-tool
curricula to find a learnable bridge from raw sticker observations to Megaminx
face selection.

Latest package:

| Field | Value |
| --- | --- |
| Hub package | `setrf/megaminx-solver@0.2.54` |
| Wheel SHA256 | `9a1edd7195f08a516596efd772ef8729149b1332d7bc885e090eb52613289748` |
| Local tests | `uv run pytest -q` -> `102 passed in 7.65s` |

Key attempts:

| Version | Idea | Representative run | Result |
| --- | --- | --- | --- |
| `0.2.40` | `select_candidate(index,direction)` among four candidate faces | `dxohucvd2ki2tjvlemfkqyeo` | clean protocol, face `0.1986` |
| `0.2.41` | expose only `select_candidate`, tighter prompt | `v7f44xxd7uur3huieya8gxs9` | face `0.3013` |
| `0.2.42` | exact validation and no rotate-hint leakage | `q6x1d0dtl3tjo03343gggy8i` | face `0.2655`; training checkpoint `d2i8rwmgm00r042dtcy8w16j` did not generalize |
| `0.2.43` | pure index-only binary bandit | `a1h34kjxrk8zux5b5ynrmlfc` | clean but random, face `0.2422` |
| `0.2.44` | candidate scorecard with non-answer geometric features | `bf37hb72kaqha4fp1cein28q` | cracked scaffold, face `0.9145`, reward `0.9277` |
| `0.2.45` | scorecard without the `frontier` bit | `qtkzn3gbi9alj12lbe7q6ijw` | still strong, face `0.8498`, reward `0.8708` |
| `0.2.46` | scorecard without scalar `support` or `frontier` | `etecohz0kxjx0hwpj06aoevq` | mask-only scaffold still clears the train gate, face `0.7034`, reward `0.7336` |
| `0.2.47` | mask-only scorecard with frontier-equivalent reward | `v6p7exy9p8h4vbek7ujvj86c` | face `0.7817`, face-frontier viable `0.8694`, action-frontier `0.1878`, reward `0.4620`; clean protocol |

Scorecard ablation:

| Version | Scaffold | Run | Reward | Face |
| --- | --- | --- | ---: | ---: |
| `0.2.44` | support + frontier scorecard | `bf37hb72kaqha4fp1cein28q` | `0.9277` | `0.9145` |
| `0.2.45` | support, no frontier | `qtkzn3gbi9alj12lbe7q6ijw` | `0.8708` | `0.8498` |
| `0.2.46` | affected-neighbor counts + direction masks only | `etecohz0kxjx0hwpj06aoevq` | `0.7336` | `0.7034` |

Capacity probes on the implicit candidate-strip prompt did not help:

| Model | Run | Face correct |
| --- | --- | ---: |
| `Qwen/Qwen3.5-35B-A3B` | `rp5f3ma0ceu325ybn1c2ifji` | `0.2248` |
| `Qwen/Qwen3.5-397B-A17B` | `abehelfmm6uxx7r3urvtxmt3` | `0.2083` |

The conclusion is sharper now: raw candidate-local strips are too implicit for
these models, including larger Qwen variants. A scorecard that exposes
geometry-derived features without printing the answer gives the model a usable
state representation. This is a scaffolded crack, not the final benchmark.

Scorecard training runs:

| Run | Config | Notes |
| --- | --- | --- |
| `fnckrnfr0o3a2jolqxu57mxu` | `configs/rl/megaminx-v044-qwen9b-candidate-scorecard-depth12-rpe16-noeval.toml` | stopped at step 9; face stayed `0.83-0.90`, zero tool/protocol errors; checkpoint `gbyb19xx8ufqn6fyo1g8a85w` at step 4 |
| `hv6ljq5jlc8w391a0q38373l` | `configs/rl/megaminx-v046-qwen9b-candidate-scorecard-mask-depth12-rpe16-noeval.toml` | stopped at step 4; best train step was step 2 reward `0.7618`, face `0.7308`, but step 4 fell to reward `0.6580`, face `0.6172`; protocol stayed clean; cost `$4.17` |
| `v6p7exy9p8h4vbek7ujvj86c` | `configs/rl/megaminx-v047-native-candidate-mask-frontier-equivalence-depth12-base-qwen9b-rpe2-fast.toml` | completed v0.2.47 baseline; reward `0.4620`, face `0.7817`, action-frontier `0.1878`, zero errors, cost `$0.38` |
| `pirt9kurev8d0okydmbo309d` | `configs/rl/megaminx-v047-qwen9b-candidate-mask-frontier-equivalence-depth12-rpe16-noeval.toml` | stopped at step 8; best step 4 reward `0.5693` vs baseline `0.4620`, direction `0.6582`, zero errors; checkpoint `ginimgkdx6okinz84klf0m98`; step 8 drifted to `0.4865`; cost `$7.43` |
| `cto44tv6sqbkynjp2g01ggpw` | `configs/rl/megaminx-v047-native-candidate-mask-frontier-equivalence-depth12-ckpt4-probe-qwen9b.toml` | completed heldout probe of checkpoint `ginimgkdx6okinz84klf0m98`; reward `0.5312` vs base `0.4620`, direction `0.5682`, zero errors, cost `$0.39` |
| `lecki2r8f3u76i9h0yk4xgkt` | `configs/rl/megaminx-v049-native-candidate-geometry-frontier-depth12-base-qwen9b-rpe2-fast.toml` against v0.2.49 | transitional baseline before hidden-target candidate removal; reward `0.3014`, solved `0.0601`, native tool calls `1.0`, zero errors |
| `dbup76z9d460x4fbcfxw8yql` | `configs/rl/megaminx-v049-native-candidate-geometry-frontier-depth12-base-qwen9b-rpe2-fast.toml` against v0.2.50 | clean geometry-frontier baseline: reward `0.3031`, solved `0.0508`, action-frontier `0.1032`, native tool calls `1.0`, zero errors |
| `dbs7pcyih846945xubanvdjr` | `configs/rl/megaminx-v049-qwen9b-candidate-geometry-frontier-depth12-rpe16-noeval.toml` | clean geometry-frontier training run from v0.2.50; rows through step 6: `0.3267`, `0.3347`, `0.2936`, `0.3166`, `0.3162`, `0.2835`, `0.3096`; native tool calls `1.0`, zero errors; stopped after heldout failed; cost `$6.16` |
| `qbrp8k55hndkrbuh4ndn4pyu` | `configs/rl/megaminx-v050-native-candidate-geometry-frontier-depth12-ckpt4-probe-qwen9b.toml` | heldout probe of clean checkpoint `wdlu817nloqo6tdheaaou0c8`; reward `0.2663` vs clean base `0.3031`, solved `0.0154`, native tool calls `1.0`, zero errors; cost `$0.35` |
| `g8egyymgds47wtb4rbieyd20` | `configs/rl/megaminx-v051-native-candidate-geometry-frontier-depth12-base-qwen9b-rpe2-fast.toml` | v0.2.51 diagnostic baseline: reward `0.3241`, solved `0.0622`, `target_face_in_candidate_set=1.0`, native tool calls `1.0`, zero errors |
| `byzwnn49pt9ztm6xiw0jumkx` | `configs/rl/megaminx-v051-qwen9b-candidate-geometry-frontier-depth12-lr1e8-b1024-rpe16-temp07.toml` | stopped at step 4; all rows were below the v0.2.51 baseline (`0.2962`, `0.2913`, `0.3014`, `0.3097`, `0.2993`), and an audit found the old reward could be beaten by a fixed slot/direction policy |
| `iwozt5azyroqtwkfztd47e6s` | `configs/rl/megaminx-v052-native-candidate-relative-flow-strict-frontier-depth12-base-qwen9b-rpe2-fast.toml` | v0.2.52 strict relative-flow baseline: reward `0.5676`, solved `0.3048`, face `0.7677`, relative-flow max `0.5882`, native tool calls `1.0`, zero errors |
| `hg24yiykjdrvounjsg6bd6si` | `configs/rl/megaminx-v052-qwen9b-candidate-relative-flow-strict-frontier-depth12-lr1e8-b1024-rpe16-temp07.toml` | stopped at step 3; online rewards `0.5557`, `0.5231`, `0.5459`, `0.5325` stayed below the v0.2.52 baseline `0.5676`; zero errors; checkpoint `lle93v210pcgw6kawfr2mr3e` still `UPLOADING` at latest poll; cost `$4.06` |
| `ax0cjhthz5w44unvvr44j712` | `configs/rl/megaminx-v052-native-candidate-relative-flow-strict-frontier-depth12-base-gpt-oss-120b-rpe2-fast.toml` | `openai/gpt-oss-120b` baseline probe failed because all RPE2 groups were zero-advantage; no environment error; cost `$0.20` |
| `hj5elbyhggckw9t975i9pdz2` | `configs/rl/megaminx-v052-qwen9b-candidate-relative-flow-strict-frontier-depth12-lr5e9-b1024-rpe32-temp07.toml` | lower-pressure Qwen 9B run stopped after step 0 reward `0.4905`, solved `0.2217`, face `0.6846`; zero errors; cost `$1.00` |
| `vxrolc00tg1h1yc8f96udnu9` | `configs/rl/megaminx-v053-native-candidate-relative-flow-rule-strict-frontier-depth12-base-qwen9b-rpe2-fast.toml` | v0.2.53 fair rule-flow baseline: reward `0.6726`, solved `0.4033`, face `0.7644`, direction `0.6978`, action-frontier `0.3178`, native tool calls `1.0`, zero errors, cost `$0.18` |
| `xpmbx7rp9ht9trhhwjq8q3zx` | `configs/rl/megaminx-v053-qwen9b-candidate-relative-flow-rule-strict-frontier-depth12-lr5e9-b1024-rpe16-temp07.toml` | stopped before updates because inline eval stayed at zero token usage |
| `vychxoaksf66c7pto4wz7rez` | `configs/rl/megaminx-v053-qwen9b-candidate-relative-flow-rule-strict-frontier-depth12-lr5e9-b1024-rpe16-noeval.toml` | pure clean v0.2.53 Qwen 9B no-eval training; step 0 reward `0.6365`, step 1 reward `0.7133`, solved `0.4456`, face `0.8065`, direction `0.7399`, action-frontier `0.3231`, step 2 reward `0.6299`; native tool calls `1.0`, zero errors; stopped after step 2; no checkpoint exposed; cost `$3.38` |
| `am22e2acsadulpdn1zjk0gcb` | `configs/rl/megaminx-v053-qwen9b-mixed-scorecard-to-relative-flow-depth12-rpe16-noeval.toml` before env names were fixed | failed config validation because duplicate env ids need unique names |
| `vtg65yvig5fstip74awggpgn` | `configs/rl/megaminx-v053-qwen9b-mixed-scorecard-to-relative-flow-depth12-rpe16-noeval.toml` | scaffold-assisted mixed run using v0.2.47 as auxiliary teacher and v0.2.53 as clean rule-flow lane; clean lane reward `0.6820` at step 0, then `0.6338` at step 1; stopped after clean degradation; no checkpoint exposed; cost `$3.03` |
| `f6ajk2cyut5om7qfb1qisz5e` | `configs/rl/megaminx-v053-qwen9b-candidate-relative-flow-rule-strict-frontier-depth12-lr5e9-b1024-rpe16-complete2.toml` | completed 2 steps naturally; step 0 reward `0.6251`, step 1 reward `0.7055`, solved `0.4083`, direction `0.7434`, native tool calls `1.0`, zero errors; no cloud checkpoint exposed; cost `$2.23` |
| `evntfyfh9m7xvkofwin76arx` | `configs/rl/megaminx-v053-qwen36-35b-rule-flow-depth2-only-rpe16-complete2.toml` | depth-2-only diagnostic with `Qwen/Qwen3.6-35B-A3B`; stopped after an apparent hosted model-start stall before token usage; cost `$0.00` |
| `owhgkpefm1wm5ibjllbeh85t` | `configs/rl/megaminx-v053-gptoss120b-rule-flow-depth2-only-rpe16-complete2.toml` | depth-2-only diagnostic with `openai/gpt-oss-120b`; stopped after an apparent hosted model-start stall before token usage; cost `$0.00` |
| `ycx5aoe58lko3gvqyyto7794` | `configs/rl/megaminx-v053-qwen9b-rule-flow-depth2-only-rpe16-complete4.toml` | depth-2-only Qwen 9B diagnostic completed 4 steps; rewards `0.5483`, `0.4986`, `0.5019`, `0.5412`; one-turn frontier `0.6973`, `0.6230`, `0.6328`, `0.6953`; first-rotate correct `0.4668`, `0.4160`, `0.4883`, `0.4805`; native tool calls `1.0`, zero errors; checkpoint `pxum4a0nnvfuq5jssm28qgwa` still `UPLOADING`; cost `$2.16` |

Leak audit note: the v0.2.47 improvement is not claimed as physical-world
generalization. Its scorecard printed local mask bits that strongly correlated
with reward, and the candidate set was seeded with hidden solution structure.
v0.2.50 is the first candidate-frontier lane intended for a cleaner learning
claim.

Reproduction commands:

```bash
prime env install setrf/megaminx-solver@0.2.54 --plain
uv run pytest -q
prime train configs/rl/megaminx-v049-native-candidate-geometry-frontier-depth12-base-qwen9b-rpe2-fast.toml --yes --plain
prime train configs/rl/megaminx-v049-qwen9b-candidate-geometry-frontier-depth12-rpe16-noeval.toml --yes --plain
prime train configs/rl/megaminx-v051-native-candidate-geometry-frontier-depth12-base-qwen9b-rpe2-fast.toml --yes --plain
prime train configs/rl/megaminx-v051-qwen9b-candidate-geometry-frontier-depth12-lr1e8-b1024-rpe16-temp07.toml --yes --plain
prime train configs/rl/megaminx-v052-native-candidate-relative-flow-strict-frontier-depth12-base-qwen9b-rpe2-fast.toml --yes --plain
prime train configs/rl/megaminx-v052-qwen9b-candidate-relative-flow-strict-frontier-depth12-lr1e8-b1024-rpe16-temp07.toml --yes --plain
prime train configs/rl/megaminx-v052-native-candidate-relative-flow-strict-frontier-depth12-base-gpt-oss-120b-rpe2-fast.toml --yes --plain
prime train configs/rl/megaminx-v052-qwen9b-candidate-relative-flow-strict-frontier-depth12-lr5e9-b1024-rpe32-temp07.toml --yes --plain
prime train configs/rl/megaminx-v053-native-candidate-relative-flow-rule-strict-frontier-depth12-base-qwen9b-rpe2-fast.toml --yes --plain
prime train configs/rl/megaminx-v053-qwen9b-candidate-relative-flow-rule-strict-frontier-depth12-lr5e9-b1024-rpe16-noeval.toml --yes --plain
prime train configs/rl/megaminx-v053-qwen9b-candidate-relative-flow-rule-strict-frontier-depth12-lr5e9-b1024-rpe16-complete2.toml --yes --plain
prime train configs/rl/megaminx-v053-qwen36-35b-rule-flow-depth2-only-rpe16-complete2.toml --yes --plain
prime train configs/rl/megaminx-v053-gptoss120b-rule-flow-depth2-only-rpe16-complete2.toml --yes --plain
prime train configs/rl/megaminx-v054-qwen9b-rule-flow-solve2-depth2-rpe16-complete6.toml --yes --plain
prime train configs/rl/megaminx-v054-qwen9b-rule-flow-solve2-depth2-heldout-base-rpe16.toml --yes --plain
prime train configs/rl/megaminx-v054-qwen9b-rule-flow-solve2-depth2-rpe16-tailroom8.toml --yes --plain
prime train configs/rl/megaminx-v054-qwen9b-rule-flow-solve2-depth2-tailroom-ckpt2-heldout-rpe16.toml --yes --plain
prime train configs/rl/megaminx-v044-native-candidate-scorecard-depth12-base-qwen9b-rpe2-fast.toml --yes --plain
prime train configs/rl/megaminx-v045-native-candidate-scorecard-no-frontier-depth12-base-qwen9b-rpe2-fast.toml --yes --plain
prime train configs/rl/megaminx-v046-native-candidate-scorecard-mask-depth12-base-qwen9b-rpe2-fast.toml --yes --plain
prime train configs/rl/megaminx-v047-native-candidate-mask-frontier-equivalence-depth12-base-qwen9b-rpe2-fast.toml --yes --plain
prime train configs/rl/megaminx-v044-qwen9b-candidate-scorecard-depth12-rpe16-noeval.toml --yes --plain
prime train configs/rl/megaminx-v046-qwen9b-candidate-scorecard-mask-depth12-rpe16-noeval.toml --yes --plain
prime train configs/rl/megaminx-v047-qwen9b-candidate-mask-frontier-equivalence-depth12-rpe16-noeval.toml --yes --plain
prime train configs/rl/megaminx-v047-native-candidate-mask-frontier-equivalence-depth12-ckpt4-probe-qwen9b.toml --yes --plain
```

Next step: treat the v0.2.53 rule-flow baseline as the new clean comparison
point. It improved the clean baseline from v0.2.52 reward `0.5676` and solved
`0.3048` to reward `0.6726` and solved `0.4033` without protocol errors. Pure
v0.2.53 hosted RL produced one positive online step (`0.7133`) but fell back on
step 2 and exposed no probeable checkpoint. The mixed teacher run briefly beat
the clean baseline on its clean lane (`0.6820`) and then degraded (`0.6338`), so
it was stopped. Two orthogonal depth-2-only attempts on Qwen 3.6 35B and
gpt-oss-120b stalled before token usage and were stopped at `$0.00`; the Qwen
9B depth-2-only run completed, but reward ended below its start (`0.5412` versus
`0.5483`) and frontier was flat. The next real experiment should not be more
low-LR PPO on this same distribution; it should generate an explicit oracle/SFT
warm start from the fair relative-flow rule, filter or oversample unique
depth-2 rows, or first resolve hosted startup for the larger/model-family-flip
configs, then evaluate the adapter on clean v0.2.53 heldout.
