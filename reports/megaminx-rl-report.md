# Megaminx RL Environment Report

Date: 2026-05-13

## Executive Summary

`setrf/megaminx-solver` is a Prime/Verifiers environment for studying whether
LLMs can act in a compact symbolic model of a physical puzzle. It defines a
Megaminx world, legal tool actions, persistent rollout state, deterministic
reward, and reproducible train/eval curricula.

The project is not at the original aggressive finish target. The environment is
hardened and pushed, the native Token-In Token-Out training lane is working, and
the best step-40 native heldout probe improves over the native base by about
`+2.52pp`. The requested `+30pp` trained-checkpoint gain was not achieved. This
is therefore a released environment plus modest positive native RL evidence and
a clear negative result on the larger target.

| Item | Value |
| --- | --- |
| GitHub repo | `setrf/megaminx-world-model-bench` |
| Working branch | `codex/megaminx-direction-breakthrough` |
| Prime owner | `setrf` |
| Hub environment | [`setrf/megaminx-solver`](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver) |
| Environment id | `ozde27sytxjkc3wm83zv4e2c` |
| Latest pushed version | `0.2.29` |
| Latest content hash | `d7be3414695e` |
| Latest wheel SHA256 | `f19e1209dabae728af50ad540528ef5c71edb838471886cb0945ca078aae8db8` |
| Latest install check | `prime env install setrf/megaminx-solver@0.2.29 --plain` succeeded |
| Latest local tests | `uv run pytest -q` -> `59 passed in 4.12s` |
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
59 passed in 4.12s
```

Coverage includes:

- topology counts: 12 centers, 30 edges, 20 corners, 132 stickers
- move multiset preservation
- `cw + ccw` identity and five-turn face identity
- generated scramble plus inverse solve
- split depth bands
- invalid tool safety
- strict binary and strict shaped reward behavior
- native/text/private tool accounting
- no direct answer leakage in staged prompts
- package metadata matching the pushed v0.2.29 environment

Latest package lifecycle commands:

```bash
prime env push megaminx-solver --path ./environments --owner setrf --visibility PUBLIC --plain
prime env install setrf/megaminx-solver@0.2.29 --plain
prime env status setrf/megaminx-solver --plain
```

Observed pushed package:

| Field | Value |
| --- | --- |
| Version | `0.2.29` |
| Hash | `d7be3414695e` |
| Wheel SHA256 | `f19e1209dabae728af50ad540528ef5c71edb838471886cb0945ca078aae8db8` |

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

## Reproducibility Commands

Install and test:

```bash
prime env install setrf/megaminx-solver@0.2.29 --plain
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

## Acceptance Status

| Criterion | Status |
| --- | --- |
| Public Hub env works | Partially blocked: owner-auth works, visibility still reports `PRIVATE` |
| Latest Hub package installs | Passed at `0.2.29` |
| Local tests pass | Passed: `59 passed` |
| Env errors in native probes | Passed: zero errors |
| Native tool calls nonzero | Passed: `native_tool_call_count=1.0` |
| Trained checkpoint improves depth-1 solved by `+30pp` | Failed: best native gain is about `+2.52pp` |
| Easy reward improves over base | Not completed after depth-1 failed large-gain gate |
| Commands/results reproducible | Passed for package, train config, and native probe shape |

Finished state: environment released and hardened, native-tool RL measurement
working, modest positive step-40 native result, no large trained-checkpoint
breakthrough yet.

## Costs

| Run | Cost |
| --- | ---: |
| Main native shaped run `n37avwtg2evd9scmdqmafbdr` | `$12.40` |
| Native base probe `gwfcyfztn8ahpwceu9n63rzd` | `$1.16` |
| Native step-30 probe `eg68wcx3ym12mhvtku4l7g5g` | `$1.11` |
| Native step-40 probe `b6qh3e36i2poqghq5eum97dd` | `$1.08` |

## Limitations And Next Experiments

- The current staged depth-1 prompt reveals the face and isolates direction. It
  is a curriculum probe, not a naturalistic full Megaminx benchmark.
- The base native heldout is already high at `0.7819`, leaving limited headroom
  for simple depth-1 RL.
- Served chat-completions adapter evals do not measure native tool-call behavior
  for these Qwen adapters.
- Hosted eval could not currently run the token-client native lane because the
  backend rejected required token-return parameters.
- The next training attempt should lower base saturation or increase signal:
  use a less scaffolded face/direction prompt, depth 1-2 curriculum, larger
  rollout groups, and a native heldout probe gate every 10 steps.

Recommended next experiment:

1. Keep v0.2.29 as the public package baseline.
2. Build `stage_solve_direction_flow_native_tool_v2` with less face leakage but
   the same strict action protocol.
3. Train `Qwen/Qwen3.5-9B` with `rollouts_per_example=16`, binary heldout
   probes every 10 steps, and stop when native heldout fails to improve for two
   checkpoints.
4. Only move to `easy` after a heldout-native gain exceeds `+10pp` on depth 1.
