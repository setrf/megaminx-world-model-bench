# Megaminx World Model Bench

[![Prime Hub](https://img.shields.io/badge/Prime%20Hub-setrf%2Fmegaminx--solver-blue)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Latest env](https://img.shields.io/badge/env-v0.2.57-0f766e)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Report](https://img.shields.io/badge/report-megaminx--rl--report-2563eb)](reports/megaminx-rl-report.md)
[![GitHub branch](https://img.shields.io/badge/branch-main-black)](https://github.com/setrf/megaminx-world-model-bench/tree/main)
[![Release tag](https://img.shields.io/badge/release-v0.2.57-0f766e)](https://github.com/setrf/megaminx-world-model-bench/releases/tag/v0.2.57)
[![License](https://img.shields.io/badge/License-MIT-0f766e)](LICENSE)

`megaminx-world-model-bench` is a Prime Intellect / Verifiers reinforcement
learning environment for testing whether language models can learn the dynamics
of a physical puzzle world through action, feedback, and repeated rollouts.

The world is a symbolic Megaminx: a twelve-faced twisty puzzle with local face
turns, persistent state, hidden scrambles, exact simulator validation, and
dense plus task-specific rewards. Models observe a compact text facelet state,
act through tools such as `rotate` or `select_candidate`, and are scored by a
deterministic simulator.

This repository contains the environment package, tests, Prime training
configs, oracle trajectory export tools, local SFT utilities, and reports from
the hosted RL and local warm-start experiments.

## Links

| Resource | Link |
| --- | --- |
| GitHub repository | <https://github.com/setrf/megaminx-world-model-bench> |
| Prime environment | <https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver> |
| Hub slug | `setrf/megaminx-solver` |
| Latest package | `setrf/megaminx-solver@0.2.57` |
| Latest report | [`reports/megaminx-rl-report.md`](reports/megaminx-rl-report.md) |
| Next-run runbook | [`reports/megaminx-next-run-runbook.md`](reports/megaminx-next-run-runbook.md) |
| License | [`MIT`](LICENSE) |

Prime now reports `setrf/megaminx-solver` as `PUBLIC` on the Environment Hub.

## Why This Exists

The larger goal is to build environments that make LLMs learn by interacting
with verifiable worlds rather than only predicting static text. Megaminx is a
useful first world because it is small enough to simulate exactly but rich
enough to test core physical-reasoning skills:

- **State persistence:** each move changes the future world state.
- **Local action effects:** a face turn changes one face and a ring of adjacent
  strips, similar to manipulating an object in space.
- **Partial observability pressure:** prompts can expose full facelets,
  compact sensors, candidate summaries, or only local evidence.
- **Planning horizon:** depth-1 tasks test action grounding; depth-2 tasks test
  whether the model can chain actions after a refreshed observation.
- **Exact verification:** every rollout has a hidden scramble and inverse
  solution used for deterministic scoring, tests, and metrics.

The result is both an RL training environment and a benchmark for comparing
prompt styles, tool protocols, model capacity, native tool-call behavior, and
warm-start data generation.

## Benchmark Claim

This benchmark measures whether an LLM can map symbolic observations of a
physical puzzle into valid actions that improve or solve that puzzle under an
exact simulator.

It is strongest as a benchmark for:

- native tool-use discipline under Prime Hosted Training
- local transition reasoning over face turns and neighboring sticker strips
- short-horizon planning from depth-1 to depth-2 scrambles
- reward design and shortcut auditing for RL environments
- comparing base models, RL checkpoints, and SFT warm starts on identical
  deterministic heldout sets

It does not yet claim:

- vision-based physical-world understanding
- full Megaminx solving at human puzzle depths
- a large hosted RL breakthrough beyond the modest heldout gains recorded here
- that older scaffolded prompt lanes are clean capability measurements

## What The Environment Does

Each rollout follows the same loop:

1. Generate a deterministic scramble from a named split, seed, and depth range.
2. Build a solved Megaminx, apply the scramble, and hide the inverse solution
   in task metadata.
3. Show the model a text observation or staged sensor prompt.
4. Let the model act through native tools or, for served compatibility evals,
   a JSON/text action fallback.
5. Update the persistent simulator state after every tool call.
6. Score the final trajectory with a reward function and a large set of
   diagnostic metrics.

The general puzzle instruction loop uses `rotate`, `inspect`, and `finish`.
The implementation defines six tools in total: `rotate`, `inspect`, `finish`,
`select_candidate`, `select_candidate_index`, and `predict_rotate`. Candidate
prompt styles automatically narrow the exposed schema to the intended
candidate tool so Prime Hosted Training can preserve exact tool-call tokens
through Token-In Token-Out rendering.

## Environment API

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

`load_environment(...)` returns a `vf.Environment`. The implementation class is
`MegaminxEnv(vf.StatefulToolEnv)`, which stores the puzzle object in rollout
state so each tool call changes the same world.

The Python package exports `MegaminxEnv`, `build_dataset`, `load_environment`,
`MegaminxPuzzle`, `MegaminxTopology`, simulator constants, `generate_scramble`,
and `inverse_moves`.

| Argument | Meaning |
| --- | --- |
| `split` | Named curriculum split such as `depth1`, `easy`, `medium`, `hard`, `eval`, or a training variant. |
| `min_depth`, `max_depth` | Custom scramble depth range when the split does not override it. |
| `num_examples` | Number of deterministic examples to generate. |
| `seed` | Dataset seed. Depth-1 moves are balanced across the 24 legal actions. |
| `max_turns` | Verifiers/global turn cap. Defaults from the depth range. |
| `move_budget` | Puzzle move budget printed to the model. Defaults to `min(32, 2 * depth + 4)`. |
| `reward_style` | Dense, action-gated, staged, candidate, or tail-solve reward. |
| `prompt_style` | Observation and protocol family. |
| `allow_text_tool_actions` | Enables JSON/private-text fallback for served chat evals. Defaults to true only for JSON prompt styles. |
| `exposed_tool_names` | Optional tool subset override. Candidate prompt styles expose their matching candidate tool by default. |

## Simulator

The simulator is intentionally simple, inspectable, and testable:

- 12 faces labeled `A` through `L`.
- Each face has one fixed center, five edge sticker positions, and five corner
  sticker positions.
- 132 visible stickers are tracked as `(face, position)` entries.
- The solved state has every sticker on face `X` equal to label `X`.
- There are 30 edge pieces and 20 corner pieces.
- Each face turn is a 72-degree Megaminx turn, so five clockwise turns return
  to identity.
- The topology stores a dodecahedron neighbor ring for every face and derives
  turn-local side strips programmatically.
- Legal moves are the 24 pairs `(face in A-L, direction in cw|ccw)`.
- Scrambles avoid immediate inverse moves.

The test suite checks topology counts, sticker multiset preservation,
`cw + ccw` identity, five-turn identity, generated scramble plus inverse solve,
invalid tool safety, package metadata, staged prompt behavior, oracle export,
SFT conversion, and RL config consistency.

## Tools

The environment implements these general puzzle tools:

| Tool | Arguments | Effect |
| --- | --- | --- |
| `rotate` | `face: A-L`, `direction: cw|ccw` | Apply one Megaminx face turn. |
| `inspect` | `face: A-L|all` | Return one face or the full compact net without changing state. |
| `finish` | none | End the rollout and report solved/not solved. |

It also implements these staged research tools:

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `select_candidate` | `index: 1-4`, `direction: cw|ccw` | Rotate a candidate face selected from a visible candidate table. |
| `select_candidate_index` | `index: 1-4` | Pick a candidate face without direction for face-discovery ablations. |
| `predict_rotate` | `face`, `direction`, `predicted_after` | Predict local post-move strips and then rotate. |

With `exposed_tool_names=None`, ordinary non-candidate prompt styles leave all
implemented tools in the schema. Native candidate prompt styles expose only
their intended candidate tool, which prevents the model from mixing protocols
inside a rollout.

## Observations And Prompt Families

The base observation prints a compact text facelet net with solvedness,
sticker accuracy, piece accuracy, move budget, and last move. The hidden
scramble, inverse solution, answer, and row identifiers are stored in metadata
for scoring and tests, not exposed in prompts.

The project tried several prompt families:

| Family | Tool surface | What it tests |
| --- | --- | --- |
| `default` / `action_first` | `rotate`, `inspect`, `finish` | General multi-turn puzzle interaction. |
| JSON action styles | Text JSON fallback | Served eval compatibility when native tool calls are unreliable. |
| Direction-flow styles | `rotate` | Depth-1 face and direction grounding. |
| Native solve-flow styles | `rotate` | Hosted RL with exact native tool calls. |
| Sensor/topology styles | `rotate` | Whether structured local evidence improves grounding. |
| Candidate scorecard styles | `select_candidate` | Choosing among visible candidate faces. |
| Candidate geometry and relative-flow styles | `select_candidate` | Less-scaffolded local geometry reasoning. |
| Two-call candidate-path style | `select_candidate` twice | Depth-2 solving with refreshed observations after the first move. |
| Prediction styles | `predict_rotate` | Whether a model can predict local transition dynamics before acting. |

The current technical lane is v0.2.56. It focuses on
`stage_candidate_relative_flow_rule_solve2_native_tool` with
`action_gated_candidate_path_tail_solve`, plus deterministic oracle export for
SFT warm-start experiments.

## Rewards

The original dense reward remains available for general interaction:

```text
0.60 * solved
+ 0.25 * sticker_accuracy
+ 0.10 * piece_accuracy
+ 0.05 * efficiency_if_solved
```

For RL, dense reward alone was not enough because early baseline models often
reasoned in text instead of calling tools. The project therefore added
action-gated rewards:

- **No action, no reward:** tool-free text completions receive no useful score.
- **Strict eval gates:** binary reward is `1.0` only for a clean exact action.
- **Shaped training gates:** valid but imperfect actions receive partial credit
  from face correctness, direction correctness, frontier progress, or local
  evidence quality.
- **Candidate-path rewards:** depth-2 runs require a first candidate action,
  refreshed observation, and second candidate action that solves the puzzle.
- **Shortcut hardening:** v0.2.56 caps non-solving second-step partial reward
  below `0.50` and removes visible row-id shortcuts from refreshed candidate
  slots.

Representative reward styles include `dense`, `action_gated_dense`,
`action_gated_binary_direction`, `action_gated_strict_shaped_direction`,
`action_gated_candidate_geometry_frontier`,
`action_gated_candidate_strict_frontier`,
`action_gated_candidate_path_solve`, and
`action_gated_candidate_path_tail_solve`.

## Metrics

The environment records both task metrics and protocol metrics. Important
groups are:

- **Outcome:** `solved_rate`, final reward, sticker accuracy, piece accuracy,
  move count, selected move count, scramble depth.
- **Protocol:** native tool-call count, text fallback count, private text action
  count, parse errors, tool-call errors, protocol violations, illegal moves.
- **General actions:** rotate, inspect, finish, first action correctness,
  first face correctness, first direction correctness.
- **Candidate actions:** candidate selection count, first and second candidate
  index, face correctness, candidate path completion, target slot diagnostics.
- **Geometry diagnostics:** neighbor overlap, action-mask counts, frontier
  viability, counterfactual value/rank, relative-flow count/margin/max flags.
- **Prediction diagnostics:** strip accuracy, exact strip count, character
  accuracy, prediction validity, extra predicted items.

These metrics are the main reason the project could diagnose where RL was
stuck: not calling tools, calling text instead of native tools, overfitting a
visible shortcut, solving depth-1 but failing direction, or improving online
reward without improving heldout solves.

## Main Results

The honest result is mixed but useful.

| Result | What happened |
| --- | --- |
| Prompt breakthrough | v0.2.22 reframed depth-1 as candidate solving actions. `Qwen/Qwen3.5-35B-A3B` reached `0.929` solved on a 240-example heldout eval: <https://app.primeintellect.ai/dashboard/evaluations/vse6uoo8c9y156svyyv41qll>. |
| Native tool lane | Hosted Training's Token-In Token-Out path preserved native tool calls and gave clean zero-error probes. v0.2.28 step 40 improved reward to `0.8071` on its native heldout probe, but this was a modest movement rather than a large solved-rate jump. |
| Leak audits | v0.2.47 through v0.2.56 removed increasingly subtle scaffolds: visible masks, hidden-solution candidate seeding, fixed second-slot shortcuts, and prompt row-id leakage. |
| Two-call depth-2 lane | v0.2.54 produced a clean candidate-path run with two native `select_candidate` calls, final online reward `0.7335`, solved `0.6615`, and zero environment errors. The best matched heldout checkpoint improved solved rate from `0.5625` to `0.6048` on one seed. |
| Local SFT warm start | A small local MPS LoRA smoke for `Qwen/Qwen3.5-0.8B` trained on 16 oracle rows for 5 steps. On two 16-row heldout probes, base solved `0/32`; the adapter solved `18/32` with clean two-call parsing. This is a local warm-start signal, not a hosted Prime RL checkpoint. |
| Original finish target | The environment is released and hardened, but the original `+30 percentage point` hosted RL acceptance target was not met before Prime billing/auth limits stopped further runs. |

See [`reports/megaminx-rl-report.md`](reports/megaminx-rl-report.md) for the
full run ledger, costs, checkpoint ids, failure analysis, and links.

## For Researchers

Start with the environment package if you want to run or modify the task:
[`environments/megaminx_solver`](environments/megaminx_solver).

Use the report if you want to understand the experiment history:
[`reports/megaminx-rl-report.md`](reports/megaminx-rl-report.md).

Use the runbook if you want to continue the work after Prime auth/billing is
restored:
[`reports/megaminx-next-run-runbook.md`](reports/megaminx-next-run-runbook.md).

The most important experimental split is between native Hosted Training and
served chat eval compatibility. Native runs preserve tool-call tokens and are
the credible RL lane. Served evals are useful for public scaling history, but
they can rely on JSON/private-text action parsing and should not be mixed into
native tool-call claims.

The next serious experiment is to use the v0.2.56 oracle/SFT path as a
warm-start, train on a larger GPU than local MPS, and then launch matched
hosted Prime probes against the base model and the warm-started checkpoint.

## Reproduce Locally

Install the current Hub package if you have Prime access:

```bash
prime env install setrf/megaminx-solver@0.2.57 --plain
```

Run the local test suite from this repository:

```bash
uv run pytest -q
```

Latest recorded local result:

```text
129 passed in 45.50s
```

Export deterministic oracle trajectories for the current two-call tail-solve
lane:

```bash
uv run python scripts/export_oracle_trajectories.py \
  --num-examples 1024 \
  --seed 64 \
  --split train_candidate_relative_flow_rule_tail_solve_depth2 \
  --output /tmp/megaminx-oracle-v057-1024.jsonl

uv run python scripts/summarize_oracle_trajectories.py \
  /tmp/megaminx-oracle-v057-1024.jsonl
```

Convert those trajectories into chat/tool-call SFT rows:

```bash
uv run python scripts/convert_oracle_to_sft_jsonl.py \
  /tmp/megaminx-oracle-v057-1024.jsonl \
  --output /tmp/megaminx-oracle-v057-1024-sft.jsonl

uv run python scripts/validate_sft_jsonl.py \
  /tmp/megaminx-oracle-v057-1024-sft.jsonl
```

Check whether the repo is ready for the next hosted Prime probes:

```bash
uv run python scripts/check_next_run_readiness.py
```

Use `--check-prime` only when Prime auth and billing are restored:

```bash
uv run python scripts/check_next_run_readiness.py --check-prime
```

## Reproduce Prime Runs

The tracked configs under [`configs/rl`](configs/rl) reproduce the major hosted
lanes. The current v0.2.56 matched heldout configs are:

```bash
prime train configs/rl/megaminx-v056-qwen9b-tail-solve-depth2-base-heldout-rpe16.toml --yes --plain
prime train configs/rl/megaminx-v056-qwen9b-tail-solve-depth2-base-heldout2-rpe16.toml --yes --plain
prime train configs/rl/megaminx-v056-qwen9b-tail-solve-depth2-ckpt2-heldout-rpe16.toml --yes --plain
prime train configs/rl/megaminx-v056-qwen9b-tail-solve-depth2-ckpt2-heldout2-rpe16.toml --yes --plain
```

These were prepared but not launched canonically because Prime run creation
started returning payment/auth errors. The exact recovery sequence is in
[`reports/megaminx-next-run-runbook.md`](reports/megaminx-next-run-runbook.md).

## Project Layout

| Path | Purpose |
| --- | --- |
| [`environments/megaminx_solver`](environments/megaminx_solver) | Prime/Verifiers environment package and Prime Hub README. |
| [`environments/megaminx_solver/megaminx_solver/simulator.py`](environments/megaminx_solver/megaminx_solver/simulator.py) | Megaminx topology, pieces, moves, scrambles, and accuracy metrics. |
| [`environments/megaminx_solver/megaminx_solver/megaminx_solver.py`](environments/megaminx_solver/megaminx_solver/megaminx_solver.py) | `MegaminxEnv`, tools, prompts, reward functions, metrics, dataset builder. |
| [`tests`](tests) | Unit, environment, config, oracle, SFT, and readiness tests. |
| [`configs/rl`](configs/rl) | Hosted Prime training and probe configs. |
| [`configs/eval`](configs/eval) | Prime eval configs for depth-1 and scaling evals. |
| [`configs/sft`](configs/sft) | Local LoRA/SFT smoke configs. |
| [`scripts`](scripts) | Oracle export, SFT conversion, local training/eval, and readiness utilities. |
| [`reports/megaminx-rl-report.md`](reports/megaminx-rl-report.md) | Full technical report and experiment ledger. |
| [`reports/megaminx-next-run-runbook.md`](reports/megaminx-next-run-runbook.md) | Concrete handoff for the next hosted probes after billing/auth recovery. |

## Current Limitations

- The current environment is text-first. It tests physical-world structure
  through symbolic facelets and local actions, not rendered vision.
- Some older runs intentionally used scaffolded prompts to find a learnable
  curriculum. Later releases remove those scaffolds; compare versions before
  making capability claims.
- Served chat-completions evals and Hosted Training native tool rollouts are
  not interchangeable. Native tool-call metrics are the credible RL lane.
- The best hosted RL results were modest. The strongest learning signal so far
  is the local oracle-SFT warm start, which should be re-run on a larger GPU and
  then used to initialize hosted RL.

## License

MIT. See [`LICENSE`](LICENSE).
