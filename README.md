# Megaminx World Model Bench

[![Prime Hub](https://img.shields.io/badge/Prime%20Hub-setrf%2Fmegaminx--solver-blue)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![GitHub branch](https://img.shields.io/badge/GitHub-codex%2Fmegaminx--rl--environment-black)](https://github.com/setrf/megaminx-world-model-bench/tree/codex/megaminx-rl-environment)
[![Latest eval](https://img.shields.io/badge/Latest%20eval-v0.2.20%20stage%20direction-16a34a)](https://app.primeintellect.ai/dashboard/evaluations/y78em945ey7fb8ijmxbej5mg)

Prime Lab workspace for a trainable Megaminx RL environment. The v0 environment
is `megaminx-solver`, a multi-turn Verifiers tool environment where LLMs solve
short Megaminx scrambles from text facelet observations.

## Links

- Prime Hub: [`setrf/megaminx-solver`](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
- GitHub branch: [`codex/megaminx-rl-environment`](https://github.com/setrf/megaminx-world-model-bench/tree/codex/megaminx-rl-environment)
- Latest eval: [`Qwen/Qwen3.5-35B-A3B` v0.2.20 staged direction baseline](https://app.primeintellect.ai/dashboard/evaluations/y78em945ey7fb8ijmxbej5mg)
- Previous naturalistic curriculum eval: [`Qwen/Qwen3.5-35B-A3B` v0.2.19 candidate-strips baseline](https://app.primeintellect.ai/dashboard/evaluations/c04ubpjtt27x9f74tc6130u6)
- Best clean baseline: [`Qwen/Qwen3.5-35B-A3B` v0.2.17 static match-table smoke](https://app.primeintellect.ai/dashboard/evaluations/k16letr5pyhopqwu0e5seshs)
- Historical indexed direction eval: [`Qwen/Qwen3.5-35B-A3B` v0.2.18 indexed-direction baseline](https://app.primeintellect.ai/dashboard/evaluations/gnudftur8j76ndbokfpas7jq)
- Report: [`reports/megaminx-rl-report.md`](reports/megaminx-rl-report.md)
- Latest package: `setrf/megaminx-solver@0.2.20`
- Install: `prime env install setrf/megaminx-solver@0.2.20`

## Current Local Loop

```bash
prime env install setrf/megaminx-solver@0.2.20
prime eval run setrf/megaminx-solver@0.2.20 -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"eval_depth1","num_examples":240,"seed":1042,"max_turns":2,"move_budget":1,"reward_style":"action_gated_exact_direction","prompt_style":"stage_face_hint_direction_json_action","allow_text_tool_actions":true}' \
  -n 240 -r 1 -t 48 -T 0.7 -A
```

Depth-1 staged direction eval:

```bash
prime eval run setrf/megaminx-solver@0.2.20 -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"eval_depth1","num_examples":240,"seed":1042,"max_turns":2,"move_budget":1,"reward_style":"action_gated_exact_direction","prompt_style":"stage_face_hint_direction_json_action","allow_text_tool_actions":true}' \
  -n 240 -r 1 -t 48 -T 0.7 -A
```

Hosted training config:

```bash
prime train configs/rl/megaminx-depth1-qwen-35b-a3b-face-hint-direction.toml --yes --plain
```

The environment package lives at `environments/megaminx_solver/`.

Current training path is an explicitly staged direction curriculum. It uses
candidate-local strip evidence plus a face hint so the model only has to learn
`cw` versus `ccw`; it is not a final naturalistic evaluation prompt. The latest
held-out staged eval scored `0.579` reward with `0.504` solved/direction
accuracy, `1.000` face accuracy, and zero env errors or illegal moves. Qwen's
hosted chat path reliably emits JSON in text/reasoning fields, so JSON action
styles remain the pragmatic path for this experiment. Native tool prompt styles
are kept for renderer/TITO work, where sampled tool-call tokens should remain
the source of truth across turns.
