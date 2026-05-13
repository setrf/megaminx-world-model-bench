# Megaminx World Model Bench

[![Prime Hub](https://img.shields.io/badge/Prime%20Hub-setrf%2Fmegaminx--solver-blue)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![GitHub branch](https://img.shields.io/badge/GitHub-codex%2Fmegaminx--rl--environment-black)](https://github.com/setrf/megaminx-world-model-bench/tree/codex/megaminx-rl-environment)
[![Smoke eval](https://img.shields.io/badge/Smoke%20eval-v0.2.17%20match%20table-16a34a)](https://app.primeintellect.ai/dashboard/evaluations/k16letr5pyhopqwu0e5seshs)

Prime Lab workspace for a trainable Megaminx RL environment. The v0 environment
is `megaminx-solver`, a multi-turn Verifiers tool environment where LLMs solve
short Megaminx scrambles from text facelet observations.

## Links

- Prime Hub: [`setrf/megaminx-solver`](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
- GitHub branch: [`codex/megaminx-rl-environment`](https://github.com/setrf/megaminx-world-model-bench/tree/codex/megaminx-rl-environment)
- Latest smoke eval: [`Qwen/Qwen3.5-35B-A3B` v0.2.17 static match-table smoke](https://app.primeintellect.ai/dashboard/evaluations/k16letr5pyhopqwu0e5seshs)
- Report: [`reports/megaminx-rl-report.md`](reports/megaminx-rl-report.md)
- Install: `prime env install setrf/megaminx-solver@0.2.17`

## Local Loop

```bash
prime env install setrf/megaminx-solver
prime eval run setrf/megaminx-solver -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"depth1","num_examples":120,"max_turns":2,"reward_style":"action_gated_curriculum","prompt_style":"sensor_match_json_action"}' \
  -n 120 -r 1 -t 48 -T 0.7 -A
```

Depth-1 RL smoke eval:

```bash
prime eval run setrf/megaminx-solver -m Qwen/Qwen3.5-35B-A3B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":{"type":"function","function":{"name":"rotate"}}}' \
  -a '{"split":"depth1","num_examples":120,"max_turns":2,"reward_style":"action_gated_curriculum","prompt_style":"sensor_match_json_action"}' \
  -n 120 -r 1 -t 48 -T 0.7 -A
```

Hosted training config:

```bash
prime train configs/rl/megaminx-depth1-qwen-35b-a3b-sensor-sync.toml --yes --plain
```

The environment package lives at `environments/megaminx_solver/`.

Current training path uses the JSON action prompt because the Qwen hosted chat
path reliably emits JSON in text/reasoning fields. Native tool prompt styles are
kept for renderer/TITO work, where sampled tool-call tokens should remain the
source of truth across turns.
