# Megaminx World Model Bench

[![Prime Hub](https://img.shields.io/badge/Prime%20Hub-setrf%2Fmegaminx--solver-blue)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![GitHub branch](https://img.shields.io/badge/GitHub-codex%2Fmegaminx--rl--environment-black)](https://github.com/setrf/megaminx-world-model-bench/tree/codex/megaminx-rl-environment)
[![Smoke eval](https://img.shields.io/badge/Smoke%20eval-action--gated%204B-7c3aed)](https://app.primeintellect.ai/dashboard/evaluations/dq7ajgaiw0vdi6fz284a8ulp)

Prime Lab workspace for a trainable Megaminx RL environment. The v0 environment
is `megaminx-solver`, a multi-turn Verifiers tool environment where LLMs solve
short Megaminx scrambles from text facelet observations.

## Links

- Prime Hub: [`setrf/megaminx-solver`](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
- GitHub branch: [`codex/megaminx-rl-environment`](https://github.com/setrf/megaminx-world-model-bench/tree/codex/megaminx-rl-environment)
- Latest smoke eval: [`Qwen/Qwen3.5-4B` action-gated depth-1 run](https://app.primeintellect.ai/dashboard/evaluations/dq7ajgaiw0vdi6fz284a8ulp)
- Report: [`reports/megaminx-rl-report.md`](reports/megaminx-rl-report.md)
- Install: `prime env install setrf/megaminx-solver`

## Local Loop

```bash
prime env install setrf/megaminx-solver
prime eval run setrf/megaminx-solver -m Qwen/Qwen3.5-4B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":"required"}' \
  -a '{"split":"depth1","num_examples":30,"max_turns":2,"reward_style":"action_gated_dense","prompt_style":"action_first"}' \
  -n 10 -r 3 -t 256 -A
```

Depth-1 RL smoke eval:

```bash
prime eval run setrf/megaminx-solver -m Qwen/Qwen3.5-4B \
  --api-client-type openai_chat_completions \
  -S '{"tool_choice":"required"}' \
  -a '{"split":"depth1","num_examples":30,"max_turns":2,"reward_style":"action_gated_dense","prompt_style":"action_first"}' \
  -n 10 -r 3 -t 256 -A
```

Hosted training config:

```bash
prime train configs/rl/megaminx-depth1-qwen-4b.toml --yes --plain
```

The environment package lives at `environments/megaminx_solver/`.
