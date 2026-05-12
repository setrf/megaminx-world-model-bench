# Megaminx World Model Bench

Prime Lab workspace for a trainable Megaminx RL environment. The v0 environment
is `megaminx-solver`, a multi-turn Verifiers tool environment where LLMs solve
short Megaminx scrambles from text facelet observations.

## Local Loop

```bash
prime env install megaminx-solver
prime eval run megaminx-solver -m Qwen/Qwen3.5-4B -n 10 -r 3
```

The environment package lives at `environments/megaminx_solver/`.
