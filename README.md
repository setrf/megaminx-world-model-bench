# Megaminx World Model Bench

[![Prime Hub](https://img.shields.io/badge/Prime%20Hub-setrf%2Fmegaminx--solver-blue)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Latest env](https://img.shields.io/badge/env-v0.2.56-0f766e)](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver)
[![Clean geometry](https://img.shields.io/badge/clean%20geometry-v0.2.50-16a34a)](https://app.primeintellect.ai/dashboard/training/dbup76z9d460x4fbcfxw8yql)
[![GitHub branch](https://img.shields.io/badge/branch-main-black)](https://github.com/setrf/megaminx-world-model-bench/tree/main)
[![Release tag](https://img.shields.io/badge/release-v0.2.56-0f766e)](https://github.com/setrf/megaminx-world-model-bench/releases/tag/v0.2.56)
[![License](https://img.shields.io/badge/License-MIT-0f766e)](LICENSE)

Prime Lab workspace for `setrf/megaminx-solver`, a trainable Prime/Verifiers
environment that tests whether LLMs can act in a compact symbolic Megaminx
world. The environment exposes a stateful tool loop: inspect a text facelet net,
call `rotate(face, direction)`, and receive deterministic reward from the
puzzle simulator.

## Current Status

| Item | Value |
| --- | --- |
| GitHub repo | `setrf/megaminx-world-model-bench` |
| Default branch | `main` |
| Merged PR | [`#3`](https://github.com/setrf/megaminx-world-model-bench/pull/3) from `codex/megaminx-rl-crack-v2`; latest v0.2.56 hardening is on `main` |
| Release tag | [`v0.2.56`](https://github.com/setrf/megaminx-world-model-bench/releases/tag/v0.2.56) |
| Prime owner | `setrf` |
| Hub environment | [`setrf/megaminx-solver`](https://app.primeintellect.ai/dashboard/environments/setrf/megaminx-solver) |
| Environment id | `ozde27sytxjkc3wm83zv4e2c` |
| Latest package | `setrf/megaminx-solver@0.2.56` |
| Latest Hub action | `kioezfzz4ji4uquyhm0grzwc` -> `SUCCESS` |
| Hub visibility | Still reports `PRIVATE` after `--visibility PUBLIC`; API PATCH visibility attempts return HTTP 405 |
| Latest wheel SHA256 | `f52a3858518f234c4a2df310ab465b37b536fc28ab3ad2e034373109f49e7106` |
| Latest local tests | `uv run pytest -q` -> `107 passed in 17.44s` |
| Latest scaffold baseline | [`etecohz0kxjx0hwpj06aoevq`](https://app.primeintellect.ai/dashboard/training/etecohz0kxjx0hwpj06aoevq): reward `0.7336`, face `0.7034`, zero errors |
| Stopped v0.2.46 train | [`hv6ljq5jlc8w391a0q38373l`](https://app.primeintellect.ai/dashboard/training/hv6ljq5jlc8w391a0q38373l): best step `0.7618`, final step `0.6580`, cost `$4.17` |
| v0.2.47 frontier baseline | [`v6p7exy9p8h4vbek7ujvj86c`](https://app.primeintellect.ai/dashboard/training/v6p7exy9p8h4vbek7ujvj86c): reward `0.4620`, face `0.7817`, action-frontier `0.1878` |
| Stopped v0.2.47 train | [`pirt9kurev8d0okydmbo309d`](https://app.primeintellect.ai/dashboard/training/pirt9kurev8d0okydmbo309d): best step 4 reward `0.5693`, checkpoint `ginimgkdx6okinz84klf0m98`, step 8 `0.4865`, cost `$7.43` |
| v0.2.47 checkpoint probe | [`cto44tv6sqbkynjp2g01ggpw`](https://app.primeintellect.ai/dashboard/training/cto44tv6sqbkynjp2g01ggpw): heldout reward `0.5312`, `+6.9pp` over its scaffolded v0.2.47 base |
| v0.2.50 clean geometry baseline | [`dbup76z9d460x4fbcfxw8yql`](https://app.primeintellect.ai/dashboard/training/dbup76z9d460x4fbcfxw8yql): reward `0.3031`, solved `0.0508`, native tool calls `1.0`, zero errors |
| v0.2.50 clean geometry train | [`dbs7pcyih846945xubanvdjr`](https://app.primeintellect.ai/dashboard/training/dbs7pcyih846945xubanvdjr): stopped; best online reward `0.3347`, step 6 `0.3096`, zero errors, cost `$6.16` |
| v0.2.50 ckpt4 heldout probe | [`qbrp8k55hndkrbuh4ndn4pyu`](https://app.primeintellect.ai/dashboard/training/qbrp8k55hndkrbuh4ndn4pyu): checkpoint `wdlu817nloqo6tdheaaou0c8` scored `0.2663` vs clean base `0.3031`; zero errors, cost `$0.35` |
| v0.2.51 diagnostic baseline | [`g8egyymgds47wtb4rbieyd20`](https://app.primeintellect.ai/dashboard/training/g8egyymgds47wtb4rbieyd20): reward `0.3241`, solved `0.0622`, `target_face_in_candidate_set=1.0`, native tool calls `1.0`, zero errors |
| v0.2.51 conservative train | [`byzwnn49pt9ztm6xiw0jumkx`](https://app.primeintellect.ai/dashboard/training/byzwnn49pt9ztm6xiw0jumkx): stopped at step 4; all rows below baseline, old reward had a fixed-policy shortcut |
| v0.2.52 relative-flow baseline | [`iwozt5azyroqtwkfztd47e6s`](https://app.primeintellect.ai/dashboard/training/iwozt5azyroqtwkfztd47e6s): reward `0.5676`, solved `0.3048`, face `0.7677`, native tool calls `1.0`, zero errors |
| v0.2.52 relative-flow train | [`hg24yiykjdrvounjsg6bd6si`](https://app.primeintellect.ai/dashboard/training/hg24yiykjdrvounjsg6bd6si): stopped at step 3; rows `0.5557`, `0.5231`, `0.5459`, `0.5325` stayed below baseline; checkpoint `lle93v210pcgw6kawfr2mr3e` still uploading; cost `$4.06` |
| v0.2.52 gpt-oss-120b probe | [`ax0cjhthz5w44unvvr44j712`](https://app.primeintellect.ai/dashboard/training/ax0cjhthz5w44unvvr44j712): failed zero-advantage baseline probe, no env errors, cost `$0.20` |
| v0.2.52 lower-pressure train | [`hj5elbyhggckw9t975i9pdz2`](https://app.primeintellect.ai/dashboard/training/hj5elbyhggckw9t975i9pdz2): stopped after step 0 reward `0.4905`; cost `$1.00` |
| v0.2.53 rule-flow baseline | [`vxrolc00tg1h1yc8f96udnu9`](https://app.primeintellect.ai/dashboard/training/vxrolc00tg1h1yc8f96udnu9): reward `0.6726`, solved `0.4033`, face `0.7644`, frontier `0.3178`, native tool calls `1.0`, zero errors, cost `$0.18` |
| v0.2.53 inline-eval train | [`xpmbx7rp9ht9trhhwjq8q3zx`](https://app.primeintellect.ai/dashboard/training/xpmbx7rp9ht9trhhwjq8q3zx): stopped before updates because inline eval hung with zero token usage |
| v0.2.53 rule-flow train | [`vychxoaksf66c7pto4wz7rez`](https://app.primeintellect.ai/dashboard/training/vychxoaksf66c7pto4wz7rez): stopped after step 2; step 1 reward `0.7133` beat baseline, step 2 fell to `0.6299`; zero errors; no checkpoint exposed; cost `$3.38` |
| v0.2.53 mixed teacher train | [`vtg65yvig5fstip74awggpgn`](https://app.primeintellect.ai/dashboard/training/vtg65yvig5fstip74awggpgn): stopped after step 1; clean lane `0.6820` at step 0 then `0.6338` at step 1; no checkpoint exposed; cost `$3.03` |
| v0.2.53 completed short train | [`f6ajk2cyut5om7qfb1qisz5e`](https://app.primeintellect.ai/dashboard/training/f6ajk2cyut5om7qfb1qisz5e): completed 2 steps; step 1 reward `0.7055`, solved `0.4083`, zero errors; no checkpoint exposed; cost `$2.23` |
| v0.2.53 Qwen 3.6 35B depth-2 run | [`evntfyfh9m7xvkofwin76arx`](https://app.primeintellect.ai/dashboard/training/evntfyfh9m7xvkofwin76arx): stopped after model-start stall before token usage; cost `$0.00` |
| v0.2.53 gpt-oss-120b depth-2 run | [`owhgkpefm1wm5ibjllbeh85t`](https://app.primeintellect.ai/dashboard/training/owhgkpefm1wm5ibjllbeh85t): stopped after model-start stall before token usage; cost `$0.00` |
| v0.2.53 Qwen 9B depth-2 run | [`ycx5aoe58lko3gvqyyto7794`](https://app.primeintellect.ai/dashboard/training/ycx5aoe58lko3gvqyyto7794): completed 4 steps; rewards `0.5483`, `0.4986`, `0.5019`, `0.5412`; frontier `0.6973` -> `0.6953`, zero errors; checkpoint `pxum4a0nnvfuq5jssm28qgwa` still uploading; cost `$2.16` |
| v0.2.54 solve-2 train | [`bg0vbir6u6d521qcr8kghvvv`](https://app.primeintellect.ai/dashboard/training/bg0vbir6u6d521qcr8kghvvv): completed Qwen 9B two-call candidate-path run; final online reward `0.7335`, solved `0.6615`, two native tool calls, zero errors, cost `$7.70`; checkpoint `ce4skj0ockwhx4zq7ztutsap` step 3 READY |
| v0.2.54 solve-2 heldout base | [`xx4pql9agtfl2se7046brmio`](https://app.primeintellect.ai/dashboard/training/xx4pql9agtfl2se7046brmio): reward `0.6335`, solved `0.5625`, two native tool calls, zero errors, cost `$1.33` |
| v0.2.54 solve-2 ckpt3 probe | [`k2j8tjj7gra5ukmzmrff9epu`](https://app.primeintellect.ai/dashboard/training/k2j8tjj7gra5ukmzmrff9epu): checkpoint `ce4skj0ockwhx4zq7ztutsap` heldout reward `0.6282`, solved `0.5746`, two native tool calls, zero errors, cost `$1.33` |
| v0.2.54 solve-2 ckpt2 heldout win | [`j4y1xf2i6cg3lw40m8e3yom9`](https://app.primeintellect.ai/dashboard/training/j4y1xf2i6cg3lw40m8e3yom9): tail-room checkpoint `lbujflb1zyzv764lh9dhzu3s` heldout reward `0.6631` (`+2.96pp`), solved `0.6048` (`+4.23pp`), two native tool calls, zero errors, cost `$1.32` |
| v0.2.54 solve-2 second heldout | base [`lj8g10rkj4haatuhq1iqxqzn`](https://app.primeintellect.ai/dashboard/training/lj8g10rkj4haatuhq1iqxqzn) reward `0.6707`, solved `0.5723`; checkpoint [`jiwkc1h58uwyeiyc8z7pnka9`](https://app.primeintellect.ai/dashboard/training/jiwkc1h58uwyeiyc8z7pnka9) reward `0.6712`, solved `0.5801`, zero errors |
| v0.2.55 tail-solve base checks | [`q98zu1mmnl4z80uxo10jnt24`](https://app.primeintellect.ai/dashboard/training/q98zu1mmnl4z80uxo10jnt24) seed 146 reward `0.5833`, solved `0.5373`; [`rotjrhjd3g2189sf58qd2e5b`](https://app.primeintellect.ai/dashboard/training/rotjrhjd3g2189sf58qd2e5b) seed 246 reward `0.6309`, solved `0.5684`; both zero errors |
| v0.2.55 continuation | [`gnpet9lrxx16amnytkcb8vju`](https://app.primeintellect.ai/dashboard/training/gnpet9lrxx16amnytkcb8vju): warm-started from `lbujflb1zyzv764lh9dhzu3s`; step 2 reward `0.6259`, solved `0.5674`, checkpoint `o68kzy5up4e65ve6lktmkuat` READY; step 3 regressed; cost `$5.50` |
| v0.2.55 checkpoint probes | Configs exist for heldout seeds 146 and 246, but Prime rejected new run creation with `Payment required`; no canonical heldout probe result was produced |
| v0.2.56 shortcut fix | Hides prompt example ids, removes the visible-index second-slot shortcut, caps non-solving second-step tail reward below `0.50`, and adds an oracle JSONL exporter for future SFT/warm-start work |
| Visibility | CLI/API still report `PRIVATE` after `--visibility PUBLIC`; owner-auth install/eval/train work |

## What We Cracked

The important result is two-part and should be read honestly:

1. **Environment/prompt direction breakthrough.** v0.2.21 showed that the model
   could identify the depth-1 face but stayed at chance on `cw` versus `ccw`:
   [`Qwen/Qwen3.5-35B-A3B` solved `0.504`](https://app.primeintellect.ai/dashboard/evaluations/dgu4mymqqk5sy97l3kvusjxu).
   v0.2.22 reframed the same evidence as candidate solving actions and reached
   [`0.929` solved](https://app.primeintellect.ai/dashboard/evaluations/vse6uoo8c9y156svyyv41qll)
   on the identical 240-example heldout set.

2. **Native RL measurement lane.** The credible native-tool lane is Prime Hosted
   Training's Token-In Token-Out path. The v0.2.28 native heldout probes show a
   modest positive RL movement, not the original `+30pp` target:

| Probe | Model/checkpoint | Reward/solved | Native tools | Errors |
| --- | --- | ---: | ---: | ---: |
| [`gwfcyfztn8ahpwceu9n63rzd`](https://app.primeintellect.ai/dashboard/training/gwfcyfztn8ahpwceu9n63rzd) | base `Qwen/Qwen3.5-9B` | `0.7819` | `1.0` | `0` |
| [`eg68wcx3ym12mhvtku4l7g5g`](https://app.primeintellect.ai/dashboard/training/eg68wcx3ym12mhvtku4l7g5g) | step 30 | `0.7700` | `1.0` | `0` |
| [`b6qh3e36i2poqghq5eum97dd`](https://app.primeintellect.ai/dashboard/training/b6qh3e36i2poqghq5eum97dd) | step 40 | `0.8071` | `1.0` | `0` |

Served chat-completions evals are still useful for compatibility history, but
they do not measure the same native tool-call surface for these Qwen adapters:
the served endpoint often emits JSON/private text rather than native
`tool_calls`. See [the report](reports/megaminx-rl-report.md) for the full
native-vs-served accounting.

3. **Leak audit and clean lane.** v0.2.47 proved the hosted native-tool loop can
   move a checkpoint, but its printed `cw_mask`/`ccw_mask` scorecard and
   candidate construction were too scaffolded. v0.2.50 removes those public
   reward proxies and builds candidate slots from visible affected geometry.

## Quickstart

```bash
prime env install setrf/megaminx-solver@0.2.56 --plain
uv run pytest -q
```

Export oracle trajectories for a future SFT/warm-start lane:

```bash
uv run python scripts/export_oracle_trajectories.py \
  --num-examples 1024 \
  --seed 64 \
  --split train_candidate_relative_flow_rule_tail_solve_depth2 \
  --output /tmp/megaminx-oracle-v056-1024.jsonl
uv run python scripts/summarize_oracle_trajectories.py /tmp/megaminx-oracle-v056-1024.jsonl
uv run python scripts/convert_oracle_to_sft_jsonl.py \
  /tmp/megaminx-oracle-v056-1024.jsonl \
  --output /tmp/megaminx-oracle-v056-1024-sft.jsonl
```

The current 1,024-row v0.2.56 oracle audit solves every row with exactly two
native `select_candidate` actions, balanced slots/directions, and no visible
row-id prompt leakage. Re-running the export is byte-identical (`cmp_exit=0`);
the current SHA256 is
`0604bd14343aebb04f9b68ba77cb1dd34d1062f025d011d3961298393b3258e7`.
The derived messages/tools SFT JSONL has SHA256
`99575888ced056df08a8950bf20c8fe31fbbfe0b2bbf1fcf30c12401165f44ed`.
Hosted follow-up runs are blocked until Prime billing is restored;
`prime wallet --plain` reported a `$-0.80` balance on May 14, 2026.
The exact auth/billing recovery and next hosted probe sequence is in
[`reports/megaminx-next-run-runbook.md`](reports/megaminx-next-run-runbook.md).

Reproduce the tracked tail-solve baseline/training lane:

```bash
prime train configs/rl/megaminx-v054-qwen9b-rule-flow-solve2-depth2-rpe16-complete6.toml --yes --plain
prime train configs/rl/megaminx-v055-qwen9b-tail-solve-depth2-base-heldout-rpe16.toml --yes --plain
prime train configs/rl/megaminx-v055-qwen9b-tail-solve-depth2-base-heldout2-rpe16.toml --yes --plain
prime train configs/rl/megaminx-v055-qwen9b-tail-solve-depth2-lbuj-continue-b1024-lr1e9-rpe16.toml --yes --plain
prime train configs/rl/megaminx-v056-qwen9b-tail-solve-depth2-base-heldout-rpe16.toml --yes --plain
prime train configs/rl/megaminx-v056-qwen9b-tail-solve-depth2-base-heldout2-rpe16.toml --yes --plain
prime train configs/rl/megaminx-v056-qwen9b-tail-solve-depth2-ckpt2-heldout-rpe16.toml --yes --plain
prime train configs/rl/megaminx-v056-qwen9b-tail-solve-depth2-ckpt2-heldout2-rpe16.toml --yes --plain
```

The report also records older v0.2.49-v0.2.53 local run history. Some of those
historical TOMLs are intentionally left as local artifacts rather than tracked
release reproduction files.

## Environment Shape

- Entry point: `load_environment(...) -> vf.Environment`
- Implementation: `MegaminxEnv(vf.StatefulToolEnv)`
- Tools: `rotate(face: A-L, direction: cw|ccw)`, `inspect(face: A-L|all)`,
  `finish()`
- Simulator: 12 faces, 30 edge pieces, 20 corner pieces, 132 stickers, 24 legal
  face turns, deterministic scrambles and inverse metadata hidden from prompts
- Curriculum splits: `depth1`, `easy`, `medium`, `hard`, `eval`, plus `train_*`
  and `eval_*` variants

The v0.2.46 package adds the least-scaffolded scorecard so far:
`stage_candidate_scorecard_mask_native_tool` with
`select_candidate(index,direction)`. It exposes four candidate faces with
affected-neighbor counts and direction masks, but removes scalar `support`,
`frontier`, inverse solution, and answer fields from the prompt.

The v0.2.47 candidate variant adds
`stage_candidate_scorecard_mask_frontier_equivalence_native_tool` with
`action_gated_candidate_mask_frontier_equivalence`. It keeps the same public
scorecard columns as v0.2.46, while rewarding clean equivalent first moves that
leave a depth-2 scramble one turn from solved.

## Protocol Lanes

Prime's renderer/TITO direction matters here because a one-action puzzle task is
very sensitive to whether the sampled assistant/tool-call tokens are preserved
exactly. The project follows this split:

- **Native training/probe lane:** `stage_candidate_scorecard_mask_native_tool`,
  `stage_candidate_geometry_frontier_native_tool`,
  `allow_text_tool_actions=false`, Hosted Training token-native tool calls,
  metrics show `native_tool_call_count=1.0`.
- **Served eval compatibility lane:** `stage_solve_direction_flow_json_action`,
  `allow_text_tool_actions=true`, chat-completions JSON/private text fallback,
  metrics show `private_text_action_count` or `text_tool_action_count`.

The report cites the Prime Environments, Verifiers, and Renderers docs used to
set this boundary.

## Files

- Environment package: `environments/megaminx_solver/`
- Main implementation: `environments/megaminx_solver/megaminx_solver/megaminx_solver.py`
- Tests: `tests/test_megaminx_solver.py`
- Hosted RL configs: `configs/rl/`
- Eval configs: `configs/eval/`
- Final report: `reports/megaminx-rl-report.md`
- Next run runbook: `reports/megaminx-next-run-runbook.md`
