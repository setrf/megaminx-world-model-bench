# RL Missing Test / Metric Note

## Most Important Gap

The two-turn `stage_candidate_relative_flow_rule_solve2_native_tool` lane needs a second-step candidate-slot balance test and a second-step target-slot metric.

The current environment refreshes the candidate table after the first `select_candidate` call in `MegaminxEnv.select_candidate`:

- `environments/megaminx_solver/megaminx_solver/megaminx_solver.py:2295` enters the multistep candidate path branch.
- `environments/megaminx_solver/megaminx_solver/megaminx_solver.py:2298` replaces `rollout.candidate_faces` with `_candidate_geometry_faces(rollout.puzzle, rollout.move_count)`.

The current test at `tests/test_megaminx_solver.py:3323` checks that an oracle first move refreshes the table and that the oracle second face is present, then solves the puzzle. It does not check the refreshed target slot distribution, and the registered metrics at `environments/megaminx_solver/megaminx_solver/megaminx_solver.py:2698` include first-step candidate metrics plus `second_rotate_*`, but no `second_candidate_index`, `second_target_candidate_index`, or second-step relative-flow/margin metric.

## Why This Could Explain Unstable RL Gains

I ran a read-only probe over the actual solve-2 shape used by the report:

- `num_examples=2048`, seeds `46`, `146`, and `246`
- `reward_style="action_gated_candidate_path_solve"`
- `prompt_style="stage_candidate_relative_flow_rule_solve2_native_tool"`
- oracle first action, then inspect refreshed candidates

Result:

- First target slots were balanced across slots 1-4.
- After the oracle first action, the second target face was in refreshed slot 4 for every sampled example: `2048/2048` on each checked seed.
- Second target direction remained roughly balanced, so the shortcut is specifically the second candidate slot.

That means online reward can improve partly by learning a hidden structural regularity: after a good first call, choose refreshed slot 4 and only solve the direction problem. The existing aggregate metrics would show clean protocol, two native tool calls, solved rate, first exact, second exact, and frontier, but they would not reveal whether the policy is exploiting a degenerate refreshed-slot distribution. This can make gains look unstable: changes in first-step success, direction accuracy, or seed composition can swing reward while the underlying second-step slot policy remains a shortcut.

## Suggested Test

Add a solve-2 analogue to the existing first-step balance check at `tests/test_megaminx_solver.py:3134`.

Concrete shape:

- Build `stage_candidate_relative_flow_rule_solve2_native_tool` with train-sized or at least multi-seed depth-2 samples.
- For each row, execute the oracle first `select_candidate`.
- Record `rollout.candidate_faces.index(second_inverse_face) + 1`.
- Assert the refreshed second target slot distribution covers all slots 1-4 and is not dominated by one slot.
- Also assert the refreshed prompt/table still gives enough visible relative-flow signal for the second direction without relying on hidden metadata.

## Suggested Metric

Register zero-weight metrics for the two-turn lane:

- `second_target_candidate_index`
- `second_candidate_face_correct`
- `second_candidate_relative_flow_count`
- `second_candidate_relative_flow_margin`
- optionally `second_candidate_relative_flow_is_candidate_max`

These would make hosted training/probe dashboards show whether a checkpoint is improving real second-step selection or just riding the fixed refreshed slot.
