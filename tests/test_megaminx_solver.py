from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
import json
from pathlib import Path
import tomllib

from megaminx_solver import load_environment
from megaminx_solver.megaminx_solver import (
    action_gated_curriculum_reward,
    action_gated_dense_reward,
    action_gated_direction_reward,
    action_gated_exact_direction_reward,
    action_gated_binary_direction_reward,
    action_gated_candidate_index_reward,
    action_gated_candidate_geometry_frontier_reward,
    action_gated_candidate_path_solve_reward,
    action_gated_candidate_path_tail_solve_reward,
    action_gated_candidate_strict_frontier_reward,
    action_gated_candidate_mask_frontier_equivalence_reward,
    action_gated_candidate_mask_index_rank_reward,
    action_gated_candidate_tournament_reward,
    action_gated_face_discovery_reward,
    action_gated_face_tournament_reward,
    action_gated_predict_rotate_value_strict_reward,
    action_gated_predict_rotate_transition_reward,
    action_gated_counterfactual_frontier_value_strict_reward,
    action_gated_counterfactual_frontier_strict_reward,
    action_gated_mask_overlap_strict_shaped_direction_reward,
    action_gated_overlap_strict_shaped_direction_reward,
    action_gated_strict_shaped_direction_reward,
    action_gated_overlap_reward,
    action_reaches_one_turn_frontier,
    _action_mask_counts,
    action_taken,
    first_rotate_mask_count,
    first_rotate_mask_is_max,
    first_rotate_correct,
    first_rotate_direction_correct,
    first_rotate_direction_id,
    second_rotate_correct,
    second_rotate_face_correct,
    second_rotate_direction_correct,
    second_candidate_index,
    second_target_candidate_index,
    second_candidate_face_correct,
    second_candidate_relative_flow_count,
    second_candidate_relative_flow_margin,
    second_candidate_relative_flow_is_candidate_max,
    candidate_path_completed,
    inverse_prefix_length,
    first_candidate_face_correct,
    first_candidate_public_mask_count,
    first_candidate_public_mask_is_max,
    first_candidate_public_mask_rank_credit,
    first_candidate_relative_flow_count,
    first_candidate_relative_flow_margin,
    first_candidate_relative_flow_is_candidate_max,
    first_rotate_face_correct,
    first_rotate_face_id,
    first_candidate_index,
    target_candidate_index,
    candidate_relative_flow_oracle_unique,
    first_rotate_in_candidate_set,
    target_face_in_candidate_set,
    first_rotate_neighbor_overlap,
    first_rotate_counterfactual_frontier,
    frontier_equivalent,
    first_rotate_counterfactual_value,
    first_rotate_counterfactual_value_norm,
    first_rotate_counterfactual_value_rank,
    first_prediction_strip_accuracy,
    first_prediction_exact_strip_count,
    first_prediction_char_accuracy,
    first_prediction_quality,
    first_prediction_valid,
    first_prediction_item_count,
    first_prediction_extra_count,
    first_rotate_face_frontier_viable,
    first_rotate_frontier_tail_best_mask_count,
    first_rotate_frontier_tail_count,
    first_rotate_frontier_tail_unique,
    initial_max_action_mask_count,
    native_tool_call_count,
    private_text_action_count,
    protocol_violation_count,
    reward_style,
    candidate_select_call_count,
    predict_rotate_call_count,
    rotate_call_count,
    solved_reward,
    text_tool_action_count,
    tool_call_error_count,
    tool_parse_error_count,
)
from megaminx_solver.simulator import (
    CORNER_COUNT,
    DIRECTIONS,
    EDGE_COUNT,
    FACES,
    POSITIONS_PER_FACE,
    STICKERS_PER_PUZZLE,
    DEFAULT_TOPOLOGY,
    MegaminxPuzzle,
    generate_scramble,
    inverse_moves,
)
from verifiers.types import AssistantMessage


def sticker_counter(puzzle: MegaminxPuzzle) -> Counter[str]:
    return Counter(puzzle.stickers.values())


def relative_flow_rule_action_from_prompt(prompt: str) -> tuple[int, str]:
    table = prompt.split("slot | face | ring | relative_flow\n", 1)[1].split(
        "Initial compact observation:", 1
    )[0]
    best: tuple[tuple[int, int, int], int, str] | None = None
    for line in table.strip().splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 4:
            continue
        slot = int(parts[0])
        flow = parts[3]
        for direction, count in (
            ("cw", flow.count(":+1(")),
            ("ccw", flow.count(":-1(")),
        ):
            key = (count, -slot, int(direction == "ccw"))
            if best is None or key > best[0]:
                best = (key, slot, direction)
    assert best is not None
    return best[1], best[2]


def test_topology_counts() -> None:
    assert len(FACES) == 12
    assert POSITIONS_PER_FACE == 11
    assert STICKERS_PER_PUZZLE == 132
    assert len(DEFAULT_TOPOLOGY.edge_pieces) == EDGE_COUNT == 30
    assert len(DEFAULT_TOPOLOGY.corner_pieces) == CORNER_COUNT == 20
    for ring in DEFAULT_TOPOLOGY.neighbor_rings.values():
        assert len(ring) == 5


def test_moves_preserve_sticker_multiset_and_have_order_five() -> None:
    solved = MegaminxPuzzle.solved(DEFAULT_TOPOLOGY)
    expected_counter = sticker_counter(solved)
    for face in FACES:
        puzzle = solved.copy()
        puzzle.apply_move(face, "cw")
        assert sticker_counter(puzzle) == expected_counter
        puzzle.apply_move(face, "ccw")
        assert puzzle.stickers == solved.stickers

        puzzle = solved.copy()
        for _ in range(5):
            puzzle.apply_move(face, "cw")
        assert puzzle.stickers == solved.stickers


def test_scramble_inverse_solves() -> None:
    scramble = generate_scramble(8, __import__("random").Random(7))
    puzzle = MegaminxPuzzle.solved(DEFAULT_TOPOLOGY)
    puzzle.apply_moves(scramble)
    assert not puzzle.is_solved()
    puzzle.apply_moves(inverse_moves(scramble))
    assert puzzle.is_solved()


def test_cw_ccw_strip_flow_matches_prompt_definition_for_all_faces() -> None:
    for face in FACES:
        ring = DEFAULT_TOPOLOGY.neighbor_rings[face]
        for direction in DIRECTIONS:
            puzzle = MegaminxPuzzle.solved(DEFAULT_TOPOLOGY)
            puzzle.apply_move(face, direction)
            for side, destination in enumerate(ring):
                expected_source = (
                    ring[(side - 1) % len(ring)]
                    if direction == "cw"
                    else ring[(side + 1) % len(ring)]
                )
                strip_colors = [
                    puzzle.stickers[position]
                    for position in DEFAULT_TOPOLOGY.side_strip(face, destination)
                ]
                assert strip_colors == [expected_source] * len(strip_colors)

            puzzle.apply_moves(inverse_moves([(face, direction)]))
            assert puzzle.is_solved()


def test_load_environment_and_scripted_inverse_solution() -> None:
    async def run() -> None:
        env = load_environment(num_examples=1, min_depth=3, max_depth=3, seed=11)
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        assert not rollout.solved()
        for face, direction in rollout.inverse_solution:
            await env.rotate(face, direction, rollout)
        assert rollout.solved()
        assert await solved_reward(state) == 1.0

    asyncio.run(run())


def test_invalid_tool_args_are_non_crashing_and_penalized() -> None:
    async def run() -> None:
        env = load_environment(num_examples=1, min_depth=2, max_depth=2, seed=12)
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        before = rollout.puzzle.stickers.copy()
        response = await env.rotate("Z", "cw", rollout)
        assert "Illegal move" in response
        assert rollout.illegal_moves == 1
        assert rollout.puzzle.stickers == before

    asyncio.run(run())


def test_non_string_tool_args_are_penalized_without_crashing() -> None:
    async def run() -> None:
        env = load_environment(num_examples=1, min_depth=2, max_depth=2, seed=24)
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]

        rotate_response = await env.rotate(None, "cw", rollout)  # type: ignore[arg-type]
        assert "Illegal move" in rotate_response
        assert rollout.illegal_moves == 1

        inspect_response = await env.inspect(None, rollout)  # type: ignore[arg-type]
        assert "Illegal inspect target" in inspect_response
        assert rollout.illegal_moves == 2

    asyncio.run(run())


def test_depth1_split_and_prompt_contract() -> None:
    env = load_environment(split="depth1", num_examples=5, seed=13)
    rows = [dict(row) for row in env.get_dataset()]
    assert {row["scramble_depth"] for row in rows} == {1}
    assert {row["move_budget"] for row in rows} == {6}
    assert env.max_turns == 8
    first_prompt = "\n".join(message["content"] for message in rows[0]["prompt"])
    assert "one-turn scramble" in first_prompt
    assert rows[0]["answer"] not in first_prompt


def test_move_budget_can_differ_from_env_max_turns() -> None:
    async def run() -> None:
        env = load_environment(split="depth1", num_examples=1, max_turns=2, move_budget=1, seed=34)
        row = dict(env.get_dataset()[0])
        assert row["move_budget"] == 1
        assert env.max_turns == 2

        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        first_wrong = next(
            move
            for move in ((face, direction) for face in FACES for direction in DIRECTIONS)
            if move != rollout.inverse_solution[0]
        )
        await env.rotate(*first_wrong, rollout)
        assert rollout.move_count == 1
        assert rollout.exhausted()
        assert rollout.finished

        response = await env.rotate(*rollout.inverse_solution[0], rollout)
        assert "already finished" in response
        assert rollout.move_count == 1

    asyncio.run(run())


def test_depth1_scrambles_are_balanced_across_legal_moves() -> None:
    env = load_environment(split="depth1", num_examples=len(FACES) * len(DIRECTIONS), seed=31)
    rows = [dict(row) for row in env.get_dataset()]
    scrambles = [tuple(json.loads(row["scramble"])[0].values()) for row in rows]
    assert len(set(scrambles)) == len(FACES) * len(DIRECTIONS)
    assert {face for face, _ in scrambles} == set(FACES)
    assert {direction for _, direction in scrambles} == set(DIRECTIONS)


def test_named_split_depth_ranges_and_metadata() -> None:
    for split, expected in {
        "depth1": {1},
        "easy": {1, 2, 3},
        "medium": {4, 5, 6},
        "hard": {7, 8, 9, 10},
    }.items():
        env = load_environment(split=split, num_examples=len(expected) * 2, seed=14)
        rows = [dict(row) for row in env.get_dataset()]
        assert {row["scramble_depth"] for row in rows} == expected
        for row in rows:
            prompt = "\n".join(message["content"] for message in row["prompt"])
            assert row["answer"] not in prompt
            assert row["reward_style"] == "dense"
            assert row["prompt_style"] == "default"


def test_action_gated_reward_requires_rotate_and_tracks_first_move() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=15,
            reward_style="action_gated_dense",
            prompt_style="action_first",
        )
        row = dict(env.get_dataset()[0])
        prompt = "\n".join(message["content"] for message in row["prompt"])
        assert "must be a rotate tool call" in prompt
        assert "Output no explanation before the first rotate tool call" in prompt

        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        assert await reward_style(state) == 1.0
        assert await action_gated_dense_reward(state) == 0.0
        assert await action_taken(state) == 0.0
        assert rollout.initial_sticker_accuracy == rollout.puzzle.sticker_accuracy()
        assert rollout.initial_piece_accuracy == rollout.puzzle.piece_accuracy()

        correct = rollout.inverse_solution[0]
        wrong = next(
            (face, direction)
            for face in FACES
            for direction in DIRECTIONS
            if (face, direction) != correct
        )
        await env.rotate(*wrong, rollout)
        assert await action_taken(state) == 1.0
        assert await first_rotate_correct(state) == 0.0
        assert await first_rotate_face_correct(state) in {0.0, 1.0}
        assert await first_rotate_direction_correct(state) in {0.0, 1.0}
        assert 0.0 <= await action_gated_dense_reward(state) <= 0.4

    asyncio.run(run())


def test_direct_json_action_prompt_style() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=19,
        reward_style="action_gated_curriculum",
        prompt_style="direct_json_action",
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    assert "/no_think" in prompt
    assert "Return exactly one JSON object" in prompt
    assert '{"tool":"rotate","args":{"face":"A","direction":"cw"}}' in prompt
    assert row["answer"] not in prompt


def test_sensor_native_tool_prompt_style_uses_native_tools_without_json_contract() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=29,
        reward_style="action_gated_curriculum",
        prompt_style="sensor_native_tool",
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    assert "Native tool mode: call the rotate tool directly" in prompt
    assert "Structured state sensors:" in prompt
    assert "Static topology rules:" in prompt
    assert "Return exactly one JSON object" not in prompt
    assert "Choose exactly one legal action from this menu" not in prompt
    assert row["answer"] not in prompt


def test_stage_direction_flow_native_tool_prompt_uses_reasoned_native_tool_contract() -> None:
    for prompt_style, table_header in (
        ("stage_direction_flow_native_tool", "Direction flow table for face"),
        ("stage_solve_direction_flow_native_tool", "Solve-direction flow table for face"),
    ):
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=41,
            reward_style="action_gated_binary_direction",
            prompt_style=prompt_style,
            move_budget=1,
        )
        row = dict(env.get_dataset()[0])
        prompt = "\n".join(message["content"] for message in row["prompt"])

        assert env.env_args["allow_text_tool_actions"] is False
        assert "Native tool mode: compare the direction-flow table, then call rotate directly." in prompt
        assert "You may reason internally" in prompt
        assert table_header in prompt
        assert "/no_think" not in prompt
        assert "Return exactly one JSON object" not in prompt
        assert "Choose exactly one legal action from this menu" not in prompt
        assert row["answer"] not in prompt
        assert row["scramble"] not in prompt
        assert row["inverse_solution"] not in prompt


def test_native_tool_prompt_styles_do_not_use_json_contract_or_leak_metadata() -> None:
    cases = {
        "native_action": {
            "present": ["Native tool mode: call the rotate tool directly"],
            "absent": [
                "Structured state sensors:",
                "Static topology rules:",
                "Candidate neighbor sets for face matching:",
            ],
        },
        "topology_native_tool": {
            "present": [
                "Native tool mode: call the rotate tool directly",
                "Static topology rules:",
            ],
            "absent": [
                "Structured state sensors:",
                "Candidate neighbor sets for face matching:",
            ],
        },
        "sensor_native_tool": {
            "present": [
                "Native tool mode: call the rotate tool directly",
                "Structured state sensors:",
                "Static topology rules:",
            ],
            "absent": ["Candidate neighbor sets for face matching:"],
        },
        "sensor_match_native_tool": {
            "present": [
                "Native tool mode: call the rotate tool directly",
                "Structured state sensors:",
                "Static topology rules:",
                "Candidate neighbor sets for face matching:",
            ],
            "absent": [],
        },
    }
    for prompt_style, expectations in cases.items():
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=32,
            reward_style="action_gated_curriculum",
            prompt_style=prompt_style,
        )
        row = dict(env.get_dataset()[0])
        prompt = "\n".join(message["content"] for message in row["prompt"])
        for expected in expectations["present"]:
            assert expected in prompt
        for forbidden in [
            *expectations["absent"],
            "Return exactly one JSON object",
            "Choose exactly one legal action from this menu",
            '"tool": "rotate"',
            "Highest-overlap face candidates:",
            "Neighbor-overlap counts:",
            row["answer"],
            row["scramble"],
            row["inverse_solution"],
        ]:
            assert forbidden not in prompt


def test_allow_text_tool_actions_defaults_by_prompt_style() -> None:
    for prompt_style in (
        "direct_json_action",
        "choice_json_action",
        "topology_choice_json_action",
        "sensor_choice_json_action",
        "sensor_match_json_action",
        "sensor_indexed_match_json_action",
        "sensor_candidate_strips_json_action",
    ):
        env = load_environment(prompt_style=prompt_style)
        assert env.env_args["allow_text_tool_actions"] is True
    stage_env = load_environment(split="depth1", prompt_style="stage_face_hint_direction_json_action")
    assert stage_env.env_args["allow_text_tool_actions"] is True
    flow_env = load_environment(split="depth1", prompt_style="stage_direction_flow_json_action")
    assert flow_env.env_args["allow_text_tool_actions"] is True
    reasoned_flow_env = load_environment(
        split="depth1",
        prompt_style="stage_direction_flow_reasoned_json_action",
    )
    assert reasoned_flow_env.env_args["allow_text_tool_actions"] is True
    solve_flow_env = load_environment(
        split="depth1",
        prompt_style="stage_solve_direction_flow_json_action",
    )
    assert solve_flow_env.env_args["allow_text_tool_actions"] is True

    for prompt_style in (
        "default",
        "action_first",
        "native_action",
        "topology_native_tool",
        "sensor_native_tool",
        "sensor_match_native_tool",
        "stage_direction_flow_native_tool",
        "stage_solve_direction_flow_native_tool",
        "stage_solve_direction_flow_native_tool_v2",
        "stage_solve_action_table_native_tool",
        "stage_solve_action_mask_native_tool",
    ):
        kwargs = (
            {"split": "depth1"}
            if prompt_style
            in {
                "stage_direction_flow_native_tool",
                "stage_solve_direction_flow_native_tool",
                "stage_solve_direction_flow_native_tool_v2",
                "stage_solve_action_table_native_tool",
                "stage_solve_action_mask_native_tool",
            }
            else {}
        )
        env = load_environment(prompt_style=prompt_style, **kwargs)
        assert env.env_args["allow_text_tool_actions"] is False


def test_sensor_match_json_action_prompt_lists_static_candidate_sets_without_answer() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=30,
        reward_style="action_gated_curriculum",
        prompt_style="sensor_match_json_action",
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    assert "Structured state sensors:" in prompt
    assert "Candidate neighbor sets for face matching:" in prompt
    assert prompt.count('"tool": "rotate"') == len(FACES) * len(DIRECTIONS)
    for face in FACES:
        neighbors = " ".join(sorted(DEFAULT_TOPOLOGY.neighbor_rings[face]))
        assert f"{face}: {neighbors}" in prompt
    assert "Highest-overlap face candidates:" not in prompt
    assert "Neighbor-overlap counts:" not in prompt
    assert row["answer"] not in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt


def test_sensor_indexed_match_json_action_prompt_adds_index_guide_without_answer() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=35,
        reward_style="action_gated_direction",
        prompt_style="sensor_indexed_match_json_action",
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    puzzle = MegaminxPuzzle.solved(DEFAULT_TOPOLOGY)
    puzzle.apply_moves([(item["face"], item["direction"]) for item in json.loads(row["scramble"])])
    affected_faces = tuple(
        face
        for face in FACES
        if any(puzzle.stickers[(face, index)] != face for index in range(1, POSITIONS_PER_FACE))
    )

    assert "/no_think" in prompt
    assert "Return exactly one JSON object" in prompt
    assert "Choose exactly one legal action from this menu" in prompt
    assert "Structured state sensors:" in prompt
    assert "Static topology rules:" in prompt
    assert "Candidate neighbor sets for face matching:" in prompt
    assert "Direction index guide:" in prompt
    assert "corners c0..c4 and edges e0..e4" in prompt
    assert "side = ring_N.index(X)" in prompt
    assert "c[(side-1)%5], e[side], c[side]" in prompt
    assert "Indexed affected stickers:" in prompt
    assert prompt.count('"tool": "rotate"') == len(FACES) * len(DIRECTIONS)
    for face in affected_faces:
        assert f"{face}: ring={' '.join(DEFAULT_TOPOLOGY.neighbor_rings[face])}" in prompt
        assert f"{face} changed:" in prompt
    assert "Highest-overlap face candidates:" not in prompt
    assert "Neighbor-overlap counts:" not in prompt
    assert row["answer"] not in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt


def test_sensor_candidate_strips_json_action_prompt_lists_candidate_strips_without_answer() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=37,
        reward_style="action_gated_exact_direction",
        prompt_style="sensor_candidate_strips_json_action",
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    puzzle = MegaminxPuzzle.solved(DEFAULT_TOPOLOGY)
    puzzle.apply_moves([(item["face"], item["direction"]) for item in json.loads(row["scramble"])])

    assert "/no_think" in prompt
    assert "Return exactly one JSON object" in prompt
    assert "Choose exactly one legal action from this menu" in prompt
    assert "Structured state sensors:" in prompt
    assert "Static topology rules:" in prompt
    assert "Candidate neighbor sets for face matching:" in prompt
    assert "Direction index guide:" in prompt
    assert "Candidate-local moved strips:" in prompt
    assert prompt.count('"tool": "rotate"') == len(FACES) * len(DIRECTIONS)
    for candidate in FACES:
        ring = DEFAULT_TOPOLOGY.neighbor_rings[candidate]
        assert f"{candidate}: ring={' '.join(ring)}" in prompt
        for neighbor in ring:
            side = DEFAULT_TOPOLOGY.neighbor_rings[neighbor].index(candidate)
            strip_text = " ".join(
                [
                    f"c{(side - 1) % 5}={puzzle.stickers[(neighbor, 1 + ((side - 1) % 5))]}",
                    f"e{side}={puzzle.stickers[(neighbor, 6 + side)]}",
                    f"c{side}={puzzle.stickers[(neighbor, 1 + side)]}",
                ]
            )
            assert f"{neighbor}[{strip_text}]" in prompt
    assert "Highest-overlap face candidates:" not in prompt
    assert "Neighbor-overlap counts:" not in prompt
    assert row["answer"] not in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt


def test_stage_face_hint_direction_json_action_prompt_is_staged_and_two_choice() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=39,
        reward_style="action_gated_exact_direction",
        prompt_style="stage_face_hint_direction_json_action",
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    inverse_solution = json.loads(row["inverse_solution"])
    hinted_face = inverse_solution[0]["face"]

    assert "Staged face hint: affected faces identify the turned face." in prompt
    assert f"Use face {hinted_face}; infer only whether the solving direction is cw or ccw." in prompt
    assert f"Only choose rotate actions on face {hinted_face}." in prompt
    assert "Candidate-local moved strips:" in prompt
    assert prompt.count('"tool": "rotate"') == len(DIRECTIONS)
    assert '{"tool":"rotate","args":{"face":"A","direction":"cw"}}' not in prompt
    for direction in DIRECTIONS:
        assert json.dumps(
            {"tool": "rotate", "args": {"face": hinted_face, "direction": direction}}
        ) in prompt
    for face in FACES:
        if face == hinted_face:
            continue
        for direction in DIRECTIONS:
            assert json.dumps(
                {"tool": "rotate", "args": {"face": face, "direction": direction}}
            ) not in prompt
    assert "Highest-overlap face candidates:" not in prompt
    assert "Neighbor-overlap counts:" not in prompt
    assert row["answer"] not in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt


def test_stage_direction_flow_json_action_prompt_lists_local_flow_without_answer() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=41,
        reward_style="action_gated_exact_direction",
        prompt_style="stage_direction_flow_json_action",
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    puzzle = MegaminxPuzzle.solved(DEFAULT_TOPOLOGY)
    puzzle.apply_moves([(item["face"], item["direction"]) for item in json.loads(row["scramble"])])
    hinted_face = json.loads(row["inverse_solution"])[0]["face"]
    ring = DEFAULT_TOPOLOGY.neighbor_rings[hinted_face]

    assert f"Direction flow table for face {hinted_face}:" in prompt
    assert "Ring order: " + " ".join(ring) in prompt
    assert "destination | before | current | source_if_scramble_cw | source_if_scramble_ccw" in prompt
    assert "current matches source_if_scramble_cw" in prompt
    assert "current matches source_if_scramble_ccw" in prompt
    assert prompt.count('"tool": "rotate"') == len(DIRECTIONS)
    assert '{"tool":"rotate","args":{"face":"A","direction":"cw"}}' not in prompt
    for side, destination in enumerate(ring):
        previous = ring[(side - 1) % len(ring)]
        next_face = ring[(side + 1) % len(ring)]
        strip = [
            puzzle.stickers[position]
            for position in DEFAULT_TOPOLOGY.side_strip(hinted_face, destination)
        ]
        assert (
            f"{destination} | before={destination * 3} | current={''.join(strip)} | "
            f"source_if_scramble_cw={previous * 3} | source_if_scramble_ccw={next_face * 3}"
        ) in prompt
    assert "Candidate neighbor sets for face matching:" not in prompt
    assert "Direction index guide:" not in prompt
    assert "Candidate-local moved strips:" not in prompt
    for forbidden in ("correct_direction", "target_direction", "matched_source", "winner", "best"):
        assert forbidden not in prompt.lower()
    assert "Highest-overlap face candidates:" not in prompt
    assert "Neighbor-overlap counts:" not in prompt
    assert row["answer"] not in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt


def test_stage_direction_flow_reasoned_json_action_prompt_allows_private_reasoning() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=41,
        reward_style="action_gated_binary_direction",
        prompt_style="stage_direction_flow_reasoned_json_action",
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])

    assert env.env_args["allow_text_tool_actions"] is True
    assert "/no_think" not in prompt
    assert "Direct-action mode: do not reason" not in prompt
    assert "Reasoned direct-action mode: compare the direction-flow table before choosing." in prompt
    assert "You may reason internally" in prompt
    assert "Return exactly one JSON object copied from the legal-action menu below." in prompt
    assert "Direction flow table for face" in prompt
    assert "destination | before | current | source_if_scramble_cw | source_if_scramble_ccw" in prompt
    assert prompt.count('"tool": "rotate"') == len(DIRECTIONS)
    assert "Candidate neighbor sets for face matching:" not in prompt
    assert "Direction index guide:" not in prompt
    assert "Candidate-local moved strips:" not in prompt
    assert row["answer"] not in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt


def test_stage_solve_direction_flow_json_action_prompt_lists_solve_columns() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=41,
        reward_style="action_gated_binary_direction",
        prompt_style="stage_solve_direction_flow_json_action",
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    puzzle = MegaminxPuzzle.solved(DEFAULT_TOPOLOGY)
    puzzle.apply_moves([(item["face"], item["direction"]) for item in json.loads(row["scramble"])])
    hinted_face = json.loads(row["inverse_solution"])[0]["face"]
    ring = DEFAULT_TOPOLOGY.neighbor_rings[hinted_face]

    assert f"Solve-direction flow table for face {hinted_face}:" in prompt
    assert "Ring order: " + " ".join(ring) in prompt
    assert (
        "destination | solved_target | current | "
        "expected_current_if_solve_cw | expected_current_if_solve_ccw"
    ) in prompt
    assert "choose cw only if current matches expected_current_if_solve_cw on all rows" in prompt
    assert "Choose ccw only if current matches expected_current_if_solve_ccw on all rows" in prompt
    assert prompt.count('"tool": "rotate"') == len(DIRECTIONS)
    assert '{"tool":"rotate","args":{"face":"A","direction":"cw"}}' not in prompt
    for side, destination in enumerate(ring):
        previous = ring[(side - 1) % len(ring)]
        next_face = ring[(side + 1) % len(ring)]
        strip = [
            puzzle.stickers[position]
            for position in DEFAULT_TOPOLOGY.side_strip(hinted_face, destination)
        ]
        assert (
            f"{destination} | solved_target={destination * 3} | current={''.join(strip)} | "
            f"expected_current_if_solve_cw={next_face * 3} | "
            f"expected_current_if_solve_ccw={previous * 3}"
        ) in prompt
    for forbidden in ("correct_direction", "target_direction", "matched_source", "winner", "best"):
        assert forbidden not in prompt.lower()
    assert "source_if_scramble_cw" not in prompt
    assert "source_if_scramble_ccw" not in prompt
    assert row["answer"] not in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt


def test_stage_solve_direction_flow_native_tool_v2_lists_all_candidates_without_face_hint() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=41,
        reward_style="action_gated_binary_direction",
        prompt_style="stage_solve_direction_flow_native_tool_v2",
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    puzzle = MegaminxPuzzle.solved(DEFAULT_TOPOLOGY)
    puzzle.apply_moves([(item["face"], item["direction"]) for item in json.loads(row["scramble"])])

    assert env.env_args["allow_text_tool_actions"] is False
    assert "Native tool mode: compare the direction-flow table, then call rotate directly." in prompt
    assert "Structured state sensors:" in prompt
    assert "Static topology rules:" in prompt
    assert "Candidate neighbor sets for face matching:" in prompt
    assert "Candidate-local moved strips:" in prompt
    assert "All-candidate solve-direction flow table:" in prompt
    assert (
        "candidate | destination | solved_target | current | "
        "expected_current_if_solve_cw | expected_current_if_solve_ccw"
    ) in prompt
    assert "Staged face hint:" not in prompt
    assert "Use face " not in prompt
    assert "Only choose rotate actions on face" not in prompt
    assert "Solve-direction flow table for face" not in prompt
    assert "Return exactly one JSON object" not in prompt
    assert "Choose exactly one legal action from this menu" not in prompt
    assert prompt.count(" | solved_target=") == len(FACES) * 5
    for candidate in FACES:
        assert f"{candidate} ring: {' '.join(DEFAULT_TOPOLOGY.neighbor_rings[candidate])}" in prompt
    for forbidden in ("correct_direction", "target_direction", "matched_source", "winner", "best"):
        assert forbidden not in prompt.lower()
    assert row["answer"] not in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt


def test_stage_solve_direction_flow_native_tool_v2_depth_range_contract() -> None:
    env = load_environment(
        min_depth=1,
        max_depth=2,
        num_examples=2,
        seed=44,
        reward_style="action_gated_strict_shaped_direction",
        prompt_style="stage_solve_direction_flow_native_tool_v2",
        move_budget=1,
    )
    rows = [dict(row) for row in env.get_dataset()]
    assert {row["scramble_depth"] for row in rows} == {1, 2}
    assert all(
        "All-candidate solve-direction flow table:" in "\n".join(
            message["content"] for message in row["prompt"]
        )
        for row in rows
    )

    try:
        load_environment(
            min_depth=1,
            max_depth=3,
            num_examples=3,
            reward_style="action_gated_strict_shaped_direction",
            prompt_style="stage_solve_direction_flow_native_tool_v2",
            move_budget=1,
        )
    except ValueError as error:
        assert "only defined for depths 1-2" in str(error)
    else:
        raise AssertionError("Expected v2 solve-direction prompt to reject max_depth > 2")


def test_stage_direction_flow_prompt_local_oracle_solves_all_depth1_moves() -> None:
    for prompt_style in (
        "stage_direction_flow_json_action",
        "stage_direction_flow_reasoned_json_action",
        "stage_direction_flow_native_tool",
        "stage_solve_direction_flow_json_action",
        "stage_solve_direction_flow_native_tool",
    ):
        env = load_environment(
            split="depth1",
            num_examples=len(FACES) * len(DIRECTIONS),
            seed=1042,
            reward_style="action_gated_binary_direction",
            prompt_style=prompt_style,
            move_budget=1,
        )
        for row in env.get_dataset():
            row = dict(row)
            prompt = "\n".join(message["content"] for message in row["prompt"])
            inverse_solution = json.loads(row["inverse_solution"])
            hinted_face = inverse_solution[0]["face"]
            rows = [
                line
                for line in prompt.splitlines()
                if (" | before=" in line and "source_if_scramble_" in line)
                or (" | solved_target=" in line and "expected_current_if_solve_" in line)
            ]
            assert len(rows) == 5
            cw_matches = []
            ccw_matches = []
            for line in rows:
                parts = [part.strip() for part in line.split("|")]
                current = parts[2].split("=", 1)[1]
                if "source_if_scramble_" in line:
                    source_if_cw = parts[3].split("=", 1)[1]
                    source_if_ccw = parts[4].split("=", 1)[1]
                    cw_matches.append(current == source_if_cw)
                    ccw_matches.append(current == source_if_ccw)
                else:
                    expected_current_if_solve_cw = parts[3].split("=", 1)[1]
                    expected_current_if_solve_ccw = parts[4].split("=", 1)[1]
                    cw_matches.append(current == expected_current_if_solve_cw)
                    ccw_matches.append(current == expected_current_if_solve_ccw)
            if "source_if_scramble_" in rows[0]:
                if all(cw_matches):
                    chosen = (hinted_face, "ccw")
                elif all(ccw_matches):
                    chosen = (hinted_face, "cw")
                else:
                    raise AssertionError(f"Prompt has ambiguous flow rows:\n{prompt}")
            elif all(cw_matches):
                chosen = (hinted_face, "cw")
            elif all(ccw_matches):
                chosen = (hinted_face, "ccw")
            else:
                raise AssertionError(f"Prompt has ambiguous solve-flow rows:\n{prompt}")
            assert chosen == (inverse_solution[0]["face"], inverse_solution[0]["direction"])


def test_stage_solve_direction_flow_native_tool_v2_oracle_solves_all_depth1_moves() -> None:
    env = load_environment(
        split="depth1",
        num_examples=len(FACES) * len(DIRECTIONS),
        seed=1042,
        reward_style="action_gated_binary_direction",
        prompt_style="stage_solve_direction_flow_native_tool_v2",
        move_budget=1,
    )
    for row in env.get_dataset():
        row = dict(row)
        prompt = "\n".join(message["content"] for message in row["prompt"])
        inverse_solution = json.loads(row["inverse_solution"])
        rows = [
            line
            for line in prompt.splitlines()
            if " | solved_target=" in line and "expected_current_if_solve_" in line
        ]
        assert len(rows) == len(FACES) * 5
        matches: list[tuple[str, str]] = []
        for candidate in FACES:
            candidate_rows = [line for line in rows if line.startswith(f"{candidate} | ")]
            assert len(candidate_rows) == 5
            cw_matches = []
            ccw_matches = []
            for line in candidate_rows:
                parts = [part.strip() for part in line.split("|")]
                current = parts[3].split("=", 1)[1]
                expected_current_if_solve_cw = parts[4].split("=", 1)[1]
                expected_current_if_solve_ccw = parts[5].split("=", 1)[1]
                cw_matches.append(current == expected_current_if_solve_cw)
                ccw_matches.append(current == expected_current_if_solve_ccw)
            if all(cw_matches):
                matches.append((candidate, "cw"))
            if all(ccw_matches):
                matches.append((candidate, "ccw"))
        assert matches == [(inverse_solution[0]["face"], inverse_solution[0]["direction"])]


def test_stage_solve_action_table_native_tool_lists_all_actions_without_hint_or_scores() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=41,
        reward_style="action_gated_overlap_strict_shaped_direction",
        prompt_style="stage_solve_action_table_native_tool",
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])

    assert env.env_args["allow_text_tool_actions"] is False
    assert "Native tool mode: call the rotate tool directly." in prompt
    assert "Do not write JSON, prose, or analysis before the tool call." in prompt
    assert "Solve-action evidence table:" in prompt
    assert "action | evidence" in prompt
    assert "All-candidate solve-direction flow table:" not in prompt
    assert "Structured state sensors:" not in prompt
    assert "Static topology rules:" not in prompt
    assert "Candidate-local moved strips:" not in prompt
    assert "Return exactly one JSON object" not in prompt
    assert "Choose exactly one legal action from this menu" not in prompt
    assert "Staged face hint:" not in prompt
    assert "Use face " not in prompt
    assert "Only choose rotate actions on face" not in prompt
    assert prompt.count(" current=") == len(FACES) * len(DIRECTIONS) * 5
    for face in FACES:
        for direction in DIRECTIONS:
            assert f"{face}:{direction} | " in prompt
    for forbidden in (
        "correct_direction",
        "target_direction",
        "matched_source",
        "winner",
        "best",
        "score",
        "scramble:",
        "inverse_solution",
    ):
        assert forbidden not in prompt.lower()
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt


def test_stage_solve_action_table_native_tool_depth_range_contract() -> None:
    env = load_environment(
        min_depth=1,
        max_depth=2,
        num_examples=2,
        seed=45,
        reward_style="action_gated_overlap_strict_shaped_direction",
        prompt_style="stage_solve_action_table_native_tool",
        move_budget=1,
    )
    rows = [dict(row) for row in env.get_dataset()]
    assert {row["scramble_depth"] for row in rows} == {1, 2}
    assert all(
        "Solve-action evidence table:" in "\n".join(
            message["content"] for message in row["prompt"]
        )
        for row in rows
    )

    try:
        load_environment(
            min_depth=1,
            max_depth=3,
            num_examples=3,
            reward_style="action_gated_overlap_strict_shaped_direction",
            prompt_style="stage_solve_action_table_native_tool",
            move_budget=1,
        )
    except ValueError as error:
        assert "only defined for depths 1-2" in str(error)
    else:
        raise AssertionError("Expected action-table prompt to reject max_depth > 2")


def test_stage_solve_action_table_native_tool_oracle_solves_all_depth1_moves() -> None:
    env = load_environment(
        split="depth1",
        num_examples=len(FACES) * len(DIRECTIONS),
        seed=1042,
        reward_style="action_gated_binary_direction",
        prompt_style="stage_solve_action_table_native_tool",
        move_budget=1,
    )
    for row in env.get_dataset():
        row = dict(row)
        prompt = "\n".join(message["content"] for message in row["prompt"])
        inverse_solution = json.loads(row["inverse_solution"])
        rows = [
            line
            for line in prompt.splitlines()
            if any(line.startswith(f"{face}:{direction} | ") for face in FACES for direction in DIRECTIONS)
        ]
        assert len(rows) == len(FACES) * len(DIRECTIONS)
        matches: list[tuple[str, str]] = []
        for line in rows:
            action, *evidence = [part.strip() for part in line.split("|")]
            assert len(evidence) == 5
            if all(
                part.split(" current=", 1)[1].split(" expected=", 1)[0]
                == part.split(" expected=", 1)[1]
                for part in evidence
            ):
                face, direction = action.split(":", 1)
                matches.append((face, direction))
        assert matches == [(inverse_solution[0]["face"], inverse_solution[0]["direction"])]


def _parse_action_mask_rows(prompt: str) -> dict[tuple[str, str], str]:
    masks: dict[tuple[str, str], str] = {}
    for line in prompt.splitlines():
        if not any(line.startswith(f"{face} | ") for face in FACES):
            continue
        face, _ring, cw_mask, ccw_mask = [part.strip() for part in line.split("|")]
        masks[(face, "cw")] = cw_mask
        masks[(face, "ccw")] = ccw_mask
    return masks


def test_stage_solve_action_mask_native_tool_lists_compact_masks_without_leaks() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=41,
        reward_style="action_gated_mask_overlap_strict_shaped_direction",
        prompt_style="stage_solve_action_mask_native_tool",
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])

    assert env.env_args["allow_text_tool_actions"] is False
    assert "Native tool mode: call the rotate tool directly." in prompt
    assert "Solve-action equality-mask table:" in prompt
    assert "face | ring | cw_mask | ccw_mask" in prompt
    assert "Solve-action evidence table:" not in prompt
    assert "All-candidate solve-direction flow table:" not in prompt
    assert "Structured state sensors:" not in prompt
    assert "Static topology rules:" not in prompt
    assert "Candidate-local moved strips:" not in prompt
    assert "current=" not in prompt
    assert "expected=" not in prompt
    assert "Return exactly one JSON object" not in prompt
    assert "Choose exactly one legal action from this menu" not in prompt
    assert "Staged face hint:" not in prompt

    masks = _parse_action_mask_rows(prompt)
    assert len(masks) == len(FACES) * len(DIRECTIONS)
    assert all(len(mask) == 5 and set(mask) <= {"0", "1"} for mask in masks.values())
    mask_section = prompt.split("Solve-action equality-mask table:", 1)[1].split(
        "Initial observation:", 1
    )[0]
    for forbidden in (
        "answer",
        "correct",
        "target",
        "winner",
        "best",
        "score",
        "scramble:",
        "inverse_solution",
    ):
        assert forbidden not in mask_section.lower()
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt


def test_stage_solve_action_mask_native_tool_depth_range_contract() -> None:
    env = load_environment(
        min_depth=1,
        max_depth=2,
        num_examples=2,
        seed=45,
        reward_style="action_gated_mask_overlap_strict_shaped_direction",
        prompt_style="stage_solve_action_mask_native_tool",
        move_budget=1,
    )
    rows = [dict(row) for row in env.get_dataset()]
    assert {row["scramble_depth"] for row in rows} == {1, 2}
    assert all(
        "Solve-action equality-mask table:" in "\n".join(
            message["content"] for message in row["prompt"]
        )
        for row in rows
    )

    try:
        load_environment(
            min_depth=1,
            max_depth=3,
            num_examples=3,
            reward_style="action_gated_mask_overlap_strict_shaped_direction",
            prompt_style="stage_solve_action_mask_native_tool",
            move_budget=1,
        )
    except ValueError as error:
        assert "only defined for depths 1-2" in str(error)
    else:
        raise AssertionError("Expected action-mask prompt to reject max_depth > 2")


def test_stage_solve_action_mask_native_tool_oracle_solves_all_depth1_moves() -> None:
    env = load_environment(
        split="depth1",
        num_examples=len(FACES) * len(DIRECTIONS),
        seed=1042,
        reward_style="action_gated_binary_direction",
        prompt_style="stage_solve_action_mask_native_tool",
        move_budget=1,
    )
    for row in env.get_dataset():
        row = dict(row)
        prompt = "\n".join(message["content"] for message in row["prompt"])
        inverse_solution = json.loads(row["inverse_solution"])
        masks = _parse_action_mask_rows(prompt)
        matches = [move for move, mask in masks.items() if mask == "11111"]
        assert matches == [(inverse_solution[0]["face"], inverse_solution[0]["direction"])]


def test_stage_solve_action_mask_native_tool_keeps_depth2_inverse_among_max_masks() -> None:
    env = load_environment(
        min_depth=2,
        max_depth=2,
        num_examples=50,
        seed=91,
        reward_style="action_gated_mask_overlap_strict_shaped_direction",
        prompt_style="stage_solve_action_mask_native_tool",
        move_budget=1,
    )
    for row in env.get_dataset():
        row = dict(row)
        prompt = "\n".join(message["content"] for message in row["prompt"])
        inverse_solution = json.loads(row["inverse_solution"])
        masks = _parse_action_mask_rows(prompt)
        counts = {move: mask.count("1") for move, mask in masks.items()}
        max_count = max(counts.values())
        inverse_first = (inverse_solution[0]["face"], inverse_solution[0]["direction"])
        assert counts[inverse_first] == max_count


def test_staged_face_hint_prompt_styles_reject_non_depth1_splits() -> None:
    for prompt_style in (
        "stage_face_hint_direction_json_action",
        "stage_direction_flow_json_action",
        "stage_direction_flow_reasoned_json_action",
        "stage_direction_flow_native_tool",
        "stage_solve_direction_flow_json_action",
        "stage_solve_direction_flow_native_tool",
    ):
        try:
            load_environment(
                split="easy",
                num_examples=3,
                reward_style="action_gated_exact_direction",
                prompt_style=prompt_style,
                move_budget=1,
            )
        except ValueError as error:
            assert "only defined for depth-1" in str(error)
        else:
            raise AssertionError(f"Expected {prompt_style} to reject non-depth1 split")


def test_choice_json_action_prompt_style_lists_all_legal_actions() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=20,
        reward_style="action_gated_curriculum",
        prompt_style="choice_json_action",
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    assert "Choose exactly one legal action from this menu" in prompt
    assert prompt.count('"tool": "rotate"') == len(FACES) * len(DIRECTIONS)
    for face in FACES:
        for direction in DIRECTIONS:
            assert json.dumps(
                {"tool": "rotate", "args": {"face": face, "direction": direction}}
            ) in prompt
    assert row["answer"] not in prompt


def test_topology_choice_json_action_prompt_style_lists_static_rings_without_answer() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=21,
        reward_style="action_gated_curriculum",
        prompt_style="topology_choice_json_action",
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    assert "Choose exactly one legal action from this menu" in prompt
    assert prompt.count('"tool": "rotate"') == len(FACES) * len(DIRECTIONS)
    assert "Static topology rules:" in prompt
    assert "A turned face can still look solved" in prompt
    assert "the correct face is the face whose five listed neighbors are the affected faces" in prompt
    assert "For a candidate ring X: N0 N1 N2 N3 N4" in prompt
    assert "If N_i's strip shows label N_{i-1}, the scramble was X:cw" in prompt
    assert "If N_i's strip shows label N_{i+1}, the scramble was X:ccw" in prompt
    for face in FACES:
        assert f"{face}: {' '.join(DEFAULT_TOPOLOGY.neighbor_rings[face])}" in prompt
    assert row["answer"] not in prompt


def test_sensor_choice_json_action_prompt_lists_derived_affected_faces() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=26,
        reward_style="action_gated_curriculum",
        prompt_style="sensor_choice_json_action",
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    puzzle = MegaminxPuzzle.solved(DEFAULT_TOPOLOGY)
    puzzle.apply_moves([(item["face"], item["direction"]) for item in json.loads(row["scramble"])])
    affected_faces = tuple(
        face
        for face in FACES
        if any(puzzle.stickers[(face, index)] != face for index in range(1, POSITIONS_PER_FACE))
    )

    assert "Structured state sensors:" in prompt
    assert "Affected faces: " + " ".join(affected_faces) in prompt
    assert "Neighbor-overlap counts:" not in prompt
    assert "Highest-overlap face candidates:" not in prompt
    assert "Affected face summaries:" in prompt
    assert "Static topology rules:" in prompt
    assert prompt.index("Structured state sensors:") < prompt.index("Choose exactly one legal action")
    for face in affected_faces:
        assert puzzle.face_line(face) in prompt
    assert row["answer"] not in prompt


def test_action_gated_inverse_solution_gets_full_reward() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=16,
            reward_style="action_gated_dense",
            prompt_style="action_first",
        )
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        for face, direction in rollout.inverse_solution:
            await env.rotate(face, direction, rollout)
        assert rollout.solved()
        assert await action_gated_dense_reward(state) == 1.0
        assert await first_rotate_correct(state) == 1.0

    asyncio.run(run())


def test_action_gated_curriculum_gives_partial_first_move_signal() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=18,
            reward_style="action_gated_curriculum",
            prompt_style="action_first",
        )
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        target_face, target_direction = rollout.inverse_solution[0]
        wrong_direction = next(direction for direction in DIRECTIONS if direction != target_direction)

        await env.rotate(target_face, wrong_direction, rollout)

        assert await reward_style(state) == 2.0
        assert await action_taken(state) == 1.0
        assert await first_rotate_direction_correct(state) == 0.0
        assert await first_rotate_face_correct(state) == 1.0
        assert 0.35 <= await action_gated_curriculum_reward(state) <= 0.65

    asyncio.run(run())


def test_action_gated_direction_rewards_direction_separately() -> None:
    async def run() -> None:
        wrong_direction_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=36,
            reward_style="action_gated_direction",
            prompt_style="sensor_indexed_match_json_action",
            max_turns=2,
            move_budget=1,
        )
        wrong_direction_state = {"task": dict(wrong_direction_env.get_dataset()[0])}
        await wrong_direction_env.setup_state(wrong_direction_state)
        wrong_direction_rollout = wrong_direction_state["megaminx"]
        target_face, target_direction = wrong_direction_rollout.inverse_solution[0]
        wrong_direction = next(direction for direction in DIRECTIONS if direction != target_direction)

        assert await reward_style(wrong_direction_state) == 4.0
        assert await action_gated_direction_reward(wrong_direction_state) == 0.0
        await wrong_direction_env.rotate(target_face, wrong_direction, wrong_direction_rollout)
        wrong_direction_reward = await action_gated_direction_reward(wrong_direction_state)
        assert await first_rotate_face_correct(wrong_direction_state) == 1.0
        assert await first_rotate_direction_correct(wrong_direction_state) == 0.0
        assert 0.20 <= wrong_direction_reward <= 0.35

        wrong_face_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=36,
            reward_style="action_gated_direction",
            prompt_style="sensor_indexed_match_json_action",
            max_turns=2,
            move_budget=1,
        )
        wrong_face_state = {"task": dict(wrong_face_env.get_dataset()[0])}
        await wrong_face_env.setup_state(wrong_face_state)
        wrong_face_rollout = wrong_face_state["megaminx"]
        wrong_face = next(face for face in FACES if face != target_face)
        await wrong_face_env.rotate(wrong_face, target_direction, wrong_face_rollout)
        wrong_face_reward = await action_gated_direction_reward(wrong_face_state)
        assert await first_rotate_face_correct(wrong_face_state) == 0.0
        assert await first_rotate_direction_correct(wrong_face_state) == 1.0
        assert wrong_face_reward < wrong_direction_reward

        solved_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=36,
            reward_style="action_gated_direction",
            prompt_style="sensor_indexed_match_json_action",
            max_turns=2,
            move_budget=1,
        )
        solved_state = {"task": dict(solved_env.get_dataset()[0])}
        await solved_env.setup_state(solved_state)
        solved_rollout = solved_state["megaminx"]
        await solved_env.rotate(*solved_rollout.inverse_solution[0], solved_rollout)
        assert solved_rollout.solved()
        assert await action_gated_direction_reward(solved_state) == 1.0

    asyncio.run(run())


def test_action_gated_exact_direction_ignores_wrong_face_direction_bonus() -> None:
    async def run() -> None:
        target_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=38,
            reward_style="action_gated_exact_direction",
            prompt_style="sensor_candidate_strips_json_action",
            max_turns=2,
            move_budget=1,
        )
        target_state = {"task": dict(target_env.get_dataset()[0])}
        await target_env.setup_state(target_state)
        target_rollout = target_state["megaminx"]
        target_face, target_direction = target_rollout.inverse_solution[0]
        wrong_direction = next(direction for direction in DIRECTIONS if direction != target_direction)

        assert await reward_style(target_state) == 5.0
        assert await action_gated_exact_direction_reward(target_state) == 0.0
        await target_env.rotate(target_face, wrong_direction, target_rollout)
        target_reward = await action_gated_exact_direction_reward(target_state)
        assert await first_rotate_face_correct(target_state) == 1.0
        assert await first_rotate_direction_correct(target_state) == 0.0
        assert 0.14 <= target_reward <= 0.16

        wrong_face_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=38,
            reward_style="action_gated_exact_direction",
            prompt_style="sensor_candidate_strips_json_action",
            max_turns=2,
            move_budget=1,
        )
        wrong_face_state = {"task": dict(wrong_face_env.get_dataset()[0])}
        await wrong_face_env.setup_state(wrong_face_state)
        wrong_face_rollout = wrong_face_state["megaminx"]
        wrong_face = next(face for face in FACES if face != target_face)
        await wrong_face_env.rotate(wrong_face, target_direction, wrong_face_rollout)
        wrong_face_reward = await action_gated_exact_direction_reward(wrong_face_state)
        assert await first_rotate_face_correct(wrong_face_state) == 0.0
        assert await first_rotate_direction_correct(wrong_face_state) == 1.0
        assert wrong_face_reward == 0.02
        assert wrong_face_reward < target_reward

        solved_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=38,
            reward_style="action_gated_exact_direction",
            prompt_style="sensor_candidate_strips_json_action",
            max_turns=2,
            move_budget=1,
        )
        solved_state = {"task": dict(solved_env.get_dataset()[0])}
        await solved_env.setup_state(solved_state)
        solved_rollout = solved_state["megaminx"]
        await solved_env.rotate(*solved_rollout.inverse_solution[0], solved_rollout)
        assert solved_rollout.solved()
        assert await action_gated_exact_direction_reward(solved_state) == 1.0

    asyncio.run(run())


def test_action_gated_binary_direction_reward_is_binary() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_binary_direction",
            prompt_style="stage_direction_flow_json_action",
            max_turns=2,
            move_budget=1,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        target_face, target_direction = rollout.inverse_solution[0]
        wrong_direction = next(direction for direction in DIRECTIONS if direction != target_direction)

        assert await reward_style(state) == 6.0
        assert await action_gated_binary_direction_reward(state) == 0.0
        await env.rotate(target_face, wrong_direction, rollout)
        assert await action_gated_binary_direction_reward(state) == 0.0

        wrong_face_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_binary_direction",
            prompt_style="stage_direction_flow_json_action",
            max_turns=2,
            move_budget=1,
        )
        wrong_face_state = {"task": dict(wrong_face_env.get_dataset()[0])}
        await wrong_face_env.setup_state(wrong_face_state)
        wrong_face_rollout = wrong_face_state["megaminx"]
        wrong_face = next(face for face in FACES if face != target_face)
        await wrong_face_env.rotate(wrong_face, target_direction, wrong_face_rollout)
        assert await action_gated_binary_direction_reward(wrong_face_state) == 0.0

        solved_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_binary_direction",
            prompt_style="stage_direction_flow_json_action",
            max_turns=2,
            move_budget=1,
        )
        solved_state = {"task": dict(solved_env.get_dataset()[0])}
        await solved_env.setup_state(solved_state)
        solved_rollout = solved_state["megaminx"]
        face, direction = solved_rollout.inverse_solution[0]
        message = AssistantMessage(
            content=json.dumps({"tool": "rotate", "args": {"face": face, "direction": direction}})
        )
        await solved_env.env_response([message], solved_state)
        assert solved_rollout.solved()
        assert await text_tool_action_count(solved_state) == 1.0
        assert await action_gated_binary_direction_reward(solved_state) == 1.0

    asyncio.run(run())


def test_action_gated_binary_direction_reward_requires_clean_first_action() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_binary_direction",
            prompt_style="stage_direction_flow_json_action",
            max_turns=2,
            move_budget=1,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]

        await env.rotate("Z", "cw", rollout)
        await env.rotate(*rollout.inverse_solution[0], rollout)

        assert rollout.solved()
        assert rollout.illegal_moves == 1
        assert await action_gated_binary_direction_reward(state) == 0.0

    asyncio.run(run())


def test_action_gated_strict_shaped_direction_reward_teaches_clean_partial_credit() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_strict_shaped_direction",
            prompt_style="stage_solve_direction_flow_json_action",
            max_turns=2,
            move_budget=1,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        target_face, target_direction = rollout.inverse_solution[0]
        wrong_direction = next(direction for direction in DIRECTIONS if direction != target_direction)

        message = AssistantMessage(
            content=json.dumps(
                {"tool": "rotate", "args": {"face": target_face, "direction": wrong_direction}}
            )
        )
        await env.env_response([message], state)

        assert await reward_style(state) == 7.0
        assert await first_rotate_face_correct(state) == 1.0
        assert await first_rotate_direction_correct(state) == 0.0
        assert await action_gated_binary_direction_reward(state) == 0.0
        assert abs(await action_gated_strict_shaped_direction_reward(state) - 0.40) < 1e-9

        wrong_face_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_strict_shaped_direction",
            prompt_style="stage_solve_direction_flow_json_action",
            max_turns=2,
            move_budget=1,
        )
        wrong_face_state = {"task": dict(wrong_face_env.get_dataset()[0])}
        await wrong_face_env.setup_state(wrong_face_state)
        wrong_face = next(face for face in FACES if face != target_face)
        wrong_face_message = AssistantMessage(
            content=json.dumps(
                {"tool": "rotate", "args": {"face": wrong_face, "direction": target_direction}}
            )
        )
        await wrong_face_env.env_response([wrong_face_message], wrong_face_state)

        assert await first_rotate_face_correct(wrong_face_state) == 0.0
        assert await first_rotate_direction_correct(wrong_face_state) == 1.0
        assert abs(await action_gated_strict_shaped_direction_reward(wrong_face_state) - 0.10) < 1e-9

        solved_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_strict_shaped_direction",
            prompt_style="stage_solve_direction_flow_json_action",
            max_turns=2,
            move_budget=1,
        )
        solved_state = {"task": dict(solved_env.get_dataset()[0])}
        await solved_env.setup_state(solved_state)
        solved_rollout = solved_state["megaminx"]
        face, direction = solved_rollout.inverse_solution[0]
        solved_message = AssistantMessage(
            content=json.dumps({"tool": "rotate", "args": {"face": face, "direction": direction}})
        )
        await solved_env.env_response([solved_message], solved_state)

        assert solved_rollout.solved()
        assert await action_gated_binary_direction_reward(solved_state) == 1.0
        assert await action_gated_strict_shaped_direction_reward(solved_state) == 1.0

    asyncio.run(run())


def test_action_gated_overlap_strict_shaped_direction_reward_smooths_face_without_wrong_face_direction_bonus() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_overlap_strict_shaped_direction",
            prompt_style="stage_solve_action_table_native_tool",
            max_turns=2,
            move_budget=1,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        target_face, target_direction = rollout.inverse_solution[0]
        wrong_direction = next(direction for direction in DIRECTIONS if direction != target_direction)
        wrong_face = next(face for face in FACES if face != target_face)

        rollout.move_count = 1
        rollout.rotate_call_count = 1
        rollout.native_tool_call_count = 1

        rollout.first_rotate = (target_face, wrong_direction)
        correct_face_wrong_direction = await action_gated_overlap_strict_shaped_direction_reward(state)
        assert abs(correct_face_wrong_direction - 0.52) < 1e-9

        rollout.first_rotate = (wrong_face, target_direction)
        wrong_face_correct_direction = await action_gated_overlap_strict_shaped_direction_reward(state)
        rollout.first_rotate = (wrong_face, wrong_direction)
        wrong_face_wrong_direction = await action_gated_overlap_strict_shaped_direction_reward(state)

        assert wrong_face_correct_direction == wrong_face_wrong_direction
        assert wrong_face_correct_direction < correct_face_wrong_direction

        solved_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_overlap_strict_shaped_direction",
            prompt_style="stage_solve_action_table_native_tool",
            max_turns=2,
            move_budget=1,
        )
        solved_state = {"task": dict(solved_env.get_dataset()[0])}
        await solved_env.setup_state(solved_state)
        solved_rollout = solved_state["megaminx"]
        face, direction = solved_rollout.inverse_solution[0]
        solved_message = AssistantMessage(
            content="",
            tool_calls=[
                {
                    "id": "native-call",
                    "name": "rotate",
                    "arguments": json.dumps({"face": face, "direction": direction}),
                }
            ],
        )
        await solved_env.env_response([solved_message], solved_state)

        assert solved_rollout.solved()
        assert await action_gated_overlap_strict_shaped_direction_reward(solved_state) == 1.0

        invalid_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_overlap_strict_shaped_direction",
            prompt_style="stage_solve_action_table_native_tool",
            max_turns=2,
            move_budget=1,
        )
        invalid_state = {"task": dict(invalid_env.get_dataset()[0])}
        await invalid_env.setup_state(invalid_state)
        assert await action_gated_overlap_strict_shaped_direction_reward(invalid_state) == 0.0

    asyncio.run(run())


def test_action_gated_mask_overlap_strict_shaped_direction_reward_uses_public_mask_signal() -> None:
    async def run() -> None:
        env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=1,
            seed=91,
            reward_style="action_gated_mask_overlap_strict_shaped_direction",
            prompt_style="stage_solve_action_mask_native_tool",
            max_turns=2,
            move_budget=1,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        assert await reward_style(state) == 9.0
        assert await initial_max_action_mask_count(state) >= 1.0

        max_mask_action = max(
            rollout.initial_action_mask_counts,
            key=lambda move: rollout.initial_action_mask_counts[move],
        )
        message = AssistantMessage(
            content="",
            tool_calls=[
                {
                    "id": "native-call",
                    "name": "rotate",
                    "arguments": json.dumps(
                        {"face": max_mask_action[0], "direction": max_mask_action[1]}
                    ),
                }
            ],
        )
        await env.env_response([message], state)

        reward = await action_gated_mask_overlap_strict_shaped_direction_reward(state)
        assert await first_rotate_mask_is_max(state) == 1.0
        assert await first_rotate_mask_count(state) == float(
            rollout.initial_action_mask_counts[max_mask_action]
        )
        assert 0.0 < reward < 1.0

        solved_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_mask_overlap_strict_shaped_direction",
            prompt_style="stage_solve_action_mask_native_tool",
            max_turns=2,
            move_budget=1,
        )
        solved_state = {"task": dict(solved_env.get_dataset()[0])}
        await solved_env.setup_state(solved_state)
        solved_rollout = solved_state["megaminx"]
        face, direction = solved_rollout.inverse_solution[0]
        solved_message = AssistantMessage(
            content="",
            tool_calls=[
                {
                    "id": "native-call",
                    "name": "rotate",
                    "arguments": json.dumps({"face": face, "direction": direction}),
                }
            ],
        )
        await solved_env.env_response([solved_message], solved_state)
        assert solved_rollout.solved()
        assert await action_gated_mask_overlap_strict_shaped_direction_reward(solved_state) == 1.0

        invalid_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_mask_overlap_strict_shaped_direction",
            prompt_style="stage_solve_action_mask_native_tool",
            max_turns=2,
            move_budget=1,
        )
        invalid_state = {"task": dict(invalid_env.get_dataset()[0])}
        await invalid_env.setup_state(invalid_state)
        assert await action_gated_mask_overlap_strict_shaped_direction_reward(invalid_state) == 0.0

        protocol_state = {"task": dict(invalid_env.get_dataset()[0])}
        await invalid_env.setup_state(protocol_state)
        protocol_rollout = protocol_state["megaminx"]
        face, direction = protocol_rollout.inverse_solution[0]
        protocol_message = AssistantMessage(
            content="visible text",
            tool_calls=[
                {
                    "id": "native-call",
                    "name": "rotate",
                    "arguments": json.dumps({"face": face, "direction": direction}),
                }
            ],
        )
        await invalid_env.env_response([protocol_message], protocol_state)
        assert protocol_rollout.solved()
        assert await action_gated_mask_overlap_strict_shaped_direction_reward(protocol_state) == 0.0

    asyncio.run(run())


def test_stage_frontier_sensor_native_tool_uses_raw_sensors_without_action_masks() -> None:
    env = load_environment(
        min_depth=1,
        max_depth=2,
        num_examples=2,
        seed=52,
        reward_style="action_gated_counterfactual_frontier_strict",
        prompt_style="stage_frontier_sensor_native_tool",
        move_budget=1,
    )
    rows = [dict(row) for row in env.get_dataset()]
    assert env.env_args["allow_text_tool_actions"] is False
    assert {row["scramble_depth"] for row in rows} == {1, 2}
    for row in rows:
        prompt = "\n".join(message["content"] for message in row["prompt"])
        assert "Frontier sensor objective:" in prompt
        assert "Structured state sensors:" in prompt
        assert "Candidate-local moved strips:" in prompt
        assert "Solve-action equality-mask table:" not in prompt
        assert "Solve-action evidence table:" not in prompt
        assert "All-candidate solve-direction flow table:" not in prompt
        assert "expected_current_if_solve" not in prompt
        assert "expected=" not in prompt
        assert "cw_mask" not in prompt
        assert "ccw_mask" not in prompt
        assert "11111" not in prompt
        assert row["scramble"] not in prompt
        assert row["inverse_solution"] not in prompt

    try:
        load_environment(
            min_depth=1,
            max_depth=3,
            num_examples=3,
            reward_style="action_gated_counterfactual_frontier_strict",
            prompt_style="stage_frontier_sensor_native_tool",
            move_budget=1,
        )
    except ValueError as error:
        assert "only defined for depths 1-2" in str(error)
    else:
        raise AssertionError("Expected frontier sensor prompt to reject max_depth > 2")


def test_action_gated_counterfactual_frontier_reward_handles_depth1_and_depth2() -> None:
    async def run() -> None:
        depth1_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_counterfactual_frontier_strict",
            prompt_style="stage_frontier_sensor_native_tool",
            max_turns=2,
            move_budget=1,
        )
        depth1_state = {"task": dict(depth1_env.get_dataset()[0])}
        await depth1_env.setup_state(depth1_state)
        depth1_rollout = depth1_state["megaminx"]
        face, direction = depth1_rollout.inverse_solution[0]
        await depth1_env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "rotate",
                            "arguments": json.dumps({"face": face, "direction": direction}),
                        }
                    ],
                )
            ],
            depth1_state,
        )
        assert depth1_rollout.solved()
        assert await action_gated_counterfactual_frontier_strict_reward(depth1_state) == 1.0
        assert await reward_style(depth1_state) == 10.0

        wrong_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_counterfactual_frontier_strict",
            prompt_style="stage_frontier_sensor_native_tool",
            max_turns=2,
            move_budget=1,
        )
        wrong_state = {"task": dict(wrong_env.get_dataset()[0])}
        await wrong_env.setup_state(wrong_state)
        wrong_rollout = wrong_state["megaminx"]
        face, direction = wrong_rollout.inverse_solution[0]
        wrong_direction = next(item for item in DIRECTIONS if item != direction)
        await wrong_env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "rotate",
                            "arguments": json.dumps(
                                {"face": face, "direction": wrong_direction}
                            ),
                        }
                    ],
                )
            ],
            wrong_state,
        )
        wrong_reward = await action_gated_counterfactual_frontier_strict_reward(wrong_state)
        assert not wrong_rollout.solved()
        assert await first_rotate_face_frontier_viable(wrong_state) == 1.0
        assert await first_rotate_counterfactual_frontier(wrong_state) == 0.0
        assert 0.0 < wrong_reward < 0.55

        depth2_env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=1,
            seed=77,
            reward_style="action_gated_counterfactual_frontier_strict",
            prompt_style="stage_frontier_sensor_native_tool",
            max_turns=2,
            move_budget=1,
        )
        depth2_state = {"task": dict(depth2_env.get_dataset()[0])}
        await depth2_env.setup_state(depth2_state)
        depth2_rollout = depth2_state["megaminx"]
        face, direction = depth2_rollout.inverse_solution[0]
        await depth2_env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "rotate",
                            "arguments": json.dumps({"face": face, "direction": direction}),
                        }
                    ],
                )
            ],
            depth2_state,
        )
        assert not depth2_rollout.solved()
        assert await first_rotate_frontier_tail_count(depth2_state) >= 1.0
        assert await first_rotate_frontier_tail_unique(depth2_state) in {0.0, 1.0}
        assert await first_rotate_frontier_tail_best_mask_count(depth2_state) >= 1.0
        assert await first_rotate_counterfactual_frontier(depth2_state) == 1.0
        assert 0.80 <= await action_gated_counterfactual_frontier_strict_reward(depth2_state) <= 0.95

        protocol_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_counterfactual_frontier_strict",
            prompt_style="stage_frontier_sensor_native_tool",
            max_turns=2,
            move_budget=1,
        )
        protocol_state = {"task": dict(protocol_env.get_dataset()[0])}
        await protocol_env.setup_state(protocol_state)
        protocol_rollout = protocol_state["megaminx"]
        face, direction = protocol_rollout.inverse_solution[0]
        await protocol_env.env_response(
            [
                AssistantMessage(
                    content="visible text",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "rotate",
                            "arguments": json.dumps({"face": face, "direction": direction}),
                        }
                    ],
                )
            ],
            protocol_state,
        )
        assert protocol_rollout.solved()
        assert await action_gated_counterfactual_frontier_strict_reward(protocol_state) == 0.0

    asyncio.run(run())


def test_stage_frontier_sensor_native_tool_refreshes_raw_view_after_first_move() -> None:
    async def run() -> None:
        env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=1,
            seed=81,
            reward_style="action_gated_counterfactual_frontier_strict",
            prompt_style="stage_frontier_sensor_native_tool",
            max_turns=4,
            move_budget=2,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        response = await env.rotate(face, direction, rollout)
        assert "Updated frontier sensor view:" in response
        assert "Candidate-local moved strips:" in response
        assert "Solve-action equality-mask table:" not in response
        second_face, second_direction = rollout.inverse_solution[1]
        await env.rotate(second_face, second_direction, rollout)
        assert rollout.solved()

    asyncio.run(run())


def test_stage_frontier_sensor_compact_native_tool_omits_full_net_and_masks() -> None:
    env = load_environment(
        min_depth=2,
        max_depth=2,
        num_examples=1,
        seed=52,
        reward_style="action_gated_counterfactual_frontier_value_strict",
        prompt_style="stage_frontier_sensor_compact_native_tool",
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])

    assert env.env_args["allow_text_tool_actions"] is False
    assert "Frontier sensor objective:" in prompt
    assert "Decision rule: for candidate X with ring n0 n1 n2 n3 n4" in prompt
    assert "Choose an action with the most supported ring positions." in prompt
    assert "For this frontier task, choose exactly one rotate call; do not inspect or finish." in prompt
    assert "Structured state sensors:" in prompt
    assert "Candidate-local moved strips:" in prompt
    assert "Initial compact observation: use the sensor sections above." in prompt
    assert "Initial observation:" not in prompt
    assert "Facelet net:" not in prompt
    assert "Solve-action equality-mask table:" not in prompt
    assert "cw_mask" not in prompt
    assert "ccw_mask" not in prompt
    assert "expected_current_if_solve" not in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt


def test_action_gated_counterfactual_frontier_value_reward_is_dense_and_strict() -> None:
    async def run() -> None:
        env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=1,
            seed=77,
            reward_style="action_gated_counterfactual_frontier_value_strict",
            prompt_style="stage_frontier_sensor_compact_native_tool",
            max_turns=2,
            move_budget=1,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "rotate",
                            "arguments": json.dumps({"face": face, "direction": direction}),
                        }
                    ],
                )
            ],
            state,
        )

        assert await reward_style(state) == 11.0
        assert await first_rotate_counterfactual_frontier(state) == 1.0
        assert await first_rotate_counterfactual_value(state) == 1.0
        assert await first_rotate_counterfactual_value_norm(state) >= 0.0
        assert await first_rotate_counterfactual_value_rank(state) > 0.0
        assert 0.80 <= await action_gated_counterfactual_frontier_value_strict_reward(state) <= 0.95

        wrong_env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=1,
            seed=77,
            reward_style="action_gated_counterfactual_frontier_value_strict",
            prompt_style="stage_frontier_sensor_compact_native_tool",
            max_turns=2,
            move_budget=1,
        )
        wrong_state = {"task": dict(wrong_env.get_dataset()[0])}
        await wrong_env.setup_state(wrong_state)
        wrong_rollout = wrong_state["megaminx"]
        wrong_face = next(face for face in FACES if face != wrong_rollout.inverse_solution[0][0])
        await wrong_env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "rotate",
                            "arguments": json.dumps({"face": wrong_face, "direction": "cw"}),
                        }
                    ],
                )
            ],
            wrong_state,
        )
        dense_reward = await action_gated_counterfactual_frontier_value_strict_reward(
            wrong_state
        )
        assert 0.0 < dense_reward <= 0.62
        assert await first_rotate_counterfactual_value_norm(wrong_state) >= 0.0

        protocol_env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=1,
            seed=77,
            reward_style="action_gated_counterfactual_frontier_value_strict",
            prompt_style="stage_frontier_sensor_compact_native_tool",
            max_turns=2,
            move_budget=1,
        )
        protocol_state = {"task": dict(protocol_env.get_dataset()[0])}
        await protocol_env.setup_state(protocol_state)
        protocol_rollout = protocol_state["megaminx"]
        face, direction = protocol_rollout.inverse_solution[0]
        await protocol_env.env_response(
            [
                AssistantMessage(
                    content="visible text",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "rotate",
                            "arguments": json.dumps({"face": face, "direction": direction}),
                        }
                    ],
                )
            ],
            protocol_state,
        )
        assert await action_gated_counterfactual_frontier_value_strict_reward(protocol_state) == 0.0

    asyncio.run(run())


def _expected_touching_strips_after(puzzle: MegaminxPuzzle, face: str, direction: str) -> dict[str, str]:
    after = puzzle.copy()
    after.apply_move(face, direction)
    return {
        neighbor: "".join(
            after.stickers[position]
            for position in DEFAULT_TOPOLOGY.side_strip(face, neighbor)
        )
        for neighbor in DEFAULT_TOPOLOGY.neighbor_rings[face]
    }


def test_stage_predict_rotate_native_tool_prompt_and_reward() -> None:
    async def run() -> None:
        env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=1,
            seed=77,
            reward_style="action_gated_predict_rotate_value_strict",
            prompt_style="stage_predict_rotate_native_tool",
            max_turns=2,
            move_budget=1,
        )
        row = dict(env.get_dataset()[0])
        prompt = "\n".join(message["content"] for message in row["prompt"])
        assert "Predict-then-rotate objective:" in prompt
        assert "predict_rotate(face, direction, predicted_after)" in prompt
        assert "Action-first instruction: the first assistant turn must be a predict_rotate tool call." in prompt
        assert "Initial compact observation: use the sensor sections above." in prompt
        assert "predicted_after must be five three-letter strips" in prompt
        assert "Do not use rotate, inspect, or finish in this prompt style." in prompt
        assert "Initial observation:" not in prompt
        assert "Solve-action equality-mask table:" not in prompt
        assert row["scramble"] not in prompt
        assert row["inverse_solution"] not in prompt

        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        expected = _expected_touching_strips_after(rollout.initial_puzzle, face, direction)
        expected_list = [
            expected[neighbor] for neighbor in DEFAULT_TOPOLOGY.neighbor_rings[face]
        ]
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "predict_rotate",
                            "arguments": json.dumps(
                                    {
                                        "face": face,
                                        "direction": direction,
                                        "predicted_after": expected_list,
                                    }
                            ),
                        }
                    ],
                )
            ],
            state,
        )
        assert await reward_style(state) == 12.0
        assert await predict_rotate_call_count(state) == 1.0
        assert await rotate_call_count(state) == 0.0
        assert await first_prediction_valid(state) == 1.0
        assert await first_prediction_exact_strip_count(state) == 5.0
        assert await first_prediction_strip_accuracy(state) == 1.0
        assert await first_prediction_char_accuracy(state) == 1.0
        assert await first_prediction_quality(state) == 1.0
        assert await first_rotate_counterfactual_frontier(state) == 1.0
        assert 0.70 <= await action_gated_predict_rotate_value_strict_reward(state) <= 1.0

        bad_state = {"task": row}
        await env.setup_state(bad_state)
        bad_rollout = bad_state["megaminx"]
        await env.env_response(
            [
                AssistantMessage(
                    content="visible text",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "predict_rotate",
                            "arguments": json.dumps(
                                    {
                                        "face": face,
                                        "direction": direction,
                                        "predicted_after": expected_list,
                                    }
                            ),
                        }
                    ],
                )
            ],
            bad_state,
        )
        assert bad_rollout.first_rotate == (face, direction)
        assert await action_gated_predict_rotate_value_strict_reward(bad_state) == 0.0

    asyncio.run(run())


def test_predict_rotate_rejects_malformed_predictions_without_rotating() -> None:
    async def run() -> None:
        env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=1,
            seed=78,
            reward_style="action_gated_predict_rotate_value_strict",
            prompt_style="stage_predict_rotate_native_tool",
            max_turns=2,
            move_budget=1,
        )
        row = dict(env.get_dataset()[0])

        malformed_cases = [
            [],
            ["AAA"],
            ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            ["AA", "BBB", "CCC", "DDD", "EEE"],
            ["AAA", "BBB", "CCC", "DDD", "ZZZ"],
            ["AAA", "BBB", "CCC", "DDD", 3],
            "AAA BBB CCC DDD EEE",
        ]
        for predicted_after in malformed_cases:
            state = {"task": row}
            await env.setup_state(state)
            rollout = state["megaminx"]
            face, direction = rollout.inverse_solution[0]
            response = await env.env_response(
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "native-call",
                                "name": "predict_rotate",
                                "arguments": json.dumps(
                                    {
                                        "face": face,
                                        "direction": direction,
                                        "predicted_after": predicted_after,
                                    }
                                ),
                            }
                        ],
                    )
                ],
                state,
            )
            assert response[0].role == "tool"
            assert "Illegal prediction" in response[0].content
            assert rollout.illegal_moves == 1
            assert rollout.move_count == 0
            assert rollout.first_rotate is None
            assert rollout.first_prediction is None
            assert await first_prediction_valid(state) == 0.0
            assert await action_gated_predict_rotate_value_strict_reward(state) == 0.0

        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        wrong_but_valid = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "predict_rotate",
                            "arguments": json.dumps(
                                {
                                    "face": face,
                                    "direction": direction,
                                    "predicted_after": wrong_but_valid,
                                }
                            ),
                        }
                    ],
                )
            ],
            state,
        )
        assert rollout.move_count == 1
        assert await first_prediction_valid(state) == 1.0
        assert await first_prediction_item_count(state) == 5.0
        assert await first_prediction_extra_count(state) == 0.0
        assert await action_gated_predict_rotate_value_strict_reward(state) < 0.80

    asyncio.run(run())


def test_stage_predict_transition_native_tool_rewards_transition_accuracy() -> None:
    async def run() -> None:
        env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=1,
            seed=79,
            reward_style="action_gated_predict_rotate_transition",
            prompt_style="stage_predict_transition_native_tool",
            max_turns=2,
            move_budget=1,
        )
        row = dict(env.get_dataset()[0])
        prompt = "\n".join(message["content"] for message in row["prompt"])
        assert "Transition-prediction objective:" in prompt
        assert "physics transition curriculum" in prompt
        assert "hidden frontier progress" not in prompt
        assert "Goal: choose one legal face turn and predict" in prompt
        assert "Initial observation:" not in prompt
        assert row["scramble"] not in prompt
        assert row["inverse_solution"] not in prompt

        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.scramble[0]
        expected = _expected_touching_strips_after(rollout.initial_puzzle, face, direction)
        expected_list = [
            expected[neighbor] for neighbor in DEFAULT_TOPOLOGY.neighbor_rings[face]
        ]
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "predict_rotate",
                            "arguments": json.dumps(
                                {
                                    "face": face,
                                    "direction": direction,
                                    "predicted_after": expected_list,
                                }
                            ),
                        }
                    ],
                )
            ],
            state,
        )
        assert await reward_style(state) == 13.0
        assert await first_prediction_quality(state) == 1.0
        assert await action_gated_predict_rotate_transition_reward(state) == 1.0

        wrong_state = {"task": row}
        await env.setup_state(wrong_state)
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "predict_rotate",
                            "arguments": json.dumps(
                                {
                                    "face": face,
                                    "direction": direction,
                                    "predicted_after": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                                }
                            ),
                        }
                    ],
                )
            ],
            wrong_state,
        )
        assert 0.0 <= await action_gated_predict_rotate_transition_reward(wrong_state) < 1.0

    asyncio.run(run())


def test_stage_face_discovery_native_tool_rewards_face_selection() -> None:
    async def run() -> None:
        env = load_environment(
            min_depth=1,
            max_depth=1,
            num_examples=1,
            seed=80,
            reward_style="action_gated_face_discovery",
            prompt_style="stage_face_discovery_native_tool",
            max_turns=2,
            move_budget=1,
        )
        row = dict(env.get_dataset()[0])
        prompt = "\n".join(message["content"] for message in row["prompt"])
        assert "Face-discovery objective:" in prompt
        assert "Goal: identify the face whose rotation" in prompt
        assert "Action-first instruction: the first assistant turn must be a rotate tool call." in prompt
        assert "Initial observation:" not in prompt
        assert "Solve-action equality-mask table:" not in prompt
        assert row["scramble"] not in prompt
        assert row["inverse_solution"] not in prompt

        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, correct_direction = rollout.inverse_solution[0]
        wrong_direction = "ccw" if correct_direction == "cw" else "cw"
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "rotate",
                            "arguments": json.dumps(
                                {"face": face, "direction": wrong_direction}
                            ),
                        }
                    ],
                )
            ],
            state,
        )
        assert await reward_style(state) == 14.0
        assert await first_rotate_face_correct(state) == 1.0
        assert await first_rotate_direction_correct(state) == 0.0
        assert await action_gated_face_discovery_reward(state) == 1.0

        wrong_state = {"task": row}
        await env.setup_state(wrong_state)
        wrong_face = next(candidate for candidate in FACES if candidate != face)
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "rotate",
                            "arguments": json.dumps(
                                {"face": wrong_face, "direction": correct_direction}
                            ),
                        }
                    ],
                )
            ],
            wrong_state,
        )
        assert await action_gated_face_discovery_reward(wrong_state) < 1.0

    asyncio.run(run())


def test_stage_face_tournament_native_tool_constrains_candidate_set() -> None:
    async def run() -> None:
        env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=1,
            seed=81,
            reward_style="action_gated_face_tournament",
            prompt_style="stage_face_tournament_native_tool",
            max_turns=2,
            move_budget=1,
        )
        row = dict(env.get_dataset()[0])
        prompt = "\n".join(message["content"] for message in row["prompt"])
        candidate_faces = row["candidate_faces"]
        assert len(candidate_faces) == 4
        assert len(set(candidate_faces)) == 4
        target_face = json.loads(row["inverse_solution"])[0]["face"]
        assert target_face in candidate_faces
        assert "Candidate faces: " + " ".join(candidate_faces) in prompt
        assert "other faces receive zero reward" in prompt
        assert "Initial observation:" not in prompt
        assert row["scramble"] not in prompt
        assert row["inverse_solution"] not in prompt

        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        target_face, target_direction = rollout.inverse_solution[0]
        wrong_direction = "ccw" if target_direction == "cw" else "cw"
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "rotate",
                            "arguments": json.dumps(
                                {"face": target_face, "direction": wrong_direction}
                            ),
                        }
                    ],
                )
            ],
            state,
        )
        assert await reward_style(state) == 15.0
        assert await first_rotate_in_candidate_set(state) == 1.0
        assert await first_rotate_face_correct(state) == 1.0
        assert await action_gated_face_tournament_reward(state) == 1.0

        outside_state = {"task": row}
        await env.setup_state(outside_state)
        outside_face = next(face for face in FACES if face not in candidate_faces)
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "rotate",
                            "arguments": json.dumps(
                                {"face": outside_face, "direction": target_direction}
                            ),
                        }
                    ],
                )
            ],
            outside_state,
        )
        assert await first_rotate_in_candidate_set(outside_state) == 0.0
        assert await action_gated_face_tournament_reward(outside_state) == 0.0

    asyncio.run(run())


def test_stage_candidate_tournament_native_tool_selects_by_slot() -> None:
    async def run() -> None:
        env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=1,
            seed=82,
            reward_style="action_gated_candidate_tournament",
            prompt_style="stage_candidate_tournament_native_tool",
            max_turns=2,
            move_budget=1,
        )
        row = dict(env.get_dataset()[0])
        prompt = "\n".join(message["content"] for message in row["prompt"])
        candidate_faces = row["candidate_faces"]
        assert len(candidate_faces) == 4
        assert len(set(candidate_faces)) == 4
        target_face = json.loads(row["inverse_solution"])[0]["face"]
        target_direction = json.loads(row["inverse_solution"])[0]["direction"]
        target_index = candidate_faces.index(target_face) + 1
        assert target_face in candidate_faces
        assert "select_candidate" in prompt
        assert "Legal action: select one printed candidate slot 1-4" in prompt
        assert "Legal moves: rotate any face" not in prompt
        assert "Candidate faces:" not in prompt
        for index, face in enumerate(candidate_faces, start=1):
            assert f"Candidate {index}: {face}" in prompt
        assert "Call select_candidate exactly once" in prompt
        assert "Call rotate exactly once" not in prompt
        assert "Initial observation:" not in prompt
        assert row["scramble"] not in prompt
        assert row["inverse_solution"] not in prompt
        assert [tool.name for tool in env.tool_defs] == ["select_candidate"]
        assert "select_candidate tool call" in env.system_prompt
        assert "Do not call rotate" in env.system_prompt

        state = {"task": row}
        await env.setup_state(state)
        wrong_direction = "ccw" if target_direction == "cw" else "cw"
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate",
                            "arguments": json.dumps(
                                {"index": target_index, "direction": wrong_direction}
                            ),
                        }
                    ],
                )
            ],
            state,
        )
        assert await reward_style(state) == 16.0
        assert await candidate_select_call_count(state) == 1.0
        assert await rotate_call_count(state) == 0.0
        assert await first_candidate_index(state) == float(target_index)
        assert await first_rotate_in_candidate_set(state) == 1.0
        assert await first_rotate_face_correct(state) == 1.0
        assert await action_gated_candidate_tournament_reward(state) == 1.0

        outside_state = {"task": row}
        await env.setup_state(outside_state)
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate",
                            "arguments": json.dumps(
                                {"index": 5, "direction": target_direction}
                            ),
                        }
                    ],
                )
            ],
            outside_state,
        )
        assert await candidate_select_call_count(outside_state) == 1.0
        assert await first_candidate_index(outside_state) == -1.0
        assert await action_gated_candidate_tournament_reward(outside_state) == 0.0

        wrong_state = {"task": row}
        await env.setup_state(wrong_state)
        wrong_index = next(
            index
            for index, face in enumerate(candidate_faces, start=1)
            if face != target_face
        )
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate",
                            "arguments": json.dumps(
                                {"index": wrong_index, "direction": target_direction}
                            ),
                        }
                    ],
                )
            ],
            wrong_state,
        )
        assert await first_rotate_face_correct(wrong_state) == 0.0
        assert 0.0 <= await action_gated_candidate_tournament_reward(wrong_state) < 1.0

    asyncio.run(run())


def test_stage_candidate_index_native_tool_selects_by_slot_without_direction() -> None:
    async def run() -> None:
        env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=1,
            seed=85,
            reward_style="action_gated_candidate_index",
            prompt_style="stage_candidate_index_native_tool",
            max_turns=2,
            move_budget=1,
        )
        row = dict(env.get_dataset()[0])
        prompt = "\n".join(message["content"] for message in row["prompt"])
        candidate_faces = row["candidate_faces"]
        assert len(candidate_faces) == 4
        assert len(set(candidate_faces)) == 4
        target_face = json.loads(row["inverse_solution"])[0]["face"]
        target_index = candidate_faces.index(target_face) + 1
        assert "select_candidate_index" in prompt
        assert "select_candidate(index, direction)" not in prompt
        assert "direction" not in prompt
        assert "cw" not in prompt
        assert "ccw" not in prompt
        assert "rotate" not in prompt
        assert "Legal action: select one printed candidate slot 1-4." in prompt
        assert "Candidate faces:" not in prompt
        for index, face in enumerate(candidate_faces, start=1):
            assert f"Candidate {index}: {face}" in prompt
        assert "Call select_candidate_index exactly once" in prompt
        assert "Initial observation:" not in prompt
        assert [tool.name for tool in env.tool_defs] == ["select_candidate_index"]
        assert "select_candidate_index tool call" in env.system_prompt
        assert "rotate" not in env.system_prompt

        state = {"task": row}
        await env.setup_state(state)
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate_index",
                            "arguments": json.dumps({"index": target_index}),
                        }
                    ],
                )
            ],
            state,
        )
        assert await reward_style(state) == 17.0
        assert await candidate_select_call_count(state) == 1.0
        assert await action_taken(state) == 0.0
        assert await first_candidate_index(state) == float(target_index)
        assert await first_rotate_in_candidate_set(state) == 0.0
        assert await first_rotate_face_correct(state) == 0.0
        assert await first_candidate_face_correct(state) == 1.0
        assert await action_gated_candidate_index_reward(state) == 1.0

        invalid_state = {"task": row}
        await env.setup_state(invalid_state)
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate_index",
                            "arguments": json.dumps({"index": 5}),
                        }
                    ],
                )
            ],
            invalid_state,
        )
        assert await candidate_select_call_count(invalid_state) == 1.0
        assert await first_candidate_index(invalid_state) == -1.0
        assert await action_gated_candidate_index_reward(invalid_state) == 0.0

        wrong_state = {"task": row}
        await env.setup_state(wrong_state)
        wrong_index = next(
            index
            for index, face in enumerate(candidate_faces, start=1)
            if face != target_face
        )
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate_index",
                            "arguments": json.dumps({"index": wrong_index}),
                        }
                    ],
                )
            ],
            wrong_state,
        )
        assert await first_candidate_face_correct(wrong_state) == 0.0
        assert await action_gated_candidate_index_reward(wrong_state) == 0.0

    asyncio.run(run())


def test_stage_candidate_tournament_depth1_prompt_has_no_rotate_hint() -> None:
    env = load_environment(
        split="depth1",
        num_examples=1,
        seed=83,
        reward_style="action_gated_candidate_tournament",
        prompt_style="stage_candidate_tournament_native_tool",
        max_turns=2,
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])

    assert "select_candidate call" in prompt
    assert "rotate action" not in prompt
    assert "Legal moves: rotate any face" not in prompt
    assert "For this candidate-tournament task" in prompt


def test_stage_candidate_scorecard_native_tool_lists_slot_features_without_answer() -> None:
    env = load_environment(
        min_depth=2,
        max_depth=2,
        num_examples=1,
        seed=86,
        reward_style="action_gated_candidate_tournament",
        prompt_style="stage_candidate_scorecard_native_tool",
        max_turns=2,
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    candidate_faces = row["candidate_faces"]

    assert [tool.name for tool in env.tool_defs] == ["select_candidate"]
    assert "Candidate scorecard:" in prompt
    assert "slot | face | support | affected_neighbors | frontier | cw_mask | ccw_mask" in prompt
    for slot, face in enumerate(candidate_faces, start=1):
        assert f"{slot} | {face} | " in prompt
        assert f"Candidate {slot}: {face}" in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt
    assert row["answer"] not in prompt
    forbidden = ("winner", "correct_face", "best_face", "answer_face")
    assert all(word not in prompt.lower() for word in forbidden)


def test_stage_candidate_scorecard_no_frontier_omits_frontier_feature() -> None:
    env = load_environment(
        min_depth=2,
        max_depth=2,
        num_examples=1,
        seed=87,
        reward_style="action_gated_candidate_tournament",
        prompt_style="stage_candidate_scorecard_no_frontier_native_tool",
        max_turns=2,
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    candidate_faces = row["candidate_faces"]

    assert [tool.name for tool in env.tool_defs] == ["select_candidate"]
    assert "Candidate scorecard:" in prompt
    assert "slot | face | support | affected_neighbors | cw_mask | ccw_mask" in prompt
    assert "frontier" not in prompt.lower()
    for slot, face in enumerate(candidate_faces, start=1):
        assert f"{slot} | {face} | " in prompt
        assert f"Candidate {slot}: {face}" in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt


def test_stage_candidate_scorecard_mask_omits_support_and_frontier_features() -> None:
    env = load_environment(
        min_depth=2,
        max_depth=2,
        num_examples=1,
        seed=88,
        reward_style="action_gated_candidate_tournament",
        prompt_style="stage_candidate_scorecard_mask_native_tool",
        max_turns=2,
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    candidate_faces = row["candidate_faces"]

    assert [tool.name for tool in env.tool_defs] == ["select_candidate"]
    assert "Candidate scorecard:" in prompt
    assert "slot | face | affected_neighbors | cw_mask | ccw_mask" in prompt
    assert "support" not in prompt.lower()
    assert "frontier" not in prompt.lower()
    for slot, face in enumerate(candidate_faces, start=1):
        assert f"{slot} | {face} | " in prompt
        assert f"Candidate {slot}: {face}" in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt


def test_stage_candidate_scorecard_mask_index_native_tool_uses_same_masks_with_index_tool() -> None:
    env = load_environment(
        min_depth=2,
        max_depth=2,
        num_examples=1,
        seed=89,
        reward_style="action_gated_candidate_mask_index_rank",
        prompt_style="stage_candidate_scorecard_mask_index_native_tool",
        max_turns=2,
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    candidate_faces = row["candidate_faces"]

    assert [tool.name for tool in env.tool_defs] == ["select_candidate_index"]
    assert "select_candidate_index tool call" in env.system_prompt
    assert "select_candidate tool call" not in env.system_prompt
    assert "Candidate scorecard:" in prompt
    assert "slot | face | affected_neighbors | cw_mask | ccw_mask" in prompt
    assert "support" not in prompt.lower()
    assert "frontier" not in prompt.lower()
    assert "Legal action: select one printed candidate slot 1-4." in prompt
    assert "Use select_candidate_index(index)." in prompt
    assert "Use select_candidate(index, direction)" not in prompt
    assert "Initial compact observation: use the sensor sections above." in prompt
    assert "Initial observation:" not in prompt
    for slot, face in enumerate(candidate_faces, start=1):
        assert f"{slot} | {face} | " in prompt
        assert f"Candidate {slot}: {face}" in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt
    assert row["answer"] not in prompt


def test_stage_candidate_scorecard_mask_frontier_equivalence_uses_same_public_columns() -> None:
    env = load_environment(
        min_depth=2,
        max_depth=2,
        num_examples=1,
        seed=90,
        reward_style="action_gated_candidate_mask_frontier_equivalence",
        prompt_style="stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
        max_turns=2,
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    candidate_faces = row["candidate_faces"]

    assert [tool.name for tool in env.tool_defs] == ["select_candidate"]
    assert "select_candidate tool call" in env.system_prompt
    assert "Candidate scorecard:" in prompt
    assert "slot | face | affected_neighbors | cw_mask | ccw_mask" in prompt
    assert "support" not in prompt.lower()
    assert "frontier" not in prompt.lower()
    assert "leaves a one-turn solve" in prompt
    assert "Use select_candidate(index, direction)." in prompt
    assert "Initial compact observation: use the sensor sections above." in prompt
    for slot, face in enumerate(candidate_faces, start=1):
        assert f"{slot} | {face} | " in prompt
        assert f"Candidate {slot}: {face}" in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt
    assert row["answer"] not in prompt


def test_stage_candidate_geometry_frontier_hides_public_reward_proxy() -> None:
    env = load_environment(
        min_depth=2,
        max_depth=2,
        num_examples=1,
        seed=91,
        reward_style="action_gated_candidate_geometry_frontier",
        prompt_style="stage_candidate_geometry_frontier_native_tool",
        max_turns=2,
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    candidate_faces = row["candidate_faces"]
    puzzle = MegaminxPuzzle.solved(DEFAULT_TOPOLOGY)
    puzzle.apply_moves(
        [(item["face"], item["direction"]) for item in json.loads(row["scramble"])]
    )
    affected = {
        face
        for face in FACES
        if any(puzzle.stickers[(face, position)] != face for position in range(1, 11))
    }
    mask_counts = _action_mask_counts(puzzle)
    expected_top_faces = sorted(
        FACES,
        key=lambda face: (
            -len(set(DEFAULT_TOPOLOGY.neighbor_rings[face]) & affected),
            -sum(
                puzzle.stickers[position] != neighbor
                for neighbor in DEFAULT_TOPOLOGY.neighbor_rings[face]
                for side in [DEFAULT_TOPOLOGY.neighbor_rings[neighbor].index(face)]
                for position in (
                    (neighbor, 1 + ((side - 1) % 5)),
                    (neighbor, 6 + side),
                    (neighbor, 1 + side),
                )
            ),
            -max(mask_counts[(face, direction)] for direction in DIRECTIONS),
            FACES.index(face),
        ),
    )[:4]

    assert [tool.name for tool in env.tool_defs] == ["select_candidate"]
    assert "select_candidate tool call" in env.system_prompt
    assert "Candidate scorecard:" not in prompt
    assert "cw_mask" not in prompt
    assert "ccw_mask" not in prompt
    assert "support" not in prompt.lower()
    assert "frontier" not in prompt.lower()
    assert "leaves a one-turn solve" in prompt
    assert "Candidate-local moved strips:" in prompt
    assert "Initial compact observation: use the sensor sections above." in prompt
    assert set(candidate_faces) == set(expected_top_faces)
    for slot, face in enumerate(candidate_faces, start=1):
        assert f"Candidate {slot}: {face}" in prompt
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt
    assert row["answer"] not in prompt


def test_stage_candidate_relative_flow_frontier_uses_visible_coordinate_transform() -> None:
    env = load_environment(
        min_depth=1,
        max_depth=2,
        num_examples=1,
        seed=94,
        reward_style="action_gated_candidate_geometry_frontier",
        prompt_style="stage_candidate_relative_flow_frontier_native_tool",
        max_turns=2,
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    candidate_faces = row["candidate_faces"]
    section = prompt.split("Candidate relative-flow table:", 1)[1].split(
        "Initial compact observation:", 1
    )[0]

    assert [tool.name for tool in env.tool_defs] == ["select_candidate"]
    assert "select_candidate tool call" in env.system_prompt
    assert "Candidate relative-flow table:" in prompt
    assert "Candidate-local moved strips:" not in prompt
    assert "Candidate scorecard:" not in prompt
    assert "cw_mask" not in prompt
    assert "ccw_mask" not in prompt
    for forbidden in ("answer", "winner", "best", "score", "mask", "support", "frontier", "target"):
        assert forbidden not in section.lower()
    for slot, face in enumerate(candidate_faces, start=1):
        assert f"Candidate {slot}: {face}" in prompt
        assert f"{slot} | {face} |" in section
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt
    assert row["answer"] not in prompt


def test_stage_candidate_relative_flow_rule_prompt_is_fair_counting_instruction() -> None:
    env = load_environment(
        min_depth=1,
        max_depth=2,
        num_examples=1,
        seed=97,
        reward_style="action_gated_candidate_strict_frontier",
        prompt_style="stage_candidate_relative_flow_rule_frontier_native_tool",
        max_turns=2,
        move_budget=1,
    )
    row = dict(env.get_dataset()[0])
    prompt = "\n".join(message["content"] for message in row["prompt"])
    section = prompt.split("Relative-flow counting rule:", 1)[1].split(
        "Initial compact observation:", 1
    )[0]

    assert [tool.name for tool in env.tool_defs] == ["select_candidate"]
    assert "Relative-flow counting rule:" in prompt
    assert "count how many tokens have delta +1" in section
    assert "largest +1 count" in section
    assert "Candidate relative-flow table:" in section
    assert "Candidate-local moved strips:" not in prompt
    assert "Candidate scorecard:" not in prompt
    for forbidden in ("answer", "winner", "best", "score", "mask", "support", "frontier", "target"):
        assert forbidden not in section.lower()
    assert row["scramble"] not in prompt
    assert row["inverse_solution"] not in prompt
    assert row["answer"] not in prompt


def test_candidate_relative_flow_oracle_unique_on_depth1_moves() -> None:
    async def run() -> None:
        for prompt_style in (
            "stage_candidate_relative_flow_frontier_native_tool",
            "stage_candidate_relative_flow_rule_frontier_native_tool",
        ):
            env = load_environment(
                split="depth1",
                num_examples=24,
                seed=95,
                reward_style="action_gated_candidate_geometry_frontier",
                prompt_style=prompt_style,
                max_turns=2,
                move_budget=1,
            )
            target_indices: Counter[int] = Counter()
            for raw_row in env.get_dataset():
                row = dict(raw_row)
                state = {"task": row}
                await env.setup_state(state)
                rollout = state["megaminx"]
                target_face, target_direction = rollout.inverse_solution[0]
                target_index = row["candidate_faces"].index(target_face) + 1
                target_indices[target_index] += 1

                assert await target_face_in_candidate_set(state) == 1.0
                assert await target_candidate_index(state) == float(target_index)
                assert await candidate_relative_flow_oracle_unique(state) == 1.0

                await env.env_response(
                    [
                        AssistantMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "native-call",
                                    "name": "select_candidate",
                                    "arguments": json.dumps(
                                        {"index": target_index, "direction": target_direction}
                                    ),
                                }
                            ],
                        )
                    ],
                    state,
                )
                assert rollout.solved()
                assert await first_candidate_relative_flow_count(state) == 5.0
                assert await first_candidate_relative_flow_margin(state) == 5.0
                assert await first_candidate_relative_flow_is_candidate_max(state) == 1.0
                assert await action_gated_candidate_geometry_frontier_reward(state) == 1.0

            assert set(target_indices) == {1, 2, 3, 4}

    asyncio.run(run())


def test_action_gated_candidate_strict_frontier_limits_constant_policy_shortcut() -> None:
    async def run() -> None:
        env = load_environment(
            split="train_candidate_relative_flow_rule_frontier_depth12",
            min_depth=1,
            max_depth=2,
            num_examples=96,
            seed=96,
            reward_style="action_gated_candidate_strict_frontier",
            prompt_style="stage_candidate_relative_flow_rule_frontier_native_tool",
            max_turns=2,
            move_budget=1,
            allow_text_tool_actions=False,
        )
        constant_rewards: dict[tuple[int, str], list[float]] = {
            (slot, direction): [] for slot in range(1, 5) for direction in DIRECTIONS
        }
        oracle_rewards = []
        for raw_row in env.get_dataset():
            row = dict(raw_row)
            for slot, direction in constant_rewards:
                constant_state = {"task": row}
                await env.setup_state(constant_state)
                await env.env_response(
                    [
                        AssistantMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "constant-call",
                                    "name": "select_candidate",
                                    "arguments": json.dumps(
                                        {"index": slot, "direction": direction}
                                    ),
                                }
                            ],
                        )
                    ],
                    constant_state,
                )
                constant_rewards[(slot, direction)].append(
                    await action_gated_candidate_strict_frontier_reward(constant_state)
                )

            oracle_state = {"task": row}
            await env.setup_state(oracle_state)
            rollout = oracle_state["megaminx"]
            target_face, target_direction = rollout.inverse_solution[0]
            target_index = row["candidate_faces"].index(target_face) + 1
            await env.env_response(
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "oracle-call",
                                "name": "select_candidate",
                                "arguments": json.dumps(
                                    {"index": target_index, "direction": target_direction}
                                ),
                            }
                        ],
                    )
                ],
                oracle_state,
            )
            oracle_rewards.append(await action_gated_candidate_strict_frontier_reward(oracle_state))

        constant_means = [
            sum(rewards) / len(rewards) for rewards in constant_rewards.values()
        ]
        assert max(constant_means) < 0.20
        assert sum(oracle_rewards) / len(oracle_rewards) > 0.85
        assert await reward_style(oracle_state) == 21.0

    asyncio.run(run())


def test_printed_relative_flow_rule_policy_beats_constant_shortcuts() -> None:
    async def run() -> None:
        env = load_environment(
            split="train_candidate_relative_flow_rule_frontier_depth12",
            min_depth=1,
            max_depth=2,
            num_examples=96,
            seed=96,
            reward_style="action_gated_candidate_strict_frontier",
            prompt_style="stage_candidate_relative_flow_rule_frontier_native_tool",
            max_turns=2,
            move_budget=1,
            allow_text_tool_actions=False,
        )
        rewards = []
        unique_rewards = []
        for raw_row in env.get_dataset():
            row = dict(raw_row)
            prompt = "\n".join(message["content"] for message in row["prompt"])
            slot, direction = relative_flow_rule_action_from_prompt(prompt)

            state = {"task": row}
            await env.setup_state(state)
            unique = await candidate_relative_flow_oracle_unique(state) == 1.0
            await env.env_response(
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "rule-call",
                                "name": "select_candidate",
                                "arguments": json.dumps(
                                    {"index": slot, "direction": direction}
                                ),
                            }
                        ],
                    )
                ],
                state,
            )
            reward = await action_gated_candidate_strict_frontier_reward(state)
            rewards.append(reward)
            if unique:
                unique_rewards.append(reward)

        assert len(unique_rewards) > len(rewards) * 0.6
        assert sum(rewards) / len(rewards) > 0.80
        assert min(unique_rewards) >= 0.75
        assert sum(unique_rewards) / len(unique_rewards) > 0.90

    asyncio.run(run())


def test_candidate_relative_flow_rule_solve2_refreshes_candidates_and_solves_depth2() -> None:
    async def run() -> None:
        env = load_environment(
            split="train_candidate_relative_flow_rule_solve2_depth2",
            min_depth=2,
            max_depth=2,
            num_examples=24,
            seed=104,
            reward_style="action_gated_candidate_path_solve",
            prompt_style="stage_candidate_relative_flow_rule_solve2_native_tool",
            max_turns=4,
            move_budget=2,
            allow_text_tool_actions=False,
        )
        row = dict(env.get_dataset()[0])
        prompt = "\n".join(message["content"] for message in row["prompt"])
        assert [tool.name for tool in env.tool_defs] == ["select_candidate"]
        assert "candidate-path task" in env.system_prompt
        assert "refreshed candidate table" in prompt
        assert "Call select_candidate exactly once" not in prompt
        assert row["scramble"] not in prompt
        assert row["inverse_solution"] not in prompt
        assert row["answer"] not in prompt

        for raw_row in env.get_dataset():
            row = dict(raw_row)
            state = {"task": row}
            await env.setup_state(state)
            rollout = state["megaminx"]
            first_face, first_direction = rollout.inverse_solution[0]
            first_index = row["candidate_faces"].index(first_face) + 1

            first_response = await env.env_response(
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "first-call",
                                "name": "select_candidate",
                                "arguments": json.dumps(
                                    {"index": first_index, "direction": first_direction}
                                ),
                            }
                        ],
                    )
                ],
                state,
            )
            assert "Updated candidate relative-flow view" in first_response[0].content
            assert await inverse_prefix_length(state) == 1.0
            assert await action_reaches_one_turn_frontier(state) == 1.0
            assert await action_gated_candidate_path_solve_reward(state) == 0.70

            second_face, second_direction = rollout.inverse_solution[1]
            assert second_face in rollout.candidate_faces
            second_index = rollout.candidate_faces.index(second_face) + 1
            await env.env_response(
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "second-call",
                                "name": "select_candidate",
                                "arguments": json.dumps(
                                    {"index": second_index, "direction": second_direction}
                                ),
                            }
                        ],
                    )
                ],
                state,
            )
            assert rollout.solved()
            assert rollout.move_history == rollout.inverse_solution
            assert await second_rotate_correct(state) == 1.0
            assert await second_rotate_face_correct(state) == 1.0
            assert await second_rotate_direction_correct(state) == 1.0
            assert await inverse_prefix_length(state) == 2.0
            assert await action_gated_candidate_path_solve_reward(state) == 1.0
            assert await reward_style(state) == 22.0

    asyncio.run(run())


def test_candidate_relative_flow_rule_solve2_tail_reward_and_balanced_second_slots() -> None:
    async def run() -> None:
        env = load_environment(
            split="train_candidate_relative_flow_rule_solve2_depth2",
            min_depth=2,
            max_depth=2,
            num_examples=192,
            seed=46,
            reward_style="action_gated_candidate_path_tail_solve",
            prompt_style="stage_candidate_relative_flow_rule_solve2_native_tool",
            max_turns=4,
            move_budget=2,
            allow_text_tool_actions=False,
        )
        slot_counts: Counter[int] = Counter()
        slots_by_visible_index_mod: dict[int, set[int]] = defaultdict(set)
        constant_rewards = []
        visible_index_slot_rewards: dict[str, list[float]] = {direction: [] for direction in DIRECTIONS}
        for raw_row in env.get_dataset():
            row = dict(raw_row)
            prompt_text = row["prompt"][0]["content"]
            public_id = str(row.get("src_id", row["example_id"]))
            assert public_id not in prompt_text
            visible_index = int(public_id.rsplit("-", 1)[1])
            visible_index_slot = 1 + (visible_index % 4)
            state = {"task": row}
            await env.setup_state(state)
            rollout = state["megaminx"]
            first_face, first_direction = rollout.inverse_solution[0]
            first_index = row["candidate_faces"].index(first_face) + 1
            await env.env_response(
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "first-call",
                                "name": "select_candidate",
                                "arguments": json.dumps(
                                    {"index": first_index, "direction": first_direction}
                                ),
                            }
                        ],
                    )
                ],
                state,
            )
            assert await action_gated_candidate_path_tail_solve_reward(state) == 0.25
            assert await candidate_path_completed(state) == 0.0
            target_slot = int(await second_target_candidate_index(state))
            assert target_slot in {1, 2, 3, 4}
            slot_counts[target_slot] += 1
            slots_by_visible_index_mod[visible_index % 4].add(target_slot)
            assert rollout.inverse_solution[1][0] in rollout.second_candidate_faces

            constant_state = {"task": row}
            await env.setup_state(constant_state)
            await env.env_response(
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "constant-first",
                                "name": "select_candidate",
                                "arguments": json.dumps({"index": 1, "direction": "cw"}),
                            }
                        ],
                    )
                ],
                constant_state,
            )
            if not constant_state["megaminx"].finished:
                await env.env_response(
                    [
                        AssistantMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "constant-second",
                                    "name": "select_candidate",
                                    "arguments": json.dumps({"index": 1, "direction": "cw"}),
                                }
                            ],
                        )
                    ],
                    constant_state,
                )
            constant_rewards.append(
                await action_gated_candidate_path_tail_solve_reward(constant_state)
            )

            for fixed_direction in DIRECTIONS:
                shortcut_state = {"task": row}
                await env.setup_state(shortcut_state)
                await env.env_response(
                    [
                        AssistantMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": f"shortcut-first-{fixed_direction}",
                                    "name": "select_candidate",
                                    "arguments": json.dumps(
                                        {"index": first_index, "direction": first_direction}
                                    ),
                                }
                            ],
                        )
                    ],
                    shortcut_state,
                )
                if not shortcut_state["megaminx"].finished:
                    await env.env_response(
                        [
                            AssistantMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "id": f"shortcut-second-{fixed_direction}",
                                        "name": "select_candidate",
                                        "arguments": json.dumps(
                                            {
                                                "index": visible_index_slot,
                                                "direction": fixed_direction,
                                            }
                                        ),
                                    }
                                ],
                            )
                        ],
                        shortcut_state,
                    )
                visible_index_slot_rewards[fixed_direction].append(
                    await action_gated_candidate_path_tail_solve_reward(shortcut_state)
                )

        assert set(slot_counts) == {1, 2, 3, 4}
        assert max(slot_counts.values()) - min(slot_counts.values()) <= 32
        assert all(len(slots) >= 3 for slots in slots_by_visible_index_mod.values())
        assert sum(constant_rewards) / len(constant_rewards) < 0.12
        assert (
            max(
                sum(rewards) / len(rewards)
                for rewards in visible_index_slot_rewards.values()
            )
            < 0.45
        )

    asyncio.run(run())


def test_action_gated_candidate_path_tail_solve_rewards_second_step_signal() -> None:
    async def run() -> None:
        env = load_environment(
            split="train_candidate_relative_flow_rule_solve2_depth2",
            min_depth=2,
            max_depth=2,
            num_examples=48,
            seed=104,
            reward_style="action_gated_candidate_path_tail_solve",
            prompt_style="stage_candidate_relative_flow_rule_solve2_native_tool",
            max_turns=4,
            move_budget=2,
            allow_text_tool_actions=False,
        )
        wrong_tail_checked = False
        for raw_row in env.get_dataset():
            row = dict(raw_row)
            state = {"task": row}
            await env.setup_state(state)
            rollout = state["megaminx"]
            first_face, first_direction = rollout.inverse_solution[0]
            first_index = row["candidate_faces"].index(first_face) + 1
            await env.env_response(
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "first-call",
                                "name": "select_candidate",
                                "arguments": json.dumps(
                                    {"index": first_index, "direction": first_direction}
                                ),
                            }
                        ],
                    )
                ],
                state,
            )
            second_face, second_direction = rollout.inverse_solution[1]
            second_index = rollout.candidate_faces.index(second_face) + 1

            wrong_state = {"task": row}
            await env.setup_state(wrong_state)
            wrong_rollout = wrong_state["megaminx"]
            await env.env_response(
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "wrong-first",
                                "name": "select_candidate",
                                "arguments": json.dumps(
                                    {"index": first_index, "direction": first_direction}
                                ),
                            }
                        ],
                    )
                ],
                wrong_state,
            )
            wrong_direction = "ccw" if second_direction == "cw" else "cw"
            await env.env_response(
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "wrong-second",
                                "name": "select_candidate",
                                "arguments": json.dumps(
                                    {"index": second_index, "direction": wrong_direction}
                                ),
                            }
                        ],
                    )
                ],
                wrong_state,
            )
            wrong_reward = await action_gated_candidate_path_tail_solve_reward(wrong_state)
            if not wrong_rollout.solved():
                assert await candidate_path_completed(wrong_state) == 1.0
                assert await second_candidate_index(wrong_state) == float(second_index)
                assert await second_candidate_face_correct(wrong_state) == 1.0
                assert await second_rotate_direction_correct(wrong_state) == 0.0
                assert 0.35 <= wrong_reward < 0.45
                wrong_tail_checked = True
                break

        assert wrong_tail_checked

        solved_state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(solved_state)
        solved_rollout = solved_state["megaminx"]
        first_face, first_direction = solved_rollout.inverse_solution[0]
        first_index = solved_state["task"]["candidate_faces"].index(first_face) + 1
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "solve-first",
                            "name": "select_candidate",
                            "arguments": json.dumps(
                                {"index": first_index, "direction": first_direction}
                            ),
                        }
                    ],
                )
            ],
            solved_state,
        )
        second_face, second_direction = solved_rollout.inverse_solution[1]
        second_index = solved_rollout.candidate_faces.index(second_face) + 1
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "solve-second",
                            "name": "select_candidate",
                            "arguments": json.dumps(
                                {"index": second_index, "direction": second_direction}
                            ),
                        }
                    ],
                )
            ],
            solved_state,
        )
        assert solved_rollout.solved()
        assert await second_rotate_correct(solved_state) == 1.0
        assert await second_candidate_face_correct(solved_state) == 1.0
        assert await second_candidate_relative_flow_count(solved_state) >= 1.0
        assert await second_candidate_relative_flow_margin(solved_state) > 0.0
        assert await second_candidate_relative_flow_is_candidate_max(solved_state) == 1.0
        assert await candidate_path_completed(solved_state) == 1.0
        assert await action_gated_candidate_path_tail_solve_reward(solved_state) == 1.0
        assert await reward_style(solved_state) == 23.0

    asyncio.run(run())


def test_target_face_in_candidate_set_metric_for_geometry_frontier() -> None:
    async def run() -> None:
        env = load_environment(
            min_depth=1,
            max_depth=2,
            num_examples=24,
            seed=92,
            reward_style="action_gated_candidate_geometry_frontier",
            prompt_style="stage_candidate_geometry_frontier_native_tool",
            max_turns=2,
            move_budget=1,
        )
        values = []
        for raw_row in env.get_dataset():
            state = {"task": dict(raw_row)}
            await env.setup_state(state)
            rollout = state["megaminx"]
            expected = rollout.inverse_solution[0][0] in state["task"]["candidate_faces"]
            value = await target_face_in_candidate_set(state)
            assert value == float(expected)
            values.append(value)
        assert any(value == 1.0 for value in values)
        assert all(value in {0.0, 1.0} for value in values)

    asyncio.run(run())


def test_action_gated_candidate_geometry_frontier_rewards_without_mask_proxy() -> None:
    async def run() -> None:
        solved_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_candidate_geometry_frontier",
            prompt_style="stage_candidate_geometry_frontier_native_tool",
            max_turns=2,
            move_budget=1,
        )
        solved_state = {"task": dict(solved_env.get_dataset()[0])}
        await solved_env.setup_state(solved_state)
        solved_rollout = solved_state["megaminx"]
        face, direction = solved_rollout.inverse_solution[0]
        solved_index = solved_state["task"]["candidate_faces"].index(face) + 1
        await solved_env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate",
                            "arguments": json.dumps(
                                {"index": solved_index, "direction": direction}
                            ),
                        }
                    ],
                )
            ],
            solved_state,
        )
        assert solved_rollout.solved()
        assert await reward_style(solved_state) == 20.0
        assert await action_gated_candidate_geometry_frontier_reward(solved_state) == 1.0

        invalid_state = {"task": dict(solved_env.get_dataset()[0])}
        await solved_env.setup_state(invalid_state)
        await solved_env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate",
                            "arguments": json.dumps(
                                {"index": str(solved_index), "direction": direction}
                            ),
                        }
                    ],
                )
            ],
            invalid_state,
        )
        assert await candidate_select_call_count(invalid_state) == 1.0
        assert await first_candidate_index(invalid_state) == -1.0
        assert await action_gated_candidate_geometry_frontier_reward(invalid_state) == 0.0

        frontier_env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=24,
            seed=42,
            reward_style="action_gated_candidate_geometry_frontier",
            prompt_style="stage_candidate_geometry_frontier_native_tool",
            max_turns=2,
            move_budget=1,
        )
        selected_row = None
        selected_move = None
        selected_index = None
        for raw_row in frontier_env.get_dataset():
            row = dict(raw_row)
            state = {"task": row}
            await frontier_env.setup_state(state)
            rollout = state["megaminx"]
            target_face, _ = rollout.inverse_solution[0]
            for slot, candidate_face in enumerate(row["candidate_faces"], start=1):
                for candidate_direction in DIRECTIONS:
                    move = (candidate_face, candidate_direction)
                    if candidate_face == target_face:
                        continue
                    after = rollout.initial_puzzle.copy()
                    after.apply_move(*move)
                    if after.is_solved():
                        continue
                    for tail_face in FACES:
                        for tail_direction in DIRECTIONS:
                            tail = after.copy()
                            tail.apply_move(tail_face, tail_direction)
                            if tail.is_solved():
                                selected_row = row
                                selected_move = move
                                selected_index = slot
                                break
                        if selected_row is not None:
                            break
                    if selected_row is not None:
                        break
                if selected_row is not None:
                    break
            if selected_row is not None:
                break

        assert selected_row is not None
        assert selected_move is not None
        assert selected_index is not None
        frontier_state = {"task": selected_row}
        await frontier_env.setup_state(frontier_state)
        await frontier_env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate",
                            "arguments": json.dumps(
                                {"index": selected_index, "direction": selected_move[1]}
                            ),
                        }
                    ],
                )
            ],
            frontier_state,
        )
        assert await first_rotate_face_correct(frontier_state) == 0.0
        assert await action_reaches_one_turn_frontier(frontier_state) == 1.0
        assert await action_gated_candidate_geometry_frontier_reward(frontier_state) == 0.75

    asyncio.run(run())


def test_action_gated_candidate_mask_frontier_equivalence_rewards_equivalent_frontier_moves() -> None:
    def leaves_one_turn_frontier(puzzle: MegaminxPuzzle, move: tuple[str, str]) -> bool:
        after = puzzle.copy()
        after.apply_move(*move)
        if after.is_solved():
            return False
        for tail_face in FACES:
            for tail_direction in DIRECTIONS:
                tail = after.copy()
                tail.apply_move(tail_face, tail_direction)
                if tail.is_solved():
                    return True
        return False

    async def run() -> None:
        env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=24,
            seed=42,
            reward_style="action_gated_candidate_mask_frontier_equivalence",
            prompt_style="stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
            max_turns=2,
            move_budget=1,
        )

        selected_row = None
        selected_move = None
        selected_index = None
        partial_move = None
        partial_index = None
        partial_mask_count = 0
        for raw_row in env.get_dataset():
            row = dict(raw_row)
            state = {"task": row}
            await env.setup_state(state)
            rollout = state["megaminx"]
            target_face, _ = rollout.inverse_solution[0]
            candidate_faces = row["candidate_faces"]
            row_selected_move = None
            row_selected_index = None
            row_partial_move = None
            row_partial_index = None
            row_partial_mask_count = 0
            for slot, face in enumerate(candidate_faces, start=1):
                for direction in DIRECTIONS:
                    move = (face, direction)
                    if face != target_face and leaves_one_turn_frontier(
                        rollout.initial_puzzle,
                        move,
                    ):
                        row_selected_move = move
                        row_selected_index = slot
                    if not leaves_one_turn_frontier(rollout.initial_puzzle, move):
                        after = rollout.initial_puzzle.copy()
                        after.apply_move(*move)
                        if not after.is_solved() and row_partial_move is None:
                            row_partial_move = move
                            row_partial_index = slot
                            row_partial_mask_count = rollout.initial_action_mask_counts[move]
                if row_selected_move is not None:
                    break
            if row_selected_move is not None and row_partial_move is not None:
                selected_row = row
                selected_move = row_selected_move
                selected_index = row_selected_index
                partial_move = row_partial_move
                partial_index = row_partial_index
                partial_mask_count = row_partial_mask_count
                break

        assert selected_row is not None
        assert selected_move is not None
        assert selected_index is not None
        assert partial_move is not None
        assert partial_index is not None

        state = {"task": selected_row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate",
                            "arguments": json.dumps(
                                {"index": selected_index, "direction": selected_move[1]}
                            ),
                        }
                    ],
                )
            ],
            state,
        )

        assert not rollout.solved()
        assert await reward_style(state) == 19.0
        assert await first_rotate_face_correct(state) == 0.0
        assert await action_reaches_one_turn_frontier(state) == 1.0
        assert await frontier_equivalent(state) == 1.0
        assert await action_gated_candidate_mask_frontier_equivalence_reward(state) == 0.90

        partial_state = {"task": selected_row}
        await env.setup_state(partial_state)
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate",
                            "arguments": json.dumps(
                                {"index": partial_index, "direction": partial_move[1]}
                            ),
                        }
                    ],
                )
            ],
            partial_state,
        )
        expected_partial = min(0.60, 0.02 + 0.58 * (partial_mask_count / 5.0))
        assert await action_reaches_one_turn_frontier(partial_state) == 0.0
        assert await frontier_equivalent(partial_state) == 0.0
        assert (
            await action_gated_candidate_mask_frontier_equivalence_reward(partial_state)
            == expected_partial
        )

        solved_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_candidate_mask_frontier_equivalence",
            prompt_style="stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
            max_turns=2,
            move_budget=1,
        )
        solved_state = {"task": dict(solved_env.get_dataset()[0])}
        await solved_env.setup_state(solved_state)
        solved_rollout = solved_state["megaminx"]
        face, direction = solved_rollout.inverse_solution[0]
        solved_index = solved_state["task"]["candidate_faces"].index(face) + 1
        await solved_env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate",
                            "arguments": json.dumps(
                                {"index": solved_index, "direction": direction}
                            ),
                        }
                    ],
                )
            ],
            solved_state,
        )
        assert solved_rollout.solved()
        assert await action_reaches_one_turn_frontier(solved_state) == 0.0
        assert await frontier_equivalent(solved_state) == 1.0
        assert await action_gated_candidate_mask_frontier_equivalence_reward(solved_state) == 1.0

    asyncio.run(run())


def test_action_gated_candidate_mask_index_rank_reward_credits_hidden_face_and_public_mask_rank() -> None:
    async def run() -> None:
        env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=1,
            seed=89,
            reward_style="action_gated_candidate_mask_index_rank",
            prompt_style="stage_candidate_scorecard_mask_index_native_tool",
            max_turns=2,
            move_budget=1,
        )
        row = dict(env.get_dataset()[0])
        candidate_faces = row["candidate_faces"]

        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        target_face = rollout.inverse_solution[0][0]
        target_index = candidate_faces.index(target_face) + 1
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate_index",
                            "arguments": json.dumps({"index": target_index}),
                        }
                    ],
                )
            ],
            state,
        )

        assert await reward_style(state) == 18.0
        assert await candidate_select_call_count(state) == 1.0
        assert await action_taken(state) == 0.0
        assert await first_candidate_face_correct(state) == 1.0
        assert await first_candidate_public_mask_is_max(state) == 1.0
        assert await action_gated_candidate_mask_index_rank_reward(state) == 1.0

        wrong_state = {"task": row}
        await env.setup_state(wrong_state)
        wrong_rollout = wrong_state["megaminx"]
        wrong_index = next(
            index
            for index, face in enumerate(candidate_faces, start=1)
            if face != target_face
        )
        wrong_face = candidate_faces[wrong_index - 1]
        scores = [
            max(
                wrong_rollout.initial_action_mask_counts[(face, direction)]
                for direction in DIRECTIONS
            )
            for face in candidate_faces
        ]
        chosen_score = max(
            wrong_rollout.initial_action_mask_counts[(wrong_face, direction)]
            for direction in DIRECTIONS
        )
        expected_rank_credit = sum(score <= chosen_score for score in scores) / len(scores)
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate_index",
                            "arguments": json.dumps({"index": wrong_index}),
                        }
                    ],
                )
            ],
            wrong_state,
        )

        assert await first_candidate_face_correct(wrong_state) == 0.0
        assert await first_candidate_public_mask_count(wrong_state) == float(chosen_score)
        assert await first_candidate_public_mask_is_max(wrong_state) == float(
            chosen_score == max(scores)
        )
        assert await first_candidate_public_mask_rank_credit(wrong_state) == expected_rank_credit
        assert (
            await action_gated_candidate_mask_index_rank_reward(wrong_state)
            == 0.45 * expected_rank_credit
        )

        protocol_state = {"task": row}
        await env.setup_state(protocol_state)
        await env.env_response(
            [
                AssistantMessage(
                    content="visible text",
                    tool_calls=[
                        {
                            "id": "native-call",
                            "name": "select_candidate_index",
                            "arguments": json.dumps({"index": target_index}),
                        }
                    ],
                )
            ],
            protocol_state,
        )
        assert await first_candidate_face_correct(protocol_state) == 1.0
        assert await action_gated_candidate_mask_index_rank_reward(protocol_state) == 0.0

    asyncio.run(run())


def test_face_discovery_reward_styles_reject_prompt_mismatches() -> None:
    mismatches = [
        ("action_gated_candidate_tournament", "stage_face_tournament_native_tool"),
        ("action_gated_candidate_index", "stage_candidate_tournament_native_tool"),
        ("action_gated_candidate_mask_index_rank", "stage_candidate_index_native_tool"),
        ("action_gated_candidate_mask_index_rank", "stage_candidate_scorecard_mask_native_tool"),
        (
            "action_gated_candidate_mask_frontier_equivalence",
            "stage_candidate_scorecard_mask_native_tool",
        ),
        (
            "action_gated_candidate_geometry_frontier",
            "stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
        ),
        (
            "action_gated_candidate_geometry_frontier",
            "stage_candidate_tournament_native_tool",
        ),
        ("action_gated_face_tournament", "stage_candidate_tournament_native_tool"),
        ("action_gated_face_tournament", "stage_candidate_index_native_tool"),
        ("action_gated_face_discovery", "stage_face_tournament_native_tool"),
    ]
    for reward_style_name, prompt_style in mismatches:
        try:
            load_environment(
                min_depth=1,
                max_depth=2,
                num_examples=1,
                reward_style=reward_style_name,
                prompt_style=prompt_style,
                move_budget=1,
            )
        except ValueError as error:
            assert reward_style_name in str(error)
        else:
            raise AssertionError(
                f"Expected {reward_style_name} to reject prompt_style={prompt_style}"
            )


def test_candidate_select_first_index_metric_is_first_valid_call() -> None:
    async def run() -> None:
        env = load_environment(
            min_depth=2,
            max_depth=2,
            num_examples=1,
            seed=84,
            reward_style="action_gated_candidate_tournament",
            prompt_style="stage_candidate_tournament_native_tool",
            max_turns=3,
            move_budget=2,
        )
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        await env.env_response(
            [
                AssistantMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "native-call-1",
                            "name": "select_candidate",
                            "arguments": json.dumps({"index": 1, "direction": "cw"}),
                        },
                        {
                            "id": "native-call-2",
                            "name": "select_candidate",
                            "arguments": json.dumps({"index": 2, "direction": "ccw"}),
                        },
                    ],
                )
            ],
            state,
        )

        assert await candidate_select_call_count(state) == 2.0
        assert await first_candidate_index(state) == 1.0
        assert await action_gated_candidate_tournament_reward(state) == 0.0

    asyncio.run(run())


def test_action_gated_strict_shaped_direction_reward_uses_strict_protocol_gate() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=20,
            reward_style="action_gated_strict_shaped_direction",
            prompt_style="stage_solve_direction_flow_native_tool",
            max_turns=2,
            move_budget=1,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        message = AssistantMessage(
            content="extra visible content",
            tool_calls=[
                {
                    "id": "native-call",
                    "name": "rotate",
                    "arguments": json.dumps({"face": face, "direction": direction}),
                }
            ],
        )
        await env.env_response([message], state)

        assert rollout.solved()
        assert await protocol_violation_count(state) == 1.0
        assert await action_gated_binary_direction_reward(state) == 0.0
        assert await action_gated_strict_shaped_direction_reward(state) == 0.0

    asyncio.run(run())


def test_action_gated_strict_shaped_direction_reward_handles_native_tool_calls() -> None:
    async def run() -> None:
        solved_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_strict_shaped_direction",
            prompt_style="stage_solve_direction_flow_native_tool",
            max_turns=2,
            move_budget=1,
        )
        solved_state = {"task": dict(solved_env.get_dataset()[0])}
        await solved_env.setup_state(solved_state)
        solved_rollout = solved_state["megaminx"]
        face, direction = solved_rollout.inverse_solution[0]
        solved_message = AssistantMessage(
            content="",
            tool_calls=[
                {
                    "id": "native-call",
                    "name": "rotate",
                    "arguments": json.dumps({"face": face, "direction": direction}),
                }
            ],
        )

        await solved_env.env_response([solved_message], solved_state)

        assert solved_rollout.solved()
        assert await native_tool_call_count(solved_state) == 1.0
        assert await text_tool_action_count(solved_state) == 0.0
        assert await first_rotate_face_id(solved_state) == float(FACES.index(face))
        assert await first_rotate_direction_id(solved_state) == float(DIRECTIONS.index(direction))
        assert await action_gated_binary_direction_reward(solved_state) == 1.0
        assert await action_gated_strict_shaped_direction_reward(solved_state) == 1.0

        wrong_direction_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_strict_shaped_direction",
            prompt_style="stage_solve_direction_flow_native_tool",
            max_turns=2,
            move_budget=1,
        )
        wrong_direction_state = {"task": dict(wrong_direction_env.get_dataset()[0])}
        await wrong_direction_env.setup_state(wrong_direction_state)
        wrong_direction_rollout = wrong_direction_state["megaminx"]
        target_face, target_direction = wrong_direction_rollout.inverse_solution[0]
        wrong_direction = next(item for item in DIRECTIONS if item != target_direction)
        wrong_direction_message = AssistantMessage(
            content="",
            tool_calls=[
                {
                    "id": "native-call",
                    "name": "rotate",
                    "arguments": json.dumps({"face": target_face, "direction": wrong_direction}),
                }
            ],
        )

        await wrong_direction_env.env_response([wrong_direction_message], wrong_direction_state)

        assert await first_rotate_face_correct(wrong_direction_state) == 1.0
        assert await first_rotate_direction_correct(wrong_direction_state) == 0.0
        assert abs(await action_gated_strict_shaped_direction_reward(wrong_direction_state) - 0.40) < 1e-9

        wrong_face_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=42,
            reward_style="action_gated_strict_shaped_direction",
            prompt_style="stage_solve_direction_flow_native_tool",
            max_turns=2,
            move_budget=1,
        )
        wrong_face_state = {"task": dict(wrong_face_env.get_dataset()[0])}
        await wrong_face_env.setup_state(wrong_face_state)
        wrong_face_rollout = wrong_face_state["megaminx"]
        wrong_face = next(item for item in FACES if item != target_face)
        wrong_face_message = AssistantMessage(
            content="",
            tool_calls=[
                {
                    "id": "native-call",
                    "name": "rotate",
                    "arguments": json.dumps({"face": wrong_face, "direction": target_direction}),
                }
            ],
        )

        await wrong_face_env.env_response([wrong_face_message], wrong_face_state)

        assert wrong_face_rollout.move_count == 1
        assert await first_rotate_face_correct(wrong_face_state) == 0.0
        assert await first_rotate_direction_correct(wrong_face_state) == 1.0
        assert abs(await action_gated_strict_shaped_direction_reward(wrong_face_state) - 0.10) < 1e-9

    asyncio.run(run())


def test_action_gated_strict_shaped_direction_reward_rejects_no_action_and_invalid_native_args() -> None:
    async def run() -> None:
        no_action_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=43,
            reward_style="action_gated_strict_shaped_direction",
            prompt_style="stage_solve_direction_flow_native_tool",
            max_turns=2,
            move_budget=1,
        )
        no_action_state = {"task": dict(no_action_env.get_dataset()[0])}
        await no_action_env.setup_state(no_action_state)
        no_action_message = AssistantMessage(content="")
        no_action_state["trajectory"] = [{"completion": [no_action_message]}]

        assert await no_action_env.no_tools_called(no_action_state) is True
        assert await no_action_env.env_response([no_action_message], no_action_state) == []
        assert await native_tool_call_count(no_action_state) == 0.0
        assert await text_tool_action_count(no_action_state) == 0.0
        assert await action_gated_strict_shaped_direction_reward(no_action_state) == 0.0

        invalid_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=43,
            reward_style="action_gated_strict_shaped_direction",
            prompt_style="stage_solve_direction_flow_native_tool",
            max_turns=2,
            move_budget=1,
        )
        invalid_state = {"task": dict(invalid_env.get_dataset()[0])}
        await invalid_env.setup_state(invalid_state)
        invalid_rollout = invalid_state["megaminx"]
        invalid_message = AssistantMessage(
            content="",
            tool_calls=[
                {
                    "id": "bad-native-call",
                    "name": "rotate",
                    "arguments": json.dumps({"face": None, "direction": "cw"}),
                }
            ],
        )

        response = await invalid_env.env_response([invalid_message], invalid_state)

        assert response[0].role == "tool"
        assert "Illegal move" in response[0].content
        assert invalid_rollout.move_count == 0
        assert invalid_rollout.illegal_moves == 1
        assert await native_tool_call_count(invalid_state) == 1.0
        assert await action_gated_strict_shaped_direction_reward(invalid_state) == 0.0

    asyncio.run(run())


def test_action_gated_overlap_rewards_neighbor_ring_overlap() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=23,
            reward_style="action_gated_overlap",
            prompt_style="choice_json_action",
        )
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        target_face, target_direction = rollout.inverse_solution[0]
        assert set(rollout.initial_affected_faces) == set(
            DEFAULT_TOPOLOGY.neighbor_rings[target_face]
        )

        await env.rotate(target_face, next(d for d in DIRECTIONS if d != target_direction), rollout)

        assert await first_rotate_face_correct(state) == 1.0
        assert await first_rotate_neighbor_overlap(state) == 1.0
        assert 0.45 <= await action_gated_overlap_reward(state) <= 0.65

    asyncio.run(run())


def test_json_text_tool_action_fallback_requires_opt_in() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=17,
            reward_style="action_gated_dense",
            prompt_style="action_first",
        )
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        action_json = json.dumps(
            [
                {
                    "name": "rotate",
                    "parameters": {"face": face, "direction": direction},
                }
            ]
        )
        message = AssistantMessage(reasoning_content=action_json)
        state["trajectory"] = [{"completion": [message]}]
        assert await env.no_tools_called(state) is True
        response = await env.env_response([message], state)
        assert response == []
        assert rollout.move_count == 0
        assert await native_tool_call_count(state) == 0.0
        assert await text_tool_action_count(state) == 0.0

        legacy_env = load_environment(
            split="depth1",
            num_examples=1,
            seed=17,
            reward_style="action_gated_dense",
            prompt_style="action_first",
            allow_text_tool_actions=True,
        )
        legacy_state = {"task": dict(legacy_env.get_dataset()[0])}
        await legacy_env.setup_state(legacy_state)
        legacy_rollout = legacy_state["megaminx"]
        face, direction = legacy_rollout.inverse_solution[0]
        legacy_message = AssistantMessage(
            reasoning_content=json.dumps(
                [
                    {
                        "name": "rotate",
                        "parameters": {"face": face, "direction": direction},
                    }
                ]
            )
        )
        legacy_state["trajectory"] = [{"completion": [legacy_message]}]
        assert await legacy_env.no_tools_called(legacy_state) is False
        response = await legacy_env.env_response([legacy_message], legacy_state)
        assert response[0].role == "tool"
        assert legacy_rollout.solved()
        assert await rotate_call_count(legacy_state) == 1.0
        assert await native_tool_call_count(legacy_state) == 0.0
        assert await text_tool_action_count(legacy_state) == 1.0
        assert await action_gated_dense_reward(legacy_state) == 1.0

    asyncio.run(run())


def test_native_tool_calls_take_precedence_over_text_action_fallback() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=19,
            reward_style="action_gated_dense",
            prompt_style="action_first",
            allow_text_tool_actions=True,
        )
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        wrong_direction = next(d for d in DIRECTIONS if d != direction)
        visible_json = json.dumps({"tool": "rotate", "args": {"face": face, "direction": wrong_direction}})
        message = AssistantMessage(
            content=visible_json,
            tool_calls=[
                {
                    "id": "native-call",
                    "name": "rotate",
                    "arguments": json.dumps({"face": face, "direction": direction}),
                }
            ],
        )

        response = await env.env_response([message], state)

        assert response[0].role == "tool"
        assert rollout.solved()
        assert await rotate_call_count(state) == 1.0
        assert await native_tool_call_count(state) == 1.0
        assert await text_tool_action_count(state) == 0.0
        assert await action_gated_dense_reward(state) == 1.0

    asyncio.run(run())


def test_strict_binary_rejects_native_call_with_visible_content() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=20,
            reward_style="action_gated_binary_direction",
            prompt_style="stage_solve_direction_flow_native_tool",
            max_turns=2,
            move_budget=1,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        message = AssistantMessage(
            content="I will solve this first.",
            tool_calls=[
                {
                    "id": "native-call",
                    "name": "rotate",
                    "arguments": json.dumps({"face": face, "direction": direction}),
                }
            ],
        )

        response = await env.env_response([message], state)

        assert response[0].role == "tool"
        assert rollout.solved()
        assert await native_tool_call_count(state) == 1.0
        assert await protocol_violation_count(state) == 1.0
        assert await action_gated_binary_direction_reward(state) == 0.0

    asyncio.run(run())


def test_strict_binary_rejects_hidden_json_after_visible_text() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=21,
            reward_style="action_gated_binary_direction",
            prompt_style="stage_solve_direction_flow_json_action",
            max_turns=2,
            move_budget=1,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        message = AssistantMessage(
            content="I will solve it now.",
            reasoning_content=json.dumps({"tool": "rotate", "args": {"face": face, "direction": direction}}),
        )
        state["trajectory"] = [{"completion": [message]}]

        assert await env.no_tools_called(state) is False
        response = await env.env_response([message], state)

        assert response[0].role == "tool"
        assert "Tool error" in response[0].content
        assert rollout.move_count == 0
        assert await text_tool_action_count(state) == 1.0
        assert await private_text_action_count(state) == 0.0
        assert await tool_parse_error_count(state) == 1.0
        assert await protocol_violation_count(state) == 1.0
        assert await action_gated_binary_direction_reward(state) == 0.0

    asyncio.run(run())


def test_strict_binary_accepts_private_only_json_action() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=22,
            reward_style="action_gated_binary_direction",
            prompt_style="stage_solve_direction_flow_json_action",
            max_turns=2,
            move_budget=1,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        message = AssistantMessage(
            content="",
            reasoning_content=json.dumps({"tool": "rotate", "args": {"face": face, "direction": direction}}),
        )
        state["trajectory"] = [{"completion": [message]}]

        assert await env.no_tools_called(state) is False
        response = await env.env_response([message], state)

        assert response[0].role == "tool"
        assert rollout.solved()
        assert await text_tool_action_count(state) == 1.0
        assert await private_text_action_count(state) == 1.0
        assert await action_gated_binary_direction_reward(state) == 1.0

    asyncio.run(run())


def test_private_structured_reasoning_metadata_is_ignored() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=26,
            reward_style="action_gated_binary_direction",
            prompt_style="stage_solve_direction_flow_json_action",
            allow_text_tool_actions=True,
            max_turns=2,
            move_budget=1,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        message = AssistantMessage(content="", reasoning={"summary": []})
        state["trajectory"] = [{"completion": [message]}]

        assert await env.no_tools_called(state) is True
        assert await env.env_response([message], state) == []
        assert rollout.move_count == 0
        assert await text_tool_action_count(state) == 0.0
        assert await private_text_action_count(state) == 0.0
        assert await tool_parse_error_count(state) == 0.0
        assert await action_gated_binary_direction_reward(state) == 0.0

    asyncio.run(run())


def test_private_malformed_json_counts_private_parse_error() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=26,
            reward_style="action_gated_binary_direction",
            prompt_style="stage_solve_direction_flow_json_action",
            allow_text_tool_actions=True,
            max_turns=2,
            move_budget=1,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        message = AssistantMessage(content="", reasoning_content='{"tool": "rotate", "args": ')
        state["trajectory"] = [{"completion": [message]}]

        assert await env.no_tools_called(state) is False
        response = await env.env_response([message], state)

        assert response[0].role == "tool"
        assert "Tool error" in response[0].content
        assert rollout.move_count == 0
        assert await text_tool_action_count(state) == 1.0
        assert await private_text_action_count(state) == 1.0
        assert await tool_parse_error_count(state) == 1.0
        assert await protocol_violation_count(state) == 1.0
        assert await action_gated_binary_direction_reward(state) == 0.0

    asyncio.run(run())


def test_json_action_fallback_rejects_plain_text_before_action() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=25,
            reward_style="action_gated_dense",
            prompt_style="action_first",
        )
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        action_json = json.dumps({"tool": "rotate", "args": {"face": face, "direction": direction}})
        message = AssistantMessage(content=f"I will solve it now. {action_json}")
        state["trajectory"] = [{"completion": [message]}]

        assert await env.no_tools_called(state) is True
        assert rollout.move_count == 0
        assert await action_gated_dense_reward(state) == 0.0

    asyncio.run(run())


def test_json_action_fallback_penalizes_malformed_args() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=27,
            reward_style="action_gated_curriculum",
            prompt_style="sensor_choice_json_action",
            allow_text_tool_actions=True,
        )
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]

        for payload in (
            {"tool": "rotate", "args": None},
            {"tool": "rotate"},
            {"tool": "rotate", "args": {"face": None, "direction": "cw"}},
            {"tool": "rotate", "args": {"face": "A", "direction": None}},
        ):
            message = AssistantMessage(content=json.dumps(payload))
            state["trajectory"] = [{"completion": [message]}]
            assert await env.no_tools_called(state) is False
            response = await env.env_response([message], state)
            assert response[0].role == "tool"
            assert "Tool error" in response[0].content or "Illegal move" in response[0].content

        assert rollout.illegal_moves == 4
        assert rollout.move_count == 0
        assert await action_gated_curriculum_reward(state) == 0.0

    asyncio.run(run())


def test_text_fallback_unknown_tool_counts_one_illegal_action() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=29,
            reward_style="action_gated_curriculum",
            prompt_style="sensor_choice_json_action",
            allow_text_tool_actions=True,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        message = AssistantMessage(content=json.dumps({"tool": "bogus", "args": {}}))

        response = await env.env_response([message], state)

        assert response[0].role == "tool"
        assert "Tool error" in response[0].content
        assert rollout.illegal_moves == 1
        assert await text_tool_action_count(state) == 1.0
        assert await tool_call_error_count(state) == 1.0

    asyncio.run(run())


def test_text_fallback_rejects_multi_action_json_lists() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=30,
            reward_style="action_gated_binary_direction",
            prompt_style="stage_solve_direction_flow_json_action",
            allow_text_tool_actions=True,
            max_turns=2,
            move_budget=1,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        message = AssistantMessage(
            content=json.dumps(
                [
                    {"tool": "rotate", "args": {"face": face, "direction": direction}},
                    {"tool": "inspect", "args": {"face": "all"}},
                ]
            )
        )

        assert await env.no_tools_called(state | {"trajectory": [{"completion": [message]}]}) is False
        response = await env.env_response([message], state)

        assert response[0].role == "tool"
        assert "Tool error" in response[0].content
        assert rollout.move_count == 0
        assert rollout.illegal_moves == 1
        assert await text_tool_action_count(state) == 1.0
        assert await tool_parse_error_count(state) == 1.0
        assert await action_gated_binary_direction_reward(state) == 0.0

    asyncio.run(run())


def test_text_fallback_malformed_json_counts_parse_error() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=32,
            reward_style="action_gated_binary_direction",
            prompt_style="stage_solve_direction_flow_json_action",
            allow_text_tool_actions=True,
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        message = AssistantMessage(content='{"tool": "rotate", "args": ')
        state["trajectory"] = [{"completion": [message]}]

        assert await env.no_tools_called(state) is False
        response = await env.env_response([message], state)

        assert response[0].role == "tool"
        assert "Tool error" in response[0].content
        assert rollout.move_count == 0
        assert rollout.illegal_moves == 1
        assert await text_tool_action_count(state) == 1.0
        assert await tool_parse_error_count(state) == 1.0
        assert await action_gated_binary_direction_reward(state) == 0.0

    asyncio.run(run())


def test_sensor_json_action_fallback_executes_rotate() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=28,
            reward_style="action_gated_curriculum",
            prompt_style="sensor_choice_json_action",
            allow_text_tool_actions=True,
        )
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        message = AssistantMessage(
            content=json.dumps({"tool": "rotate", "args": {"face": face, "direction": direction}})
        )
        state["trajectory"] = [{"completion": [message]}]

        assert await env.no_tools_called(state) is False
        response = await env.env_response([message], state)
        assert response[0].role == "tool"
        assert rollout.solved()
        assert await first_rotate_correct(state) == 1.0
        assert await action_gated_curriculum_reward(state) == 1.0

    asyncio.run(run())


def test_json_action_fallback_reads_extra_reasoning_fields() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=22,
            reward_style="action_gated_curriculum",
            prompt_style="topology_choice_json_action",
            allow_text_tool_actions=True,
        )
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        action_json = json.dumps(
            [
                {
                    "name": "rotate",
                    "parameters": {"face": face, "direction": direction},
                }
            ]
        )
        message = AssistantMessage(content=None, reasoning=action_json)
        state["trajectory"] = [{"completion": [message]}]
        assert await env.no_tools_called(state) is False
        response = await env.env_response([message], state)
        assert response[0].role == "tool"
        assert rollout.solved()
        assert await rotate_call_count(state) == 1.0
        assert await action_gated_curriculum_reward(state) == 1.0

    asyncio.run(run())


def test_reasoned_json_action_requires_visible_final_json() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=41,
            reward_style="action_gated_binary_direction",
            prompt_style="stage_direction_flow_reasoned_json_action",
            allow_text_tool_actions=True,
        )
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        face, direction = rollout.inverse_solution[0]
        action_json = json.dumps({"tool": "rotate", "args": {"face": face, "direction": direction}})

        hidden_only = AssistantMessage(content="", reasoning_content=action_json)
        state["trajectory"] = [{"completion": [hidden_only]}]
        assert await env.no_tools_called(state) is True
        response = await env.env_response([hidden_only], state)
        assert response == []
        assert rollout.move_count == 0

        visible_final = AssistantMessage(
            content=action_json,
            reasoning_content="I compare the table privately.",
        )
        state["trajectory"] = [{"completion": [visible_final]}]
        assert await env.no_tools_called(state) is False
        response = await env.env_response([visible_final], state)
        assert response[0].role == "tool"
        assert rollout.solved()
        assert await rotate_call_count(state) == 1.0
        assert await action_gated_binary_direction_reward(state) == 1.0

    asyncio.run(run())


def test_invalid_native_tool_call_args_are_penalized_without_crashing() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=33,
            reward_style="action_gated_curriculum",
            prompt_style="sensor_native_tool",
        )
        row = dict(env.get_dataset()[0])
        state = {"task": row}
        await env.setup_state(state)
        rollout = state["megaminx"]
        message = AssistantMessage(
            tool_calls=[
                {
                    "id": "bad-native-call",
                    "name": "rotate",
                    "arguments": json.dumps({"face": None, "direction": "cw"}),
                }
            ]
        )

        response = await env.env_response([message], state)

        assert response[0].role == "tool"
        assert "Illegal move" in response[0].content
        assert rollout.illegal_moves == 1
        assert rollout.move_count == 0
        assert await native_tool_call_count(state) == 1.0
        assert await text_tool_action_count(state) == 0.0
        assert await action_gated_curriculum_reward(state) == 0.0

    asyncio.run(run())


def test_malformed_native_tool_call_is_counted_as_parse_error() -> None:
    async def run() -> None:
        env = load_environment(
            split="depth1",
            num_examples=1,
            seed=35,
            reward_style="action_gated_binary_direction",
            prompt_style="stage_solve_direction_flow_native_tool",
        )
        state = {"task": dict(env.get_dataset()[0])}
        await env.setup_state(state)
        rollout = state["megaminx"]
        message = AssistantMessage(
            content=json.dumps({"tool": "rotate", "args": {"face": rollout.inverse_solution[0][0], "direction": rollout.inverse_solution[0][1]}}),
            tool_calls=[
                {
                    "id": "malformed-native-call",
                    "name": "rotate",
                    "arguments": "{not-json",
                }
            ],
        )

        response = await env.env_response([message], state)

        assert response[0].role == "tool"
        assert "Tool error" in response[0].content
        assert rollout.move_count == 0
        assert rollout.illegal_moves == 1
        assert await native_tool_call_count(state) == 1.0
        assert await text_tool_action_count(state) == 0.0
        assert await tool_parse_error_count(state) == 1.0
        assert await action_gated_binary_direction_reward(state) == 0.0

    asyncio.run(run())


def test_tool_schema_and_metadata_path() -> None:
    env = load_environment(split="depth1", num_examples=1)
    tool_defs = {tool.name: tool.parameters for tool in env.tool_defs}
    assert tool_defs["rotate"]["required"] == ["face", "direction"]
    assert set(tool_defs["rotate"]["properties"]) == {"face", "direction"}
    assert tool_defs["rotate"]["properties"]["face"]["enum"] == list(FACES)
    assert tool_defs["rotate"]["properties"]["direction"]["enum"] == list(DIRECTIONS)
    assert tool_defs["predict_rotate"]["required"] == ["face", "direction", "predicted_after"]
    assert set(tool_defs["predict_rotate"]["properties"]) == {
        "face",
        "direction",
        "predicted_after",
    }
    assert tool_defs["predict_rotate"]["properties"]["face"]["enum"] == list(FACES)
    assert tool_defs["predict_rotate"]["properties"]["direction"]["enum"] == list(DIRECTIONS)
    predicted_after_schema = tool_defs["predict_rotate"]["properties"]["predicted_after"]
    assert predicted_after_schema["type"] == "array"
    assert predicted_after_schema["minItems"] == 5
    assert predicted_after_schema["maxItems"] == 5
    assert predicted_after_schema["items"]["type"] == "string"
    assert predicted_after_schema["items"]["pattern"] == "^[A-L]{3}$"
    assert tool_defs["select_candidate"]["required"] == ["index", "direction"]
    assert set(tool_defs["select_candidate"]["properties"]) == {"index", "direction"}
    assert tool_defs["select_candidate"]["properties"]["index"]["type"] == "integer"
    assert tool_defs["select_candidate"]["properties"]["index"]["minimum"] == 1
    assert tool_defs["select_candidate"]["properties"]["index"]["maximum"] == 4
    assert tool_defs["select_candidate"]["properties"]["direction"]["enum"] == list(DIRECTIONS)
    assert tool_defs["select_candidate_index"]["required"] == ["index"]
    assert set(tool_defs["select_candidate_index"]["properties"]) == {"index"}
    assert tool_defs["select_candidate_index"]["properties"]["index"]["type"] == "integer"
    assert tool_defs["select_candidate_index"]["properties"]["index"]["minimum"] == 1
    assert tool_defs["select_candidate_index"]["properties"]["index"]["maximum"] == 4
    assert tool_defs["inspect"]["required"] == ["face"]
    assert tool_defs["inspect"]["properties"]["face"]["enum"] == [*FACES, "all"]
    assert tool_defs["finish"]["required"] == []
    assert env.env_args["reward_style"] == "dense"
    assert env.env_args["prompt_style"] == "default"
    assert env.env_args["allow_text_tool_actions"] is False
    assert env.env_args["exposed_tool_names"] is None
    assert env.env_args["move_budget"] is None

    candidate_env = load_environment(
        split="depth1",
        num_examples=1,
        reward_style="action_gated_candidate_tournament",
        prompt_style="stage_candidate_tournament_native_tool",
    )
    assert [tool.name for tool in candidate_env.tool_defs] == ["select_candidate"]
    assert candidate_env.env_args["exposed_tool_names"] == ("select_candidate",)


def test_package_metadata_matches_published_environment() -> None:
    env_dir = Path(__file__).resolve().parents[1] / "environments" / "megaminx_solver"
    pyproject = tomllib.loads((env_dir / "pyproject.toml").read_text())
    metadata = json.loads((env_dir / ".prime" / ".env-metadata.json").read_text())

    assert pyproject["project"]["name"] == "megaminx-solver"
    assert pyproject["project"]["version"] == "0.2.57"
    assert pyproject["project"]["license"] == "MIT"
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "megaminx_solver"
    ]
    assert metadata["owner"] == "setrf"
    assert metadata["name"] == "megaminx-solver"
    assert metadata["environment_id"] == "ozde27sytxjkc3wm83zv4e2c"
    assert isinstance(metadata["wheel_sha256"], str)
    assert len(metadata["wheel_sha256"]) == 64
