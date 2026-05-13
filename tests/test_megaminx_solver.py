from __future__ import annotations

import asyncio
from collections import Counter
import json

from megaminx_solver import load_environment
from megaminx_solver.megaminx_solver import (
    action_gated_curriculum_reward,
    action_gated_dense_reward,
    action_gated_direction_reward,
    action_gated_exact_direction_reward,
    action_gated_overlap_reward,
    action_taken,
    first_rotate_correct,
    first_rotate_direction_correct,
    first_rotate_face_correct,
    first_rotate_neighbor_overlap,
    reward_style,
    rotate_call_count,
    solved_reward,
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

    for prompt_style in (
        "default",
        "action_first",
        "native_action",
        "topology_native_tool",
        "sensor_native_tool",
        "sensor_match_native_tool",
    ):
        env = load_environment(prompt_style=prompt_style)
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


def test_stage_face_hint_direction_json_action_rejects_non_depth1_splits() -> None:
    try:
        load_environment(
            split="easy",
            num_examples=3,
            reward_style="action_gated_exact_direction",
            prompt_style="stage_face_hint_direction_json_action",
            move_budget=1,
        )
    except ValueError as error:
        assert "only defined for depth-1" in str(error)
    else:
        raise AssertionError("Expected staged face-hint prompt to reject non-depth1 split")


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
    assert "A cw turn moves neighbor-strip stickers forward" in prompt
    assert "A ccw turn moves neighbor-strip stickers reverse" in prompt
    assert "If the visible sticker flow is forward, undo with ccw" in prompt
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
        assert await env.no_tools_called(state) is False
        response = await env.env_response([message], state)
        assert response == []
        assert rollout.move_count == 0

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
        assert await action_gated_dense_reward(legacy_state) == 1.0

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
        assert await action_gated_curriculum_reward(state) == 0.0

    asyncio.run(run())


def test_tool_schema_and_metadata_path() -> None:
    env = load_environment(split="depth1", num_examples=1)
    tool_defs = {tool.name: tool.parameters for tool in env.tool_defs}
    assert tool_defs["rotate"]["required"] == ["face", "direction"]
    assert set(tool_defs["rotate"]["properties"]) == {"face", "direction"}
    assert tool_defs["rotate"]["properties"]["face"]["enum"] == list(FACES)
    assert tool_defs["rotate"]["properties"]["direction"]["enum"] == list(DIRECTIONS)
    assert tool_defs["inspect"]["required"] == ["face"]
    assert tool_defs["inspect"]["properties"]["face"]["enum"] == [*FACES, "all"]
    assert tool_defs["finish"]["required"] == []
    assert env.env_args["reward_style"] == "dense"
    assert env.env_args["prompt_style"] == "default"
    assert env.env_args["allow_text_tool_actions"] is False
    assert env.env_args["move_budget"] is None
