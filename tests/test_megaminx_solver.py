from __future__ import annotations

import asyncio
from collections import Counter
import json

from megaminx_solver import load_environment
from megaminx_solver.megaminx_solver import (
    action_gated_curriculum_reward,
    action_gated_dense_reward,
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
    ):
        env = load_environment(prompt_style=prompt_style)
        assert env.env_args["allow_text_tool_actions"] is True

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
