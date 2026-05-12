from __future__ import annotations

import asyncio
from collections import Counter
import json

from megaminx_solver import load_environment
from megaminx_solver.megaminx_solver import (
    action_gated_dense_reward,
    action_taken,
    first_rotate_correct,
    first_rotate_direction_correct,
    first_rotate_face_correct,
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


def test_depth1_split_and_prompt_contract() -> None:
    env = load_environment(split="depth1", num_examples=5, seed=13)
    rows = [dict(row) for row in env.get_dataset()]
    assert {row["scramble_depth"] for row in rows} == {1}
    assert {row["move_budget"] for row in rows} == {6}
    assert env.max_turns == 8
    first_prompt = "\n".join(message["content"] for message in rows[0]["prompt"])
    assert "one-turn scramble" in first_prompt
    assert rows[0]["answer"] not in first_prompt


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


def test_json_text_tool_action_fallback_executes_rotate() -> None:
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
        assert response[0].role == "tool"
        assert rollout.solved()
        assert await rotate_call_count(state) == 1.0
        assert await action_gated_dense_reward(state) == 1.0

    asyncio.run(run())


def test_tool_schema_and_metadata_path() -> None:
    env = load_environment(split="depth1", num_examples=1)
    tool_defs = {tool.name: tool.parameters for tool in env.tool_defs}
    assert tool_defs["rotate"]["required"] == ["face", "direction"]
    assert set(tool_defs["rotate"]["properties"]) == {"face", "direction"}
    assert tool_defs["inspect"]["required"] == ["face"]
    assert tool_defs["finish"]["required"] == []
    assert env.env_args["reward_style"] == "dense"
    assert env.env_args["prompt_style"] == "default"
