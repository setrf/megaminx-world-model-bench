from __future__ import annotations

import asyncio
from collections import Counter

from megaminx_solver import load_environment
from megaminx_solver.megaminx_solver import solved_reward
from megaminx_solver.simulator import (
    CORNER_COUNT,
    EDGE_COUNT,
    FACES,
    POSITIONS_PER_FACE,
    STICKERS_PER_PUZZLE,
    DEFAULT_TOPOLOGY,
    MegaminxPuzzle,
    generate_scramble,
    inverse_moves,
)


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
