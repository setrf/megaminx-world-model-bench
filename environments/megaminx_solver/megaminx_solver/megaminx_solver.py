from __future__ import annotations

import json
from dataclasses import dataclass, field
from random import Random
from typing import Any

from datasets import Dataset
import verifiers as vf

from .simulator import (
    DEFAULT_TOPOLOGY,
    DIRECTIONS,
    FACES,
    MegaminxPuzzle,
    MegaminxTopology,
    Move,
    generate_scramble,
    inverse_moves,
    move_to_text,
    moves_to_text,
    validate_move,
)

SYSTEM_PROMPT = """You are solving a text Megaminx.

The puzzle has twelve faces labeled A through L. A solved face contains only its
own label. Use the rotate and inspect tools to change or inspect the state.
Call finish only when the puzzle is solved or when you are giving up.
"""

SPLIT_DEPTHS = {
    "easy": (1, 3),
    "eval_easy": (1, 3),
    "medium": (4, 6),
    "eval_medium": (4, 6),
    "hard": (7, 10),
    "eval_hard": (7, 10),
    "eval": (1, 10),
}


@dataclass
class RolloutMegaminx:
    puzzle: MegaminxPuzzle
    scramble: list[Move]
    inverse_solution: list[Move]
    scramble_depth: int
    move_budget: int
    move_count: int = 0
    illegal_moves: int = 0
    tool_calls: int = 0
    finished: bool = False
    last_move: str = "none"
    history: list[str] = field(default_factory=list)

    def solved(self) -> bool:
        return self.puzzle.is_solved()

    def moves_left(self) -> int:
        return max(0, self.move_budget - self.move_count)

    def exhausted(self) -> bool:
        return self.move_count >= self.move_budget

    def observation(self) -> str:
        status = [
            f"solved={self.solved()}",
            f"sticker_accuracy={self.puzzle.sticker_accuracy():.3f}",
            f"piece_accuracy={self.puzzle.piece_accuracy():.3f}",
            f"moves={self.move_count}/{self.move_budget}",
            f"moves_left={self.moves_left()}",
            f"illegal_moves={self.illegal_moves}",
            f"last_move={self.last_move}",
        ]
        return "\n".join(
            [
                "Status: " + " | ".join(status),
                "Facelet net:",
                self.puzzle.render_net(),
            ]
        )


def _rollout_from_state(state: vf.State) -> RolloutMegaminx | None:
    rollout = state.get("megaminx")
    if isinstance(rollout, RolloutMegaminx):
        return rollout
    return None


async def solved_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(bool(rollout and rollout.solved()))


async def sticker_accuracy(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return rollout.puzzle.sticker_accuracy() if rollout else 0.0


async def piece_accuracy(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return rollout.puzzle.piece_accuracy() if rollout else 0.0


async def efficiency_if_solved(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.solved():
        return 0.0
    ideal = max(1, rollout.scramble_depth)
    used = max(ideal, rollout.move_count)
    return ideal / used


async def illegal_move_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.illegal_moves if rollout else 0)


async def move_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.move_count if rollout else 0)


async def solved_rate(state: vf.State) -> float:
    return await solved_reward(state)


async def scramble_depth(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.scramble_depth if rollout else 0)


async def tool_call_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.tool_calls if rollout else 0)


class MegaminxEnv(vf.StatefulToolEnv):
    def __init__(self, topology: MegaminxTopology | None = None, **kwargs: Any):
        self.topology = topology or DEFAULT_TOPOLOGY
        super().__init__(tools=[], error_formatter=lambda error: f"Tool error: {error}", **kwargs)
        self.add_tool(self.rotate, args_to_skip=["rollout"])
        self.add_tool(self.inspect, args_to_skip=["rollout"])
        self.add_tool(self.finish, args_to_skip=["rollout"])

    async def setup_state(self, state: vf.State) -> None:
        await super().setup_state(state)
        task = _task_from_state(state)
        scramble = _decode_moves(task["scramble"])
        inverse_solution = _decode_moves(task["inverse_solution"])
        puzzle = MegaminxPuzzle.solved(self.topology)
        puzzle.apply_moves(scramble)
        state["megaminx"] = RolloutMegaminx(
            puzzle=puzzle,
            scramble=scramble,
            inverse_solution=inverse_solution,
            scramble_depth=int(task["scramble_depth"]),
            move_budget=int(task["move_budget"]),
        )

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict:
        rollout = _rollout_from_state(state)
        if rollout is not None:
            tool_args["rollout"] = rollout
            if tool_name not in self.tool_map:
                rollout.illegal_moves += 1
        return tool_args

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        tool_messages = await super().env_response(messages, state, **kwargs)
        rollout = _rollout_from_state(state)
        if rollout and (rollout.solved() or rollout.finished or rollout.exhausted()):
            state["final_env_response"] = tool_messages
        return tool_messages

    async def rotate(self, face: str, direction: str, rollout: RolloutMegaminx) -> str:
        """Rotate one Megaminx face.

        Args:
            face: One face label from A through L.
            direction: cw for clockwise or ccw for counterclockwise, viewed from outside the face.

        Returns:
            Updated puzzle observation after the move.
        """
        rollout.tool_calls += 1
        if rollout.finished:
            return "The rollout is already finished.\n" + rollout.observation()
        if rollout.exhausted():
            rollout.finished = True
            return "Move budget exhausted.\n" + rollout.observation()
        try:
            face, direction = validate_move(face, direction)
        except ValueError as error:
            rollout.illegal_moves += 1
            return f"Illegal move: {error}\n" + rollout.observation()

        rollout.puzzle.apply_move(face, direction)
        rollout.move_count += 1
        rollout.last_move = move_to_text((face, direction))
        rollout.history.append(rollout.last_move)
        if rollout.solved() or rollout.exhausted():
            rollout.finished = True
        return rollout.observation()

    async def inspect(self, face: str, rollout: RolloutMegaminx) -> str:
        """Inspect the current Megaminx state.

        Args:
            face: One face label from A through L, or all for the full facelet net.

        Returns:
            Current puzzle observation without rotating any face.
        """
        rollout.tool_calls += 1
        face = face.strip().upper()
        if face == "ALL":
            return rollout.observation()
        if face not in FACES:
            rollout.illegal_moves += 1
            return f"Illegal inspect target: expected A-L or all, got {face!r}.\n" + rollout.observation()
        return "\n".join(
            [
                f"Status: solved={rollout.solved()} | moves={rollout.move_count}/{rollout.move_budget}",
                rollout.puzzle.face_line(face),
            ]
        )

    async def finish(self, rollout: RolloutMegaminx) -> str:
        """End the rollout and receive the final state.

        Returns:
            Final puzzle observation.
        """
        rollout.tool_calls += 1
        rollout.finished = True
        verdict = "Solved." if rollout.solved() else "Not solved."
        return verdict + "\n" + rollout.observation()


def load_environment(
    split: str = "train",
    min_depth: int = 1,
    max_depth: int = 8,
    num_examples: int = 200,
    seed: int = 42,
    max_turns: int | None = None,
) -> vf.Environment:
    dataset = build_dataset(
        split=split,
        min_depth=min_depth,
        max_depth=max_depth,
        num_examples=num_examples,
        seed=seed,
        max_turns=max_turns,
    )
    rubric = vf.Rubric(
        funcs=[
            solved_reward,
            sticker_accuracy,
            piece_accuracy,
            efficiency_if_solved,
            illegal_move_count,
            move_count,
            solved_rate,
            scramble_depth,
            tool_call_count,
        ],
        weights=[0.60, 0.25, 0.10, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    global_max_turns = max_turns if max_turns is not None else 32
    return MegaminxEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        max_turns=global_max_turns,
        env_id="megaminx-solver",
        env_args={
            "split": split,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "num_examples": num_examples,
            "seed": seed,
            "max_turns": max_turns,
        },
    )


def build_dataset(
    split: str,
    min_depth: int,
    max_depth: int,
    num_examples: int,
    seed: int,
    max_turns: int | None,
) -> Dataset:
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    min_depth, max_depth = _resolve_depths(split, min_depth, max_depth)
    if min_depth < 0 or max_depth < min_depth:
        raise ValueError("Expected 0 <= min_depth <= max_depth")

    rng = Random(seed)
    rows: list[dict[str, Any]] = []
    depth_span = max_depth - min_depth + 1
    for index in range(num_examples):
        depth = min_depth + (index % depth_span)
        scramble = generate_scramble(depth, rng)
        inverse_solution = inverse_moves(scramble)
        move_budget = max_turns if max_turns is not None else min(32, 2 * depth + 4)
        puzzle = MegaminxPuzzle.solved(DEFAULT_TOPOLOGY)
        puzzle.apply_moves(scramble)
        rows.append(
            _build_row(
                index=index,
                split=split,
                depth=depth,
                move_budget=int(move_budget),
                puzzle=puzzle,
                scramble=scramble,
                inverse_solution=inverse_solution,
            )
        )
    return Dataset.from_list(rows)


def _build_row(
    index: int,
    split: str,
    depth: int,
    move_budget: int,
    puzzle: MegaminxPuzzle,
    scramble: Sequence[Move],
    inverse_solution: Sequence[Move],
) -> dict[str, Any]:
    prompt = [
        {
            "role": "user",
            "content": _build_prompt(index, split, depth, move_budget, puzzle),
        }
    ]
    task = {
        "prompt": prompt,
        "answer": moves_to_text(inverse_solution),
        "example_id": f"{split}-{index:05d}",
        "split": split,
        "scramble_depth": depth,
        "move_budget": move_budget,
        "scramble": _encode_moves(scramble),
        "inverse_solution": _encode_moves(inverse_solution),
    }
    return {
        **task,
        "info": json.dumps(
            {
                "task": task,
                "env_id": "megaminx-solver",
            }
        ),
    }


def _task_from_state(state: vf.State) -> dict[str, Any]:
    task = state.get("task")
    if isinstance(task, dict) and "scramble" in task:
        return task
    input_data = state.get("input")
    if isinstance(input_data, dict) and "scramble" in input_data:
        return input_data
    info = state.get("info")
    if isinstance(info, str):
        info = json.loads(info)
    if isinstance(info, dict):
        nested = info.get("task")
        if isinstance(nested, str):
            nested = json.loads(nested)
        if isinstance(nested, dict) and "scramble" in nested:
            return nested
    raise KeyError("scramble")


def _resolve_depths(split: str, min_depth: int, max_depth: int) -> tuple[int, int]:
    if split in SPLIT_DEPTHS:
        return SPLIT_DEPTHS[split]
    return min_depth, max_depth


def _build_prompt(
    index: int,
    split: str,
    depth: int,
    move_budget: int,
    puzzle: MegaminxPuzzle,
) -> str:
    return "\n".join(
        [
            f"Example: {split}-{index:05d}",
            f"Scramble depth: {depth}",
            f"Move budget: {move_budget}",
            "Goal: solve the Megaminx so every face contains only its own label.",
            "Use rotate(face, direction), inspect(face), and finish().",
            "Initial observation:",
            puzzle.render_net(),
        ]
    )


def _encode_moves(moves: Sequence[Move]) -> str:
    return json.dumps([{"face": face, "direction": direction} for face, direction in moves])


def _decode_moves(raw: str | list[dict[str, str]]) -> list[Move]:
    if isinstance(raw, str):
        decoded = json.loads(raw)
    else:
        decoded = raw
    moves: list[Move] = []
    for item in decoded:
        moves.append(validate_move(item["face"], item["direction"]))
    return moves
