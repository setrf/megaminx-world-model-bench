from __future__ import annotations

import json
from dataclasses import dataclass, field
from random import Random
from typing import Any, Sequence

from datasets import Dataset
import verifiers as vf
from verifiers.types import ToolMessage

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
own label. Centers identify the target color for each face and do not move.
Corner and edge strings show the sticker colors currently visible on that face.

Use rotate to change the puzzle. Inspect only when you need more state. Call
finish only after the puzzle is solved or when you are giving up.
"""

REWARD_STYLES = {"dense", "action_gated_dense"}
PROMPT_STYLES = {"default", "action_first"}

SPLIT_DEPTHS = {
    "depth1": (1, 1),
    "train_depth1": (1, 1),
    "eval_depth1": (1, 1),
    "easy": (1, 3),
    "train_easy": (1, 3),
    "eval_easy": (1, 3),
    "medium": (4, 6),
    "train_medium": (4, 6),
    "eval_medium": (4, 6),
    "hard": (7, 10),
    "train_hard": (7, 10),
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
    reward_style: str
    initial_sticker_accuracy: float
    initial_piece_accuracy: float
    move_count: int = 0
    illegal_moves: int = 0
    tool_calls: int = 0
    rotate_call_count: int = 0
    inspect_call_count: int = 0
    finish_call_count: int = 0
    finished: bool = False
    last_move: str = "none"
    first_rotate: Move | None = None
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


async def rotate_call_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.rotate_call_count if rollout else 0)


async def inspect_call_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.inspect_call_count if rollout else 0)


async def finish_call_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.finish_call_count if rollout else 0)


async def action_taken(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(bool(rollout and rollout.move_count > 0))


async def first_rotate_correct(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate or not rollout.inverse_solution:
        return 0.0
    return float(rollout.first_rotate == rollout.inverse_solution[0])


async def first_rotate_face_correct(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate or not rollout.inverse_solution:
        return 0.0
    return float(rollout.first_rotate[0] == rollout.inverse_solution[0][0])


async def first_rotate_direction_correct(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate or not rollout.inverse_solution:
        return 0.0
    return float(rollout.first_rotate[1] == rollout.inverse_solution[0][1])


async def reward_style(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(bool(rollout and rollout.reward_style == "action_gated_dense"))


async def initial_sticker_accuracy(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return rollout.initial_sticker_accuracy if rollout else 0.0


async def initial_piece_accuracy(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return rollout.initial_piece_accuracy if rollout else 0.0


async def action_gated_dense_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout:
        return 0.0
    if rollout.solved():
        return 1.0
    if rollout.move_count == 0:
        return 0.0

    sticker_delta = rollout.puzzle.sticker_accuracy() - rollout.initial_sticker_accuracy
    piece_delta = rollout.puzzle.piece_accuracy() - rollout.initial_piece_accuracy
    progress_delta = 0.7 * sticker_delta + 0.3 * piece_delta
    return max(0.0, min(0.4, progress_delta))


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
        initial_sticker = puzzle.sticker_accuracy()
        initial_piece = puzzle.piece_accuracy()
        state["megaminx"] = RolloutMegaminx(
            puzzle=puzzle,
            scramble=scramble,
            inverse_solution=inverse_solution,
            scramble_depth=int(task["scramble_depth"]),
            move_budget=int(task["move_budget"]),
            reward_style=task.get("reward_style", "dense"),
            initial_sticker_accuracy=initial_sticker,
            initial_piece_accuracy=initial_piece,
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
        parsed_action = _parse_message_tool_action(messages[-1])
        if parsed_action and not getattr(messages[-1], "tool_calls", None):
            tool_name, tool_args = parsed_action
            tool_args = self.update_tool_args(tool_name, tool_args, messages, state, **kwargs)
            try:
                tool_messages = [await self.call_tool(tool_name, tool_args, "text-tool-0")]
            except Exception as error:
                rollout = _rollout_from_state(state)
                if rollout is not None:
                    rollout.illegal_moves += 1
                tool_messages = [
                    ToolMessage(
                        role="tool",
                        content=self.error_formatter(error),
                        tool_call_id="text-tool-0",
                    )
                ]
        else:
            tool_messages = await super().env_response(messages, state, **kwargs)
        rollout = _rollout_from_state(state)
        if rollout and (rollout.solved() or rollout.finished or rollout.exhausted()):
            state["final_env_response"] = tool_messages
        return tool_messages

    @vf.stop
    async def no_tools_called(self, state: vf.State) -> bool:
        if len(state["trajectory"]) == 0:
            return False
        last_message = state["trajectory"][-1]["completion"][-1]
        is_assistant_message = getattr(last_message, "role", None) == "assistant"
        has_tool_calls = bool(getattr(last_message, "tool_calls", None))
        if not is_assistant_message or has_tool_calls:
            return False
        return _parse_message_tool_action(last_message) is None

    async def rotate(self, face: str, direction: str, rollout: RolloutMegaminx) -> str:
        """Rotate one Megaminx face.

        Args:
            face: One face label from A through L.
            direction: cw for clockwise or ccw for counterclockwise, viewed from outside the face.

        Returns:
            Updated puzzle observation after the move.
        """
        rollout.tool_calls += 1
        rollout.rotate_call_count += 1
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
        if rollout.first_rotate is None:
            rollout.first_rotate = (face, direction)
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
        rollout.inspect_call_count += 1
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
        rollout.finish_call_count += 1
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
    reward_style: str = "dense",
    prompt_style: str = "default",
) -> vf.Environment:
    _validate_styles(reward_style, prompt_style)
    dataset = build_dataset(
        split=split,
        min_depth=min_depth,
        max_depth=max_depth,
        num_examples=num_examples,
        seed=seed,
        max_turns=max_turns,
        reward_style=reward_style,
        prompt_style=prompt_style,
    )
    rubric = _build_rubric(reward_style)
    global_max_turns = _resolve_global_max_turns(split, min_depth, max_depth, max_turns)
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
            "reward_style": reward_style,
            "prompt_style": prompt_style,
        },
    )


def build_dataset(
    split: str,
    min_depth: int,
    max_depth: int,
    num_examples: int,
    seed: int,
    max_turns: int | None,
    reward_style: str = "dense",
    prompt_style: str = "default",
) -> Dataset:
    _validate_styles(reward_style, prompt_style)
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
                reward_style=reward_style,
                prompt_style=prompt_style,
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
    reward_style: str,
    prompt_style: str,
) -> dict[str, Any]:
    prompt = [
        {
            "role": "user",
            "content": _build_prompt(index, split, depth, move_budget, puzzle, prompt_style),
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
        "reward_style": reward_style,
        "prompt_style": prompt_style,
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


def _resolve_global_max_turns(
    split: str,
    min_depth: int,
    max_depth: int,
    max_turns: int | None,
) -> int:
    if max_turns is not None:
        return max_turns
    _, resolved_max_depth = _resolve_depths(split, min_depth, max_depth)
    return min(32, max(8, 2 * resolved_max_depth + 6))


def _build_rubric(reward_style_name: str) -> vf.Rubric:
    metric_funcs = [
        illegal_move_count,
        move_count,
        solved_rate,
        scramble_depth,
        tool_call_count,
        rotate_call_count,
        inspect_call_count,
        finish_call_count,
        action_taken,
        first_rotate_correct,
        first_rotate_face_correct,
        first_rotate_direction_correct,
        reward_style,
        initial_sticker_accuracy,
        initial_piece_accuracy,
    ]
    if reward_style_name == "dense":
        return vf.Rubric(
            funcs=[
                solved_reward,
                sticker_accuracy,
                piece_accuracy,
                efficiency_if_solved,
                *metric_funcs,
            ],
            weights=[0.60, 0.25, 0.10, 0.05, *([0.0] * len(metric_funcs))],
        )
    return vf.Rubric(
        funcs=[
            action_gated_dense_reward,
            sticker_accuracy,
            piece_accuracy,
            efficiency_if_solved,
            *metric_funcs,
        ],
        weights=[1.0, 0.0, 0.0, 0.0, *([0.0] * len(metric_funcs))],
    )


def _validate_styles(reward_style: str, prompt_style: str) -> None:
    if reward_style not in REWARD_STYLES:
        expected = ", ".join(sorted(REWARD_STYLES))
        raise ValueError(f"reward_style must be one of {expected}, got {reward_style!r}")
    if prompt_style not in PROMPT_STYLES:
        expected = ", ".join(sorted(PROMPT_STYLES))
        raise ValueError(f"prompt_style must be one of {expected}, got {prompt_style!r}")


def _parse_text_tool_action(content: Any) -> tuple[str, dict[str, Any]] | None:
    text = _content_to_text(content).strip()
    if not text:
        return None
    for start in (index for index, char in enumerate(text) if char in "[{"):
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name") or parsed.get("tool") or parsed.get("tool_name")
        args = parsed.get("parameters") or parsed.get("arguments") or parsed.get("args")
        if name is None and {"face", "direction"} <= set(parsed):
            name = "rotate"
            args = {"face": parsed["face"], "direction": parsed["direction"]}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if isinstance(name, str) and isinstance(args, dict):
            return name, args
    return None


def _parse_message_tool_action(message: Any) -> tuple[str, dict[str, Any]] | None:
    return _parse_text_tool_action(getattr(message, "content", None)) or _parse_text_tool_action(
        getattr(message, "reasoning_content", None)
    )


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _build_prompt(
    index: int,
    split: str,
    depth: int,
    move_budget: int,
    puzzle: MegaminxPuzzle,
    prompt_style: str,
) -> str:
    lines = [
        f"Example: {split}-{index:05d}",
        f"Scramble depth: {depth}",
        f"Move budget: {move_budget}",
        "Goal: solve the Megaminx so every face contains only its own label.",
        "Legal moves: rotate any face A-L with direction cw or ccw.",
        "Use rotate(face, direction), inspect(face), and finish().",
        _curriculum_hint(depth),
    ]
    if prompt_style == "action_first":
        lines.extend(
            [
                "Action-first instruction: the first assistant turn must be a rotate tool call.",
                "Plain text before a rotate tool call receives zero reward and ends the rollout.",
                "Output no explanation before the first rotate tool call.",
            ]
        )
        if depth == 1:
            lines.append(
                "For this one-turn scramble, inspect is unnecessary; choose exactly one rotate call."
            )
        else:
            lines.append("Inspect only after making at least one rotate call.")
    lines.extend(["Initial observation:", puzzle.render_net()])
    return "\n".join(lines)


def _curriculum_hint(depth: int) -> str:
    if depth == 1:
        return (
            "Curriculum hint: this is a one-turn scramble. Exactly one legal rotate "
            "action solves it if you identify the scrambled face and inverse direction."
        )
    return (
        "Curriculum hint: the puzzle was scrambled by a short sequence of face turns; "
        "reverse progress is measured by solvedness, sticker accuracy, and piece accuracy."
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
