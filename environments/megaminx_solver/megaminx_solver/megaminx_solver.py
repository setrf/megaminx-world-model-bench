from __future__ import annotations

import json
from dataclasses import dataclass
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
    POSITIONS_PER_FACE,
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

REWARD_STYLES = {
    "dense",
    "action_gated_dense",
    "action_gated_curriculum",
    "action_gated_overlap",
    "action_gated_direction",
    "action_gated_exact_direction",
    "action_gated_binary_direction",
    "action_gated_strict_shaped_direction",
}
PROMPT_STYLES = {
    "default",
    "action_first",
    "direct_json_action",
    "choice_json_action",
    "topology_choice_json_action",
    "sensor_choice_json_action",
    "sensor_match_json_action",
    "sensor_indexed_match_json_action",
    "sensor_candidate_strips_json_action",
    "stage_face_hint_direction_json_action",
    "stage_direction_flow_json_action",
    "stage_direction_flow_reasoned_json_action",
    "stage_solve_direction_flow_json_action",
    "native_action",
    "topology_native_tool",
    "sensor_native_tool",
    "sensor_match_native_tool",
    "stage_direction_flow_native_tool",
    "stage_solve_direction_flow_native_tool",
}

ACTION_FIRST_PROMPT_STYLES = {
    "action_first",
    "direct_json_action",
    "choice_json_action",
    "topology_choice_json_action",
    "sensor_choice_json_action",
    "sensor_match_json_action",
    "sensor_indexed_match_json_action",
    "sensor_candidate_strips_json_action",
    "stage_face_hint_direction_json_action",
    "stage_direction_flow_json_action",
    "stage_direction_flow_reasoned_json_action",
    "stage_solve_direction_flow_json_action",
    "native_action",
    "topology_native_tool",
    "sensor_native_tool",
    "sensor_match_native_tool",
    "stage_direction_flow_native_tool",
    "stage_solve_direction_flow_native_tool",
}
JSON_ACTION_PROMPT_STYLES = {
    "direct_json_action",
    "choice_json_action",
    "topology_choice_json_action",
    "sensor_choice_json_action",
    "sensor_match_json_action",
    "sensor_indexed_match_json_action",
    "sensor_candidate_strips_json_action",
    "stage_face_hint_direction_json_action",
    "stage_direction_flow_json_action",
    "stage_direction_flow_reasoned_json_action",
    "stage_solve_direction_flow_json_action",
}
REASONED_JSON_PROMPT_STYLES = {
    "stage_direction_flow_reasoned_json_action",
}
CHOICE_JSON_PROMPT_STYLES = {
    "choice_json_action",
    "topology_choice_json_action",
    "sensor_choice_json_action",
    "sensor_match_json_action",
    "sensor_indexed_match_json_action",
    "sensor_candidate_strips_json_action",
    "stage_face_hint_direction_json_action",
    "stage_direction_flow_json_action",
    "stage_direction_flow_reasoned_json_action",
    "stage_solve_direction_flow_json_action",
}
TOPOLOGY_PROMPT_STYLES = {
    "topology_choice_json_action",
    "sensor_choice_json_action",
    "sensor_match_json_action",
    "sensor_indexed_match_json_action",
    "sensor_candidate_strips_json_action",
    "stage_face_hint_direction_json_action",
    "topology_native_tool",
    "sensor_native_tool",
    "sensor_match_native_tool",
}
SENSOR_PROMPT_STYLES = {
    "sensor_choice_json_action",
    "sensor_match_json_action",
    "sensor_indexed_match_json_action",
    "sensor_candidate_strips_json_action",
    "stage_face_hint_direction_json_action",
    "stage_direction_flow_json_action",
    "stage_direction_flow_reasoned_json_action",
    "stage_solve_direction_flow_json_action",
    "stage_direction_flow_native_tool",
    "stage_solve_direction_flow_native_tool",
    "sensor_native_tool",
    "sensor_match_native_tool",
}
MATCH_TABLE_PROMPT_STYLES = {
    "sensor_match_json_action",
    "sensor_indexed_match_json_action",
    "sensor_candidate_strips_json_action",
    "stage_face_hint_direction_json_action",
    "sensor_match_native_tool",
}
INDEXED_DIRECTION_PROMPT_STYLES = {
    "sensor_indexed_match_json_action",
    "sensor_candidate_strips_json_action",
    "stage_face_hint_direction_json_action",
}
CANDIDATE_STRIPS_PROMPT_STYLES = {
    "sensor_candidate_strips_json_action",
    "stage_face_hint_direction_json_action",
}
FACE_HINT_PROMPT_STYLES = {
    "stage_face_hint_direction_json_action",
    "stage_direction_flow_json_action",
    "stage_direction_flow_reasoned_json_action",
    "stage_direction_flow_native_tool",
    "stage_solve_direction_flow_json_action",
    "stage_solve_direction_flow_native_tool",
}
DIRECTION_FLOW_PROMPT_STYLES = {
    "stage_direction_flow_json_action",
    "stage_direction_flow_reasoned_json_action",
    "stage_direction_flow_native_tool",
}
SOLVE_DIRECTION_FLOW_PROMPT_STYLES = {
    "stage_solve_direction_flow_json_action",
    "stage_solve_direction_flow_native_tool",
}
NATIVE_TOOL_PROMPT_STYLES = {
    "native_action",
    "topology_native_tool",
    "sensor_native_tool",
    "sensor_match_native_tool",
    "stage_direction_flow_native_tool",
    "stage_solve_direction_flow_native_tool",
}
REASONED_NATIVE_TOOL_PROMPT_STYLES = {
    "stage_direction_flow_native_tool",
    "stage_solve_direction_flow_native_tool",
}

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
    initial_affected_faces: tuple[str, ...]
    move_count: int = 0
    illegal_moves: int = 0
    tool_calls: int = 0
    native_tool_call_count: int = 0
    text_tool_action_count: int = 0
    private_text_action_count: int = 0
    tool_parse_error_count: int = 0
    tool_call_error_count: int = 0
    protocol_violation_count: int = 0
    rotate_call_count: int = 0
    inspect_call_count: int = 0
    finish_call_count: int = 0
    finished: bool = False
    last_move: str = "none"
    first_rotate: Move | None = None

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


@dataclass(frozen=True)
class ParsedTextToolAction:
    action: tuple[str, dict[str, Any]] | None = None
    error: str | None = None
    source: str = "visible"


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


async def native_tool_call_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.native_tool_call_count if rollout else 0)


async def text_tool_action_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.text_tool_action_count if rollout else 0)


async def private_text_action_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.private_text_action_count if rollout else 0)


async def tool_parse_error_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.tool_parse_error_count if rollout else 0)


async def tool_call_error_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.tool_call_error_count if rollout else 0)


async def protocol_violation_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.protocol_violation_count if rollout else 0)


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


async def first_rotate_neighbor_overlap(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate or not rollout.initial_affected_faces:
        return 0.0
    ring = set(rollout.puzzle.topology.neighbor_rings[rollout.first_rotate[0]])
    affected = set(rollout.initial_affected_faces)
    return len(ring & affected) / len(affected)


async def reward_style(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout:
        return 0.0
    return {
        "dense": 0.0,
        "action_gated_dense": 1.0,
        "action_gated_curriculum": 2.0,
        "action_gated_overlap": 3.0,
        "action_gated_direction": 4.0,
        "action_gated_exact_direction": 5.0,
        "action_gated_binary_direction": 6.0,
        "action_gated_strict_shaped_direction": 7.0,
    }.get(rollout.reward_style, 0.0)


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


async def action_gated_curriculum_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout:
        return 0.0
    if rollout.solved():
        return 1.0
    if rollout.move_count == 0 or rollout.first_rotate is None:
        return 0.0

    reward = 0.02
    if rollout.inverse_solution:
        target_face, target_direction = rollout.inverse_solution[0]
        first_face, first_direction = rollout.first_rotate
        if first_face == target_face:
            reward += 0.35
            if first_direction == target_direction:
                reward += 0.05

    sticker_delta = rollout.puzzle.sticker_accuracy() - rollout.initial_sticker_accuracy
    piece_delta = rollout.puzzle.piece_accuracy() - rollout.initial_piece_accuracy
    progress_delta = max(0.0, 0.7 * sticker_delta + 0.3 * piece_delta)
    reward += min(0.30, progress_delta)
    return min(0.65, reward)


async def action_gated_overlap_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout:
        return 0.0
    if rollout.solved():
        return 1.0
    if rollout.move_count == 0 or rollout.first_rotate is None:
        return 0.0

    reward = 0.02
    overlap = await first_rotate_neighbor_overlap(state)
    reward += 0.30 * overlap

    if rollout.inverse_solution:
        target_face, target_direction = rollout.inverse_solution[0]
        first_face, first_direction = rollout.first_rotate
        if first_face == target_face:
            reward += 0.15
            if first_direction == target_direction:
                reward += 0.05

    sticker_delta = rollout.puzzle.sticker_accuracy() - rollout.initial_sticker_accuracy
    piece_delta = rollout.puzzle.piece_accuracy() - rollout.initial_piece_accuracy
    progress_delta = max(0.0, 0.7 * sticker_delta + 0.3 * piece_delta)
    reward += min(0.10, progress_delta)
    return min(0.65, reward)


async def action_gated_direction_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout:
        return 0.0
    if rollout.solved():
        return 1.0
    if rollout.move_count == 0 or rollout.first_rotate is None:
        return 0.0

    reward = 0.02
    if rollout.inverse_solution:
        target_face, target_direction = rollout.inverse_solution[0]
        first_face, first_direction = rollout.first_rotate
        if first_face == target_face:
            reward += 0.20
        if first_direction == target_direction:
            reward += 0.08

    sticker_delta = rollout.puzzle.sticker_accuracy() - rollout.initial_sticker_accuracy
    piece_delta = rollout.puzzle.piece_accuracy() - rollout.initial_piece_accuracy
    progress_delta = max(0.0, 0.7 * sticker_delta + 0.3 * piece_delta)
    reward += min(0.05, progress_delta)
    return min(0.35, reward)


async def action_gated_exact_direction_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout:
        return 0.0
    if rollout.solved():
        return 1.0
    if rollout.move_count == 0 or rollout.first_rotate is None:
        return 0.0

    reward = 0.02
    if rollout.inverse_solution:
        target_face, target_direction = rollout.inverse_solution[0]
        first_face, first_direction = rollout.first_rotate
        if first_face == target_face:
            reward += 0.13
            if first_direction == target_direction:
                reward += 0.20

    return min(0.35, reward)


async def action_gated_binary_direction_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    is_clean_single_rotate = _is_clean_single_rotate_attempt(rollout) and (
        rollout.solved()
        and rollout.first_rotate == rollout.inverse_solution[0]
    )
    return float(is_clean_single_rotate)


async def action_gated_strict_shaped_direction_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    if not _is_clean_single_rotate_attempt(rollout):
        return 0.0

    target_face, target_direction = rollout.inverse_solution[0]
    first_face, first_direction = rollout.first_rotate
    if rollout.solved() and rollout.first_rotate == rollout.inverse_solution[0]:
        return 1.0

    reward = 0.05
    if first_face == target_face:
        reward += 0.35
        if first_direction == target_direction:
            reward += 0.45
    elif first_direction == target_direction:
        reward += 0.05

    sticker_delta = rollout.puzzle.sticker_accuracy() - rollout.initial_sticker_accuracy
    piece_delta = rollout.puzzle.piece_accuracy() - rollout.initial_piece_accuracy
    progress_delta = max(0.0, 0.7 * sticker_delta + 0.3 * piece_delta)
    reward += min(0.15, progress_delta)
    return min(0.95, reward)


def _is_clean_single_rotate_attempt(rollout: RolloutMegaminx) -> bool:
    action_attempts = rollout.native_tool_call_count + rollout.text_tool_action_count
    return (
        rollout.illegal_moves == 0
        and rollout.protocol_violation_count == 0
        and rollout.tool_parse_error_count == 0
        and rollout.tool_call_error_count == 0
        and rollout.move_count == 1
        and rollout.rotate_call_count == 1
        and rollout.inspect_call_count == 0
        and rollout.finish_call_count == 0
        and action_attempts == 1
        and rollout.first_rotate is not None
    )


class MegaminxEnv(vf.StatefulToolEnv):
    def __init__(
        self,
        topology: MegaminxTopology | None = None,
        allow_text_tool_actions: bool = False,
        **kwargs: Any,
    ):
        self.topology = topology or DEFAULT_TOPOLOGY
        self.allow_text_tool_actions = allow_text_tool_actions
        super().__init__(tools=[], error_formatter=lambda error: f"Tool error: {error}", **kwargs)
        self.add_tool(self.rotate, args_to_skip=["rollout"])
        self.add_tool(self.inspect, args_to_skip=["rollout"])
        self.add_tool(self.finish, args_to_skip=["rollout"])
        self._constrain_tool_schemas()

    def _constrain_tool_schemas(self) -> None:
        for tool in self.tool_defs:
            properties = tool.parameters.get("properties", {})
            if tool.name == "rotate":
                properties.get("face", {})["enum"] = list(FACES)
                properties.get("direction", {})["enum"] = list(DIRECTIONS)
            elif tool.name == "inspect":
                properties.get("face", {})["enum"] = [*FACES, "all"]

    async def setup_state(self, state: vf.State) -> None:
        await super().setup_state(state)
        task = _task_from_state(state)
        scramble = _decode_moves(task["scramble"])
        inverse_solution = _decode_moves(task["inverse_solution"])
        puzzle = MegaminxPuzzle.solved(self.topology)
        puzzle.apply_moves(scramble)
        initial_sticker = puzzle.sticker_accuracy()
        initial_piece = puzzle.piece_accuracy()
        initial_affected = _affected_faces(puzzle)
        state["megaminx"] = RolloutMegaminx(
            puzzle=puzzle,
            scramble=scramble,
            inverse_solution=inverse_solution,
            scramble_depth=int(task["scramble_depth"]),
            move_budget=int(task["move_budget"]),
            reward_style=task.get("reward_style", "dense"),
            initial_sticker_accuracy=initial_sticker,
            initial_piece_accuracy=initial_piece,
            initial_affected_faces=initial_affected,
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
        last_message = messages[-1]
        native_tool_calls = getattr(last_message, "tool_calls", None) or []
        has_native_tool_calls = bool(native_tool_calls)
        rollout = _rollout_from_state(state)
        if rollout and has_native_tool_calls:
            rollout.native_tool_call_count += len(native_tool_calls)
            if _message_visible_content_text(last_message):
                rollout.protocol_violation_count += 1
        parsed_result = (
            _parse_message_tool_action(
                last_message,
                include_private_reasoning=not _requires_visible_text_tool_action(state),
            )
            if self.allow_text_tool_actions and not has_native_tool_calls
            else None
        )
        if parsed_result and parsed_result.action and not has_native_tool_calls:
            if rollout is not None:
                rollout.text_tool_action_count += 1
                if parsed_result.source == "private":
                    rollout.private_text_action_count += 1
            tool_name, tool_args = parsed_result.action
            illegal_before = rollout.illegal_moves if rollout is not None else 0
            tool_args = self.update_tool_args(tool_name, tool_args, messages, state, **kwargs)
            try:
                tool_messages = [await self.call_tool(tool_name, tool_args, "text-tool-0")]
            except Exception as error:
                rollout = _rollout_from_state(state)
                if rollout is not None:
                    rollout.tool_call_error_count += 1
                    if rollout.illegal_moves == illegal_before:
                        rollout.illegal_moves += 1
                tool_messages = [
                    ToolMessage(
                        role="tool",
                        content=self.error_formatter(error),
                        tool_call_id="text-tool-0",
                    )
                ]
        elif parsed_result and parsed_result.error and not has_native_tool_calls:
            if rollout is not None:
                rollout.text_tool_action_count += 1
                if parsed_result.source == "private":
                    rollout.private_text_action_count += 1
                rollout.tool_parse_error_count += 1
                rollout.protocol_violation_count += 1
                rollout.illegal_moves += 1
            tool_messages = [
                ToolMessage(
                    role="tool",
                    content=self.error_formatter(ValueError(parsed_result.error)),
                    tool_call_id="text-tool-0",
                )
            ]
        elif not has_native_tool_calls:
            tool_messages = []
        else:
            tool_messages = await self._native_tool_responses(
                native_tool_calls,
                messages,
                state,
                **kwargs,
            )
        if rollout and (rollout.solved() or rollout.finished or rollout.exhausted()):
            state["final_env_response"] = tool_messages
        return tool_messages

    async def _native_tool_responses(
        self,
        tool_calls: Sequence[Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> vf.Messages:
        rollout = _rollout_from_state(state)
        tool_messages: list[ToolMessage] = []
        for index, tool_call in enumerate(tool_calls):
            tool_call_id = _tool_call_field(tool_call, "id") or f"native-tool-{index}"
            try:
                tool_name = _tool_call_field(tool_call, "name")
                raw_args = _tool_call_field(tool_call, "arguments")
                if not isinstance(tool_name, str):
                    raise ValueError("native tool call is missing a string name")
                if isinstance(raw_args, str):
                    tool_args = json.loads(raw_args)
                elif isinstance(raw_args, dict):
                    tool_args = raw_args
                else:
                    raise ValueError("native tool call arguments must be a JSON object")
                if not isinstance(tool_args, dict):
                    raise ValueError("native tool call arguments must decode to an object")
            except Exception as error:
                if rollout is not None:
                    rollout.tool_parse_error_count += 1
                    rollout.illegal_moves += 1
                tool_messages.append(
                    ToolMessage(
                        role="tool",
                        content=self.error_formatter(error),
                        tool_call_id=str(tool_call_id),
                    )
                )
                continue

            illegal_before = rollout.illegal_moves if rollout is not None else 0
            tool_args = self.update_tool_args(tool_name, tool_args, messages, state, **kwargs)
            try:
                tool_messages.append(await self.call_tool(tool_name, tool_args, str(tool_call_id)))
            except Exception as error:
                rollout = _rollout_from_state(state)
                if rollout is not None:
                    rollout.tool_call_error_count += 1
                    if rollout.illegal_moves == illegal_before:
                        rollout.illegal_moves += 1
                tool_messages.append(
                    ToolMessage(
                        role="tool",
                        content=self.error_formatter(error),
                        tool_call_id=str(tool_call_id),
                    )
                )
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
        if not self.allow_text_tool_actions:
            return True
        include_private_reasoning = not (
            self.allow_text_tool_actions and _requires_visible_text_tool_action(state)
        )
        parsed_result = _parse_message_tool_action(
            last_message,
            include_private_reasoning=include_private_reasoning,
        )
        return parsed_result is None

    async def rotate(self, face: str, direction: str, rollout: RolloutMegaminx) -> str:
        """Rotate one Megaminx face.

        Args:
            face: One face label from A through L.
            direction: cw advances neighbor strips one slot through the printed ring; ccw reverses them.

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
        if not isinstance(face, str):
            rollout.illegal_moves += 1
            return (
                f"Illegal inspect target: expected A-L or all, got {type(face).__name__}.\n"
                + rollout.observation()
            )
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
    move_budget: int | None = None,
    reward_style: str = "dense",
    prompt_style: str = "default",
    allow_text_tool_actions: bool | None = None,
) -> vf.Environment:
    _validate_styles(reward_style, prompt_style)
    if allow_text_tool_actions is None:
        allow_text_tool_actions = prompt_style in JSON_ACTION_PROMPT_STYLES
    dataset = build_dataset(
        split=split,
        min_depth=min_depth,
        max_depth=max_depth,
        num_examples=num_examples,
        seed=seed,
        max_turns=max_turns,
        move_budget=move_budget,
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
        allow_text_tool_actions=allow_text_tool_actions,
        env_id="megaminx-solver",
        env_args={
            "split": split,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "num_examples": num_examples,
            "seed": seed,
            "max_turns": max_turns,
            "move_budget": move_budget,
            "reward_style": reward_style,
            "prompt_style": prompt_style,
            "allow_text_tool_actions": allow_text_tool_actions,
        },
    )


def build_dataset(
    split: str,
    min_depth: int,
    max_depth: int,
    num_examples: int,
    seed: int,
    max_turns: int | None,
    move_budget: int | None = None,
    reward_style: str = "dense",
    prompt_style: str = "default",
) -> Dataset:
    _validate_styles(reward_style, prompt_style)
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    min_depth, max_depth = _resolve_depths(split, min_depth, max_depth)
    if min_depth < 0 or max_depth < min_depth:
        raise ValueError("Expected 0 <= min_depth <= max_depth")
    if prompt_style in FACE_HINT_PROMPT_STYLES and (min_depth, max_depth) != (1, 1):
        raise ValueError("stage face-hint prompt styles are only defined for depth-1 splits")
    if max_turns is not None and max_turns <= 0:
        raise ValueError("max_turns must be positive when provided")
    if move_budget is not None and move_budget <= 0:
        raise ValueError("move_budget must be positive when provided")

    rng = Random(seed)
    rows: list[dict[str, Any]] = []
    depth_span = max_depth - min_depth + 1
    depth_occurrences: dict[int, int] = {}
    depth1_moves = _balanced_depth1_moves(seed)
    for index in range(num_examples):
        depth = min_depth + (index % depth_span)
        depth_occurrences[depth] = depth_occurrences.get(depth, 0) + 1
        if depth == 1:
            occurrence = depth_occurrences[depth] - 1
            scramble = [depth1_moves[occurrence % len(depth1_moves)]]
        else:
            scramble = generate_scramble(depth, rng)
        inverse_solution = inverse_moves(scramble)
        row_move_budget = move_budget if move_budget is not None else (
            max_turns if max_turns is not None else min(32, 2 * depth + 4)
        )
        puzzle = MegaminxPuzzle.solved(DEFAULT_TOPOLOGY)
        puzzle.apply_moves(scramble)
        rows.append(
            _build_row(
                index=index,
                split=split,
                depth=depth,
                move_budget=int(row_move_budget),
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
        native_tool_call_count,
        text_tool_action_count,
        private_text_action_count,
        tool_parse_error_count,
        tool_call_error_count,
        protocol_violation_count,
        rotate_call_count,
        inspect_call_count,
        finish_call_count,
        action_taken,
        first_rotate_correct,
        first_rotate_face_correct,
        first_rotate_direction_correct,
        first_rotate_neighbor_overlap,
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
    reward_func = (
        action_gated_overlap_reward
        if reward_style_name == "action_gated_overlap"
        else action_gated_exact_direction_reward
        if reward_style_name == "action_gated_exact_direction"
        else action_gated_binary_direction_reward
        if reward_style_name == "action_gated_binary_direction"
        else action_gated_strict_shaped_direction_reward
        if reward_style_name == "action_gated_strict_shaped_direction"
        else action_gated_direction_reward
        if reward_style_name == "action_gated_direction"
        else action_gated_curriculum_reward
        if reward_style_name == "action_gated_curriculum"
        else action_gated_dense_reward
    )
    return vf.Rubric(
        funcs=[
            reward_func,
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


def _tool_action_from_json(parsed: Any) -> tuple[str, dict[str, Any]] | None:
    if isinstance(parsed, list) and len(parsed) == 1:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return None
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
    if isinstance(name, str):
        return name, args if isinstance(args, dict) else {}
    return None


def _parse_text_tool_action_result(content: Any) -> ParsedTextToolAction | None:
    text = _content_to_text(content).strip()
    if not text or text[0] not in "[{":
        return None
    try:
        parsed, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as error:
        return ParsedTextToolAction(error=f"Could not parse JSON tool action: {error.msg}")
    if text[end:].strip():
        return ParsedTextToolAction(error="Unexpected text after JSON tool action")
    action = _tool_action_from_json(parsed)
    if action is None:
        return ParsedTextToolAction(error="Expected exactly one JSON tool action")
    return ParsedTextToolAction(action=action)


def _tool_call_field(tool_call: Any, field: str) -> Any:
    payload = tool_call
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if isinstance(payload, dict):
        if field in payload:
            return payload[field]
        function = payload.get("function")
        if isinstance(function, dict):
            return function.get(field)
        return None
    value = getattr(payload, field, None)
    if value is not None:
        return value
    function = getattr(payload, "function", None)
    if function is not None:
        return getattr(function, field, None)
    return None


def _parse_text_tool_action(content: Any) -> tuple[str, dict[str, Any]] | None:
    result = _parse_text_tool_action_result(content)
    return result.action if result else None


def _message_visible_content_text(message: Any) -> str:
    return _content_to_text(getattr(message, "content", None)).strip()


def _prompt_style_from_state(state: vf.State) -> str:
    try:
        task = _task_from_state(state)
    except (KeyError, json.JSONDecodeError, TypeError):
        return ""
    prompt_style = task.get("prompt_style", "")
    return prompt_style if isinstance(prompt_style, str) else ""


def _requires_visible_text_tool_action(state: vf.State) -> bool:
    return _prompt_style_from_state(state) in REASONED_JSON_PROMPT_STYLES


def _parse_message_tool_action(
    message: Any,
    include_private_reasoning: bool = True,
) -> ParsedTextToolAction | None:
    visible_content = getattr(message, "content", None)
    parsed = _parse_text_tool_action_result(visible_content)
    if parsed:
        return parsed
    if _message_visible_content_text(message):
        return ParsedTextToolAction(error="Visible content must be exactly one JSON tool action")
    payloads: list[Any] = []
    if include_private_reasoning:
        payloads.extend(
            [
                getattr(message, "reasoning_content", None),
                getattr(message, "reasoning", None),
                getattr(message, "thinking", None),
                getattr(message, "thinking_blocks", None),
            ]
        )
        model_extra = getattr(message, "model_extra", None)
        if isinstance(model_extra, dict):
            payloads.extend(
                model_extra.get(key)
                for key in ("reasoning", "thinking", "reasoning_content", "thinking_blocks")
            )
    for payload in payloads:
        parsed = _parse_private_tool_action_result(payload)
        if parsed:
            return ParsedTextToolAction(
                action=parsed.action,
                error=parsed.error,
                source="private",
            )
    return None


def _parse_private_tool_action_result(payload: Any) -> ParsedTextToolAction | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        action = _tool_action_from_json(payload)
        return ParsedTextToolAction(action=action) if action else None
    if isinstance(payload, list):
        action = _tool_action_from_json(payload)
        if action:
            return ParsedTextToolAction(action=action)
    return _parse_text_tool_action_result(payload)


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
                for key in ("text", "content", "reasoning", "thinking"):
                    text = part.get(key)
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
    face_hint = (
        _face_hint_from_affected_faces(puzzle)
        if prompt_style in FACE_HINT_PROMPT_STYLES
        else None
    )
    lines = [
        f"Example: {split}-{index:05d}",
        f"Scramble depth: {depth}",
        f"Move budget: {move_budget}",
        "Goal: solve the Megaminx so every face contains only its own label.",
        "Legal moves: rotate any face A-L with direction cw or ccw.",
        "Use rotate(face, direction), inspect(face), and finish().",
        _curriculum_hint(depth),
    ]
    if prompt_style in ACTION_FIRST_PROMPT_STYLES:
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
    if prompt_style in NATIVE_TOOL_PROMPT_STYLES:
        if prompt_style in REASONED_NATIVE_TOOL_PROMPT_STYLES:
            lines.extend(
                [
                    "Native tool mode: compare the direction-flow table, then call rotate directly.",
                    "You may reason internally, but the final assistant action must be a rotate tool call.",
                    "Do not write JSON or prose in the final assistant content.",
                    "The tool schema constrains face to A-L and direction to cw or ccw.",
                ]
            )
        else:
            lines.extend(
                [
                    "Native tool mode: call the rotate tool directly.",
                    "Do not write JSON, prose, or analysis before the tool call.",
                    "The tool schema constrains face to A-L and direction to cw or ccw.",
                ]
            )
    if prompt_style in JSON_ACTION_PROMPT_STYLES:
        if prompt_style in REASONED_JSON_PROMPT_STYLES:
            lines.extend(
                [
                    "Reasoned direct-action mode: compare the direction-flow table before choosing.",
                    "You may reason internally, but the final assistant content must be exactly one JSON object.",
                    "Do not include prose before or after the JSON object.",
                ]
            )
        else:
            lines.extend(["/no_think", "Direct-action mode: do not reason, analyze, or explain."])
        if prompt_style in CHOICE_JSON_PROMPT_STYLES:
            lines.append("Return exactly one JSON object copied from the legal-action menu below.")
        else:
            lines.extend(
                [
                    "Return exactly one JSON object and no other text:",
                    '{"tool":"rotate","args":{"face":"A","direction":"cw"}}',
                    "Replace A and cw with your chosen legal face and direction.",
                ]
            )
    if prompt_style in SENSOR_PROMPT_STYLES:
        lines.append(_sensor_section(puzzle))
    if prompt_style in TOPOLOGY_PROMPT_STYLES:
        lines.append(_topology_rules_section())
    if prompt_style in MATCH_TABLE_PROMPT_STYLES:
        lines.append(_candidate_neighbor_sets_section())
    if prompt_style in INDEXED_DIRECTION_PROMPT_STYLES:
        lines.append(_indexed_direction_section(puzzle))
    if prompt_style in CANDIDATE_STRIPS_PROMPT_STYLES:
        lines.append(_candidate_strips_section(puzzle))
    if prompt_style in FACE_HINT_PROMPT_STYLES:
        lines.append(_face_hint_section(face_hint))
    if prompt_style in DIRECTION_FLOW_PROMPT_STYLES:
        lines.append(_direction_flow_section(puzzle, face_hint))
    if prompt_style in SOLVE_DIRECTION_FLOW_PROMPT_STYLES:
        lines.append(_solve_direction_flow_section(puzzle, face_hint))
    if prompt_style in CHOICE_JSON_PROMPT_STYLES:
        action_menu = _json_action_menu_for_face(face_hint) if face_hint else _json_action_menu()
        lines.extend(
            [
                "Choose exactly one legal action from this menu and copy it exactly with no bullet:",
                action_menu,
            ]
        )
    lines.extend(["Initial observation:", puzzle.render_net()])
    return "\n".join(lines)


def _curriculum_hint(depth: int) -> str:
    if depth == 1:
        return (
            "Curriculum hint: this is a one-turn scramble. Exactly one legal rotate "
            "action solves it if you identify the scrambled face and solving direction."
        )
    return (
        "Curriculum hint: the puzzle was scrambled by a short sequence of face turns; "
        "reverse progress is measured by solvedness, sticker accuracy, and piece accuracy."
    )


def _json_action_menu() -> str:
    return "\n".join(
        json.dumps({"tool": "rotate", "args": {"face": face, "direction": direction}})
        for face in FACES
        for direction in DIRECTIONS
    )


def _json_action_menu_for_face(face: str) -> str:
    return "\n".join(
        json.dumps({"tool": "rotate", "args": {"face": face, "direction": direction}})
        for direction in DIRECTIONS
    )


def _balanced_depth1_moves(seed: int) -> list[Move]:
    moves = [(face, direction) for face in FACES for direction in DIRECTIONS]
    rng = Random(seed)
    rng.shuffle(moves)
    return moves


def _affected_faces(puzzle: MegaminxPuzzle) -> tuple[str, ...]:
    return tuple(
        face
        for face in FACES
        if any(
            puzzle.stickers[(face, index)] != face for index in range(1, POSITIONS_PER_FACE)
        )
    )


def _candidate_neighbor_sets_section() -> str:
    rows = ["Candidate neighbor sets for face matching:"]
    for face in FACES:
        neighbors = " ".join(sorted(DEFAULT_TOPOLOGY.neighbor_rings[face]))
        rows.append(f"{face}: {neighbors}")
    return "\n".join(rows)


def _sensor_section(puzzle: MegaminxPuzzle) -> str:
    affected = _affected_faces(puzzle)
    lines = [
        "Structured state sensors:",
        "Affected faces: " + (" ".join(affected) if affected else "none"),
    ]
    if affected:
        lines.append("Affected face summaries:")
        lines.extend(puzzle.face_line(face) for face in affected)
    return "\n".join(lines)


def _indexed_direction_section(puzzle: MegaminxPuzzle) -> str:
    affected = _affected_faces(puzzle)
    lines = [
        "Direction index guide:",
        "Face strings are indexed left to right: corners c0..c4 and edges e0..e4.",
        "On any neighbor face N, edge e_i borders ring_N[i].",
        "On any neighbor face N, corner c_i is between ring_N[i] and ring_N[(i+1)%5].",
        (
            "For a candidate turned face X on neighbor N, side = ring_N.index(X); "
            "X's visible moved strip on N is c[(side-1)%5], e[side], c[side]."
        ),
        "Compare those indexed strips around candidate X's ring to infer cw vs ccw.",
        "Indexed affected stickers:",
    ]
    if not affected:
        lines.append("none")
        return "\n".join(lines)

    for face in affected:
        ring = " ".join(DEFAULT_TOPOLOGY.neighbor_rings[face])
        corners = " ".join(
            f"c{side}={puzzle.stickers[(face, 1 + side)]}" for side in range(5)
        )
        edges = " ".join(
            f"e{side}={puzzle.stickers[(face, 6 + side)]}" for side in range(5)
        )
        changed = []
        for side in range(5):
            corner_color = puzzle.stickers[(face, 1 + side)]
            edge_color = puzzle.stickers[(face, 6 + side)]
            if corner_color != face:
                changed.append(f"c{side}={corner_color}")
            if edge_color != face:
                changed.append(f"e{side}={edge_color}")
        lines.append(f"{face}: ring={ring} | {corners} | {edges}")
        lines.append(f"{face} changed: " + (" ".join(changed) if changed else "none"))
    return "\n".join(lines)


def _candidate_strips_section(puzzle: MegaminxPuzzle) -> str:
    lines = [
        "Candidate-local moved strips:",
        (
            "For candidate face X, each row lists X's five neighbors in ring order and "
            "the three indexed stickers on each neighbor that touch X."
        ),
    ]
    for candidate in FACES:
        ring = DEFAULT_TOPOLOGY.neighbor_rings[candidate]
        parts = [f"ring={' '.join(ring)}"]
        for neighbor in ring:
            side = DEFAULT_TOPOLOGY.neighbor_rings[neighbor].index(candidate)
            strip = (
                ("c", (side - 1) % 5, puzzle.stickers[(neighbor, 1 + ((side - 1) % 5))]),
                ("e", side, puzzle.stickers[(neighbor, 6 + side)]),
                ("c", side, puzzle.stickers[(neighbor, 1 + side)]),
            )
            strip_text = " ".join(f"{kind}{index}={color}" for kind, index, color in strip)
            parts.append(f"{neighbor}[{strip_text}]")
        lines.append(f"{candidate}: " + " | ".join(parts))
    return "\n".join(lines)


def _direction_flow_section(puzzle: MegaminxPuzzle, face: str | None) -> str:
    if face is None:
        return "Direction flow table: unavailable because no single hinted face was identified."

    ring = DEFAULT_TOPOLOGY.neighbor_rings[face]
    lines = [
        f"Direction flow table for face {face}:",
        "Ring order: " + " ".join(ring),
        (
            "Each row shows the destination strip touching the hinted face before and after "
            "the scramble, plus the expected source strip for each possible scramble direction."
        ),
        (
            "If current matches source_if_scramble_cw on all rows, the scramble was face:cw; solve with face:ccw. "
            "If current matches source_if_scramble_ccw on all rows, the scramble was face:ccw; solve with face:cw."
        ),
        "destination | before | current | source_if_scramble_cw | source_if_scramble_ccw",
    ]
    for side, destination in enumerate(ring):
        previous = ring[(side - 1) % len(ring)]
        next_face = ring[(side + 1) % len(ring)]
        strip = [puzzle.stickers[position] for position in DEFAULT_TOPOLOGY.side_strip(face, destination)]
        lines.append(
            f"{destination} | before={destination * 3} | current={''.join(strip)} | "
            f"source_if_scramble_cw={previous * 3} | source_if_scramble_ccw={next_face * 3}"
        )
    return "\n".join(lines)


def _solve_direction_flow_section(puzzle: MegaminxPuzzle, face: str | None) -> str:
    if face is None:
        return "Solve-direction flow table: unavailable because no single hinted face was identified."

    ring = DEFAULT_TOPOLOGY.neighbor_rings[face]
    lines = [
        f"Solve-direction flow table for face {face}:",
        "Ring order: " + " ".join(ring),
        (
            "Each row shows the destination strip touching the hinted face, the current strip, "
            "and the visible current pattern expected for each possible solve direction."
        ),
        (
            "Decision rule: choose cw only if current matches expected_current_if_solve_cw on all rows. "
            "Choose ccw only if current matches expected_current_if_solve_ccw on all rows."
        ),
        "destination | solved_target | current | expected_current_if_solve_cw | expected_current_if_solve_ccw",
    ]
    for side, destination in enumerate(ring):
        previous = ring[(side - 1) % len(ring)]
        next_face = ring[(side + 1) % len(ring)]
        strip = [puzzle.stickers[position] for position in DEFAULT_TOPOLOGY.side_strip(face, destination)]
        lines.append(
            f"{destination} | solved_target={destination * 3} | current={''.join(strip)} | "
            f"expected_current_if_solve_cw={next_face * 3} | expected_current_if_solve_ccw={previous * 3}"
        )
    return "\n".join(lines)


def _face_hint_from_affected_faces(puzzle: MegaminxPuzzle) -> str | None:
    affected = set(_affected_faces(puzzle))
    matches = [
        face
        for face in FACES
        if set(DEFAULT_TOPOLOGY.neighbor_rings[face]) == affected
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _face_hint_section(face: str | None) -> str:
    if face is None:
        return (
            "Staged face hint: no single face is identified from affected faces; "
            "use the full topology evidence."
        )
    return "\n".join(
        [
            "Staged face hint: affected faces identify the turned face.",
            f"Use face {face}; infer only whether the solving direction is cw or ccw.",
            f"Only choose rotate actions on face {face}.",
        ]
    )


def _topology_rules_section() -> str:
    lines = [
        "Static topology rules:",
        "Each face lists its five neighbors in ring order.",
        "For a candidate ring X: N0 N1 N2 N3 N4, indices wrap mod 5.",
        "A turned face can still look solved because all stickers on that face share one label.",
        "The evidence is on the five neighboring faces around the turned face.",
        "For one-turn scrambles, the correct face is the face whose five listed neighbors are the affected faces.",
        "Each listed strip is the destination strip visible after the scramble.",
        "If N_i's strip shows label N_{i-1}, the scramble was X:cw, so solve with X:ccw.",
        "If N_i's strip shows label N_{i+1}, the scramble was X:ccw, so solve with X:cw.",
    ]
    lines.extend(
        f"{face}: {' '.join(DEFAULT_TOPOLOGY.neighbor_rings[face])}" for face in FACES
    )
    return "\n".join(lines)


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
