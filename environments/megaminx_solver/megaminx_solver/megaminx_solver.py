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

CANDIDATE_SYSTEM_PROMPT = """You are solving a text Megaminx candidate-selection task.

The puzzle has twelve faces labeled A through L. A solved face contains only its
own label. Centers identify the target color for each face and do not move.
Corner and edge strings show the sticker colors currently visible on that face.

You must make exactly one native select_candidate tool call. Do not write prose,
JSON, or visible reasoning before the tool call. Do not call rotate, inspect,
predict_rotate, or finish.
"""

CANDIDATE_PATH_SYSTEM_PROMPT = """You are solving a text Megaminx candidate-path task.

The puzzle has twelve faces labeled A through L. A solved face contains only its
own label. Centers identify the target color for each face and do not move.
Corner and edge strings show the sticker colors currently visible on that face.

Make native select_candidate tool calls only. Do not write prose, JSON, or
visible reasoning before tool calls. After each tool observation, if the puzzle
is not solved and the move budget has not been exhausted, make another
select_candidate call. Do not call rotate, inspect, predict_rotate, or finish.
"""

CANDIDATE_INDEX_SYSTEM_PROMPT = """You are solving a text Megaminx candidate-index task.

The puzzle has twelve faces labeled A through L. A solved face contains only its
own label. Centers identify the target color for each face and do not move.
Corner and edge strings show the sticker colors currently visible on that face.

You must make exactly one native select_candidate_index tool call. Do not write
prose, JSON, visible reasoning, or any other tool call.
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
    "action_gated_overlap_strict_shaped_direction",
    "action_gated_mask_overlap_strict_shaped_direction",
    "action_gated_counterfactual_frontier_strict",
    "action_gated_counterfactual_frontier_value_strict",
    "action_gated_predict_rotate_value_strict",
    "action_gated_predict_rotate_transition",
    "action_gated_face_discovery",
    "action_gated_face_tournament",
    "action_gated_candidate_tournament",
    "action_gated_candidate_index",
    "action_gated_candidate_mask_index_rank",
    "action_gated_candidate_mask_frontier_equivalence",
    "action_gated_candidate_geometry_frontier",
    "action_gated_candidate_strict_frontier",
    "action_gated_candidate_path_solve",
    "action_gated_candidate_path_tail_solve",
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
    "stage_solve_direction_flow_native_tool_v2",
    "stage_solve_action_table_native_tool",
    "stage_solve_action_mask_native_tool",
    "stage_frontier_sensor_native_tool",
    "stage_frontier_sensor_compact_native_tool",
    "stage_predict_rotate_native_tool",
    "stage_predict_transition_native_tool",
    "stage_face_discovery_native_tool",
    "stage_face_tournament_native_tool",
    "stage_candidate_tournament_native_tool",
    "stage_candidate_index_native_tool",
    "stage_candidate_scorecard_native_tool",
    "stage_candidate_scorecard_no_frontier_native_tool",
    "stage_candidate_scorecard_mask_native_tool",
    "stage_candidate_scorecard_mask_index_native_tool",
    "stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
    "stage_candidate_geometry_frontier_native_tool",
    "stage_candidate_relative_flow_frontier_native_tool",
    "stage_candidate_relative_flow_rule_frontier_native_tool",
    "stage_candidate_relative_flow_rule_solve2_native_tool",
}

CANDIDATE_MULTISTEP_PROMPT_STYLES = {
    "stage_candidate_relative_flow_rule_solve2_native_tool",
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
    "stage_solve_direction_flow_native_tool_v2",
    "stage_solve_action_table_native_tool",
    "stage_solve_action_mask_native_tool",
    "stage_frontier_sensor_native_tool",
    "stage_frontier_sensor_compact_native_tool",
    "stage_predict_rotate_native_tool",
    "stage_predict_transition_native_tool",
    "stage_face_discovery_native_tool",
    "stage_face_tournament_native_tool",
    "stage_candidate_tournament_native_tool",
    "stage_candidate_index_native_tool",
    "stage_candidate_scorecard_native_tool",
    "stage_candidate_scorecard_no_frontier_native_tool",
    "stage_candidate_scorecard_mask_native_tool",
    "stage_candidate_scorecard_mask_index_native_tool",
    "stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
    "stage_candidate_geometry_frontier_native_tool",
    "stage_candidate_relative_flow_frontier_native_tool",
    "stage_candidate_relative_flow_rule_frontier_native_tool",
    "stage_candidate_relative_flow_rule_solve2_native_tool",
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
    "stage_solve_direction_flow_native_tool_v2",
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
    "stage_solve_direction_flow_native_tool_v2",
    "stage_frontier_sensor_native_tool",
    "stage_frontier_sensor_compact_native_tool",
    "stage_predict_rotate_native_tool",
    "stage_predict_transition_native_tool",
    "stage_face_discovery_native_tool",
    "stage_face_tournament_native_tool",
    "stage_candidate_tournament_native_tool",
    "stage_candidate_index_native_tool",
    "stage_candidate_scorecard_native_tool",
    "stage_candidate_scorecard_no_frontier_native_tool",
    "stage_candidate_scorecard_mask_native_tool",
    "stage_candidate_scorecard_mask_index_native_tool",
    "stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
    "stage_candidate_geometry_frontier_native_tool",
    "stage_candidate_relative_flow_frontier_native_tool",
    "stage_candidate_relative_flow_rule_frontier_native_tool",
    "stage_candidate_relative_flow_rule_solve2_native_tool",
    "sensor_native_tool",
    "sensor_match_native_tool",
}
MATCH_TABLE_PROMPT_STYLES = {
    "sensor_match_json_action",
    "sensor_indexed_match_json_action",
    "sensor_candidate_strips_json_action",
    "stage_face_hint_direction_json_action",
    "stage_solve_direction_flow_native_tool_v2",
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
    "stage_solve_direction_flow_native_tool_v2",
    "stage_frontier_sensor_native_tool",
    "stage_frontier_sensor_compact_native_tool",
    "stage_predict_rotate_native_tool",
    "stage_predict_transition_native_tool",
    "stage_face_discovery_native_tool",
    "stage_face_tournament_native_tool",
    "stage_candidate_tournament_native_tool",
    "stage_candidate_index_native_tool",
    "stage_candidate_scorecard_native_tool",
    "stage_candidate_scorecard_no_frontier_native_tool",
    "stage_candidate_scorecard_mask_native_tool",
    "stage_candidate_scorecard_mask_index_native_tool",
    "stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
    "stage_candidate_geometry_frontier_native_tool",
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
SOLVE_DIRECTION_FLOW_V2_PROMPT_STYLES = {
    "stage_solve_direction_flow_native_tool_v2",
}
SOLVE_ACTION_TABLE_PROMPT_STYLES = {
    "stage_solve_action_table_native_tool",
}
SOLVE_ACTION_MASK_PROMPT_STYLES = {
    "stage_solve_action_mask_native_tool",
}
FACE_DISCOVERY_PROMPT_STYLES = {
    "stage_face_discovery_native_tool",
    "stage_face_tournament_native_tool",
    "stage_candidate_tournament_native_tool",
    "stage_candidate_index_native_tool",
    "stage_candidate_scorecard_native_tool",
    "stage_candidate_scorecard_no_frontier_native_tool",
    "stage_candidate_scorecard_mask_native_tool",
    "stage_candidate_scorecard_mask_index_native_tool",
    "stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
    "stage_candidate_geometry_frontier_native_tool",
    "stage_candidate_relative_flow_frontier_native_tool",
    "stage_candidate_relative_flow_rule_frontier_native_tool",
    "stage_candidate_relative_flow_rule_solve2_native_tool",
}
FACE_TOURNAMENT_PROMPT_STYLES = {
    "stage_face_tournament_native_tool",
    "stage_candidate_tournament_native_tool",
    "stage_candidate_index_native_tool",
    "stage_candidate_scorecard_native_tool",
    "stage_candidate_scorecard_no_frontier_native_tool",
    "stage_candidate_scorecard_mask_native_tool",
    "stage_candidate_scorecard_mask_index_native_tool",
    "stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
    "stage_candidate_geometry_frontier_native_tool",
    "stage_candidate_relative_flow_frontier_native_tool",
    "stage_candidate_relative_flow_rule_frontier_native_tool",
    "stage_candidate_relative_flow_rule_solve2_native_tool",
}
CANDIDATE_SELECT_PROMPT_STYLES = {
    "stage_candidate_tournament_native_tool",
    "stage_candidate_scorecard_native_tool",
    "stage_candidate_scorecard_no_frontier_native_tool",
    "stage_candidate_scorecard_mask_native_tool",
    "stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
    "stage_candidate_geometry_frontier_native_tool",
    "stage_candidate_relative_flow_frontier_native_tool",
    "stage_candidate_relative_flow_rule_frontier_native_tool",
    "stage_candidate_relative_flow_rule_solve2_native_tool",
}
CANDIDATE_INDEX_PROMPT_STYLES = {
    "stage_candidate_index_native_tool",
    "stage_candidate_scorecard_mask_index_native_tool",
}
CANDIDATE_TOOL_PROMPT_STYLES = CANDIDATE_SELECT_PROMPT_STYLES | CANDIDATE_INDEX_PROMPT_STYLES
CANDIDATE_FRONTIER_EQUIVALENCE_PROMPT_STYLES = {
    "stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
}
CANDIDATE_GEOMETRY_FRONTIER_PROMPT_STYLES = {
    "stage_candidate_geometry_frontier_native_tool",
    "stage_candidate_relative_flow_frontier_native_tool",
    "stage_candidate_relative_flow_rule_frontier_native_tool",
    "stage_candidate_relative_flow_rule_solve2_native_tool",
}
FRONTIER_SENSOR_PROMPT_STYLES = {
    "stage_frontier_sensor_native_tool",
    "stage_frontier_sensor_compact_native_tool",
    "stage_predict_rotate_native_tool",
}
PREDICT_ROTATE_PROMPT_STYLES = {
    "stage_predict_rotate_native_tool",
    "stage_predict_transition_native_tool",
}
NATIVE_TOOL_PROMPT_STYLES = {
    "native_action",
    "topology_native_tool",
    "sensor_native_tool",
    "sensor_match_native_tool",
    "stage_direction_flow_native_tool",
    "stage_solve_direction_flow_native_tool",
    "stage_solve_direction_flow_native_tool_v2",
    "stage_solve_action_table_native_tool",
    "stage_solve_action_mask_native_tool",
    "stage_frontier_sensor_native_tool",
    "stage_frontier_sensor_compact_native_tool",
    "stage_predict_rotate_native_tool",
    "stage_predict_transition_native_tool",
    "stage_face_discovery_native_tool",
    "stage_face_tournament_native_tool",
    "stage_candidate_tournament_native_tool",
    "stage_candidate_index_native_tool",
    "stage_candidate_scorecard_native_tool",
    "stage_candidate_scorecard_no_frontier_native_tool",
    "stage_candidate_scorecard_mask_native_tool",
    "stage_candidate_scorecard_mask_index_native_tool",
    "stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
    "stage_candidate_geometry_frontier_native_tool",
    "stage_candidate_relative_flow_frontier_native_tool",
    "stage_candidate_relative_flow_rule_frontier_native_tool",
    "stage_candidate_relative_flow_rule_solve2_native_tool",
}
REASONED_NATIVE_TOOL_PROMPT_STYLES = {
    "stage_direction_flow_native_tool",
    "stage_solve_direction_flow_native_tool",
    "stage_solve_direction_flow_native_tool_v2",
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
    prompt_style: str
    initial_puzzle: MegaminxPuzzle
    initial_sticker_accuracy: float
    initial_piece_accuracy: float
    initial_affected_faces: tuple[str, ...]
    initial_action_mask_counts: dict[Move, int]
    candidate_faces: tuple[str, ...] = ()
    candidate_seed: int = 0
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
    candidate_select_call_count: int = 0
    predict_rotate_call_count: int = 0
    inspect_call_count: int = 0
    finish_call_count: int = 0
    finished: bool = False
    last_move: str = "none"
    first_rotate: Move | None = None
    first_candidate_index: int | None = None
    first_prediction: dict[str, str] | None = None
    first_prediction_valid: bool = False
    first_prediction_item_count: int = 0
    first_prediction_extra_count: int = 0
    move_history: list[Move] = field(default_factory=list)
    candidate_index_history: list[int] = field(default_factory=list)
    second_candidate_faces: tuple[str, ...] = ()
    second_candidate_puzzle: MegaminxPuzzle | None = None

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


LEGAL_MOVES: tuple[Move, ...] = tuple(
    (face, direction) for face in FACES for direction in DIRECTIONS
)


def _copy_after(puzzle: MegaminxPuzzle, move: Move) -> MegaminxPuzzle:
    candidate = puzzle.copy()
    candidate.apply_move(*move)
    return candidate


def _one_turn_solvers(puzzle: MegaminxPuzzle) -> list[Move]:
    return [move for move in LEGAL_MOVES if _copy_after(puzzle, move).is_solved()]


def _frontier_after(initial_puzzle: MegaminxPuzzle, move: Move) -> tuple[int, int]:
    after = _copy_after(initial_puzzle, move)
    tail_solvers = _one_turn_solvers(after)
    tail_best_mask = max(_action_mask_counts(after).values(), default=0)
    return len(tail_solvers), tail_best_mask


def _solved_or_frontier_after(initial_puzzle: MegaminxPuzzle, move: Move) -> bool:
    after = _copy_after(initial_puzzle, move)
    return after.is_solved() or bool(_one_turn_solvers(after))


def _face_frontier_viable(initial_puzzle: MegaminxPuzzle, face: str) -> bool:
    return any(
        _solved_or_frontier_after(initial_puzzle, (face, direction))
        for direction in DIRECTIONS
    )


def _candidate_tournament_faces(
    puzzle: MegaminxPuzzle,
    inverse_solution: Sequence[Move],
    index: int,
    count: int = 4,
) -> list[str]:
    if not inverse_solution:
        return list(FACES[:count])
    target_face = inverse_solution[0][0]
    mask_counts = _action_mask_counts(puzzle)
    scored_faces: list[tuple[int, int, str]] = []
    for face in FACES:
        if face == target_face:
            continue
        score = max(mask_counts.get((face, direction), 0) for direction in DIRECTIONS)
        scored_faces.append((-score, FACES.index(face), face))
    candidates = [target_face, *(face for _, _, face in sorted(scored_faces)[: count - 1])]
    rng = Random(1009 * (index + 1) + 37 * FACES.index(target_face))
    rng.shuffle(candidates)
    return candidates


def _candidate_frontier_equivalence_faces(
    puzzle: MegaminxPuzzle,
    inverse_solution: Sequence[Move],
    index: int,
    count: int = 4,
) -> list[str]:
    if not inverse_solution:
        return list(FACES[:count])
    target_face = inverse_solution[0][0]
    mask_counts = _action_mask_counts(puzzle)

    candidates: list[str] = []
    if target_face in FACES:
        candidates.append(target_face)

    frontier_faces = [
        face
        for face in FACES
        if any(_solved_or_frontier_after(puzzle, (face, direction)) for direction in DIRECTIONS)
    ]
    ranked_frontier_faces = sorted(
        frontier_faces,
        key=lambda face: (
            -max(mask_counts.get((face, direction), 0) for direction in DIRECTIONS),
            FACES.index(face),
        ),
    )
    for face in ranked_frontier_faces:
        if face not in candidates:
            candidates.append(face)
        if len(candidates) >= count:
            break

    if len(candidates) < count:
        fillers = sorted(
            (face for face in FACES if face not in candidates),
            key=lambda face: (
                -max(mask_counts.get((face, direction), 0) for direction in DIRECTIONS),
                FACES.index(face),
            ),
        )
        candidates.extend(fillers[: count - len(candidates)])

    rng = Random(1009 * (index + 1) + 37 * FACES.index(target_face) + 211)
    rng.shuffle(candidates)
    return candidates


def _candidate_geometry_faces(
    puzzle: MegaminxPuzzle,
    index: int,
    count: int = 4,
    required_face: str | None = None,
    required_slot: int | None = None,
) -> list[str]:
    affected = set(_affected_faces(puzzle))
    mask_counts = _action_mask_counts(puzzle)
    scored_faces: list[tuple[int, int, int, int, str]] = []
    for face in FACES:
        ring = DEFAULT_TOPOLOGY.neighbor_rings[face]
        affected_neighbors = len(set(ring) & affected)
        changed_touching_stickers = 0
        for neighbor in ring:
            side = DEFAULT_TOPOLOGY.neighbor_rings[neighbor].index(face)
            touching_positions = (
                (neighbor, 1 + ((side - 1) % 5)),
                (neighbor, 6 + side),
                (neighbor, 1 + side),
            )
            changed_touching_stickers += sum(
                puzzle.stickers[position] != neighbor for position in touching_positions
            )
        best_visible_mask = max(mask_counts.get((face, direction), 0) for direction in DIRECTIONS)
        scored_faces.append(
            (
                -affected_neighbors,
                -changed_touching_stickers,
                -best_visible_mask,
                FACES.index(face),
                face,
            )
        )

    candidates = [face for *_scores, face in sorted(scored_faces)[:count]]
    if required_face in FACES and required_face not in candidates:
        candidates[-1] = required_face
    rng = Random(1103 * (index + 1) + 17 * len(affected))
    rng.shuffle(candidates)
    if (
        required_face in candidates
        and required_slot is not None
        and 1 <= required_slot <= len(candidates)
    ):
        target_index = candidates.index(required_face)
        desired_index = required_slot - 1
        candidates[target_index], candidates[desired_index] = (
            candidates[desired_index],
            candidates[target_index],
        )
    return candidates


def _best_tail_value(puzzle: MegaminxPuzzle) -> float:
    if puzzle.is_solved():
        return 1.0
    best = 0.0
    for move in LEGAL_MOVES:
        after = _copy_after(puzzle, move)
        if after.is_solved():
            return 1.0
        value = 0.70 * after.sticker_accuracy() + 0.30 * after.piece_accuracy()
        best = max(best, value)
    return best


def _counterfactual_first_move_values(initial_puzzle: MegaminxPuzzle) -> dict[Move, float]:
    return {move: _best_tail_value(_copy_after(initial_puzzle, move)) for move in LEGAL_MOVES}


def _counterfactual_value_stats(initial_puzzle: MegaminxPuzzle, move: Move) -> tuple[float, float, float]:
    values = _counterfactual_first_move_values(initial_puzzle)
    chosen = values[move]
    ordered = sorted(values.values())
    low = ordered[0]
    high = ordered[-1]
    norm = 0.0 if high == low else (chosen - low) / (high - low)
    rank = sum(value <= chosen + 1e-12 for value in ordered) / len(ordered)
    return chosen, norm, rank


def _move_touching_strips_after(puzzle: MegaminxPuzzle, move: Move) -> dict[str, str]:
    face, _ = move
    after = _copy_after(puzzle, move)
    return {
        neighbor: "".join(after.stickers[position] for position in DEFAULT_TOPOLOGY.side_strip(face, neighbor))
        for neighbor in DEFAULT_TOPOLOGY.neighbor_rings[face]
    }


def _validate_predicted_after(raw: Any, face: str) -> tuple[dict[str, str] | None, str | None, int, int]:
    if not isinstance(raw, list):
        return None, "predicted_after must be a five-item list.", 0, 0
    item_count = len(raw)
    extra_count = max(0, item_count - 5)
    if item_count != 5:
        return None, "predicted_after must contain exactly five strips.", item_count, extra_count
    normalized: dict[str, str] = {}
    ring = DEFAULT_TOPOLOGY.neighbor_rings[face]
    for neighbor, value in zip(ring, raw, strict=True):
        if not isinstance(value, str):
            return None, "each predicted_after item must be a string.", item_count, extra_count
        strip = value.strip().upper()
        if len(strip) != 3 or any(label not in FACES for label in strip):
            return None, "each predicted_after item must be exactly three labels from A-L.", item_count, extra_count
        normalized[neighbor] = strip
    return normalized, None, item_count, extra_count


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


async def selected_move_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(len(rollout.move_history) if rollout else 0)


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


async def candidate_select_call_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.candidate_select_call_count if rollout else 0)


async def predict_rotate_call_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.predict_rotate_call_count if rollout else 0)


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


async def first_rotate_in_candidate_set(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0.0
    try:
        task = _task_from_state(state)
    except (KeyError, TypeError, json.JSONDecodeError):
        return 0.0
    candidate_faces = task.get("candidate_faces")
    if not isinstance(candidate_faces, list):
        return 0.0
    return float(rollout.first_rotate[0] in candidate_faces)


async def target_face_in_candidate_set(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    try:
        task = _task_from_state(state)
    except (KeyError, TypeError, json.JSONDecodeError):
        return 0.0
    candidate_faces = task.get("candidate_faces")
    if not isinstance(candidate_faces, list):
        return 0.0
    return float(rollout.inverse_solution[0][0] in candidate_faces)


async def target_candidate_index(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return -1.0
    try:
        task = _task_from_state(state)
    except (KeyError, TypeError, json.JSONDecodeError):
        return -1.0
    candidate_faces = task.get("candidate_faces")
    if not isinstance(candidate_faces, list):
        return -1.0
    target_face = rollout.inverse_solution[0][0]
    try:
        return float(candidate_faces.index(target_face) + 1)
    except ValueError:
        return -1.0


async def first_candidate_index(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or rollout.first_candidate_index is None:
        return -1.0
    return float(rollout.first_candidate_index)


async def second_candidate_index(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or len(rollout.candidate_index_history) < 2:
        return -1.0
    return float(rollout.candidate_index_history[1])


async def first_candidate_face_correct(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or rollout.first_candidate_index is None or not rollout.inverse_solution:
        return 0.0
    try:
        task = _task_from_state(state)
    except (KeyError, TypeError, json.JSONDecodeError):
        return 0.0
    candidate_faces = task.get("candidate_faces")
    if not isinstance(candidate_faces, list):
        return 0.0
    candidate_index = rollout.first_candidate_index - 1
    if candidate_index < 0 or candidate_index >= len(candidate_faces):
        return 0.0
    return float(candidate_faces[candidate_index] == rollout.inverse_solution[0][0])


async def second_target_candidate_index(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or len(rollout.inverse_solution) < 2:
        return -1.0
    candidate_faces = rollout.second_candidate_faces or rollout.candidate_faces
    target_face = rollout.inverse_solution[1][0]
    try:
        return float(candidate_faces.index(target_face) + 1)
    except ValueError:
        return -1.0


async def second_candidate_face_correct(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or len(rollout.move_history) < 2 or len(rollout.inverse_solution) < 2:
        return 0.0
    candidate_faces = rollout.second_candidate_faces or rollout.candidate_faces
    candidate_index = (
        rollout.candidate_index_history[1] - 1
        if len(rollout.candidate_index_history) >= 2
        else -1
    )
    if candidate_index < 0 or candidate_index >= len(candidate_faces):
        return 0.0
    return float(candidate_faces[candidate_index] == rollout.inverse_solution[1][0])


def _first_candidate_face_from_state(state: vf.State) -> str | None:
    rollout = _rollout_from_state(state)
    if not rollout or rollout.first_candidate_index is None:
        return None
    try:
        task = _task_from_state(state)
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    candidate_faces = task.get("candidate_faces")
    if not isinstance(candidate_faces, list):
        return None
    candidate_index = rollout.first_candidate_index - 1
    if candidate_index < 0 or candidate_index >= len(candidate_faces):
        return None
    face = candidate_faces[candidate_index]
    return face if isinstance(face, str) and face in FACES else None


def _candidate_public_mask_scores_from_state(state: vf.State) -> list[tuple[str, int]]:
    rollout = _rollout_from_state(state)
    if not rollout:
        return []
    try:
        task = _task_from_state(state)
    except (KeyError, TypeError, json.JSONDecodeError):
        return []
    candidate_faces = task.get("candidate_faces")
    if not isinstance(candidate_faces, list):
        return []
    scores: list[tuple[str, int]] = []
    for face in candidate_faces:
        if not isinstance(face, str) or face not in FACES:
            continue
        scores.append(
            (
                face,
                max(
                    rollout.initial_action_mask_counts.get((face, direction), 0)
                    for direction in DIRECTIONS
                ),
            )
        )
    return scores


def _first_candidate_public_mask_stats(state: vf.State) -> tuple[int, int, float, bool]:
    chosen_face = _first_candidate_face_from_state(state)
    scores = _candidate_public_mask_scores_from_state(state)
    if chosen_face is None or not scores:
        return 0, 0, 0.0, False
    score_map = dict(scores)
    if chosen_face not in score_map:
        return 0, 0, 0.0, False
    chosen_score = score_map[chosen_face]
    max_score = max(score for _, score in scores)
    rank_credit = sum(score <= chosen_score for _, score in scores) / len(scores)
    return chosen_score, max_score, rank_credit, chosen_score == max_score


async def first_candidate_public_mask_count(state: vf.State) -> float:
    chosen_score, _, _, _ = _first_candidate_public_mask_stats(state)
    return float(chosen_score)


async def first_candidate_public_mask_is_max(state: vf.State) -> float:
    _, _, _, is_max = _first_candidate_public_mask_stats(state)
    return float(is_max)


async def first_candidate_public_mask_rank_credit(state: vf.State) -> float:
    _, _, rank_credit, _ = _first_candidate_public_mask_stats(state)
    return float(rank_credit)


def _candidate_relative_flow_action_scores_from_state(state: vf.State) -> list[tuple[Move, int]]:
    rollout = _rollout_from_state(state)
    if not rollout:
        return []
    try:
        task = _task_from_state(state)
    except (KeyError, TypeError, json.JSONDecodeError):
        return []
    candidate_faces = task.get("candidate_faces")
    if not isinstance(candidate_faces, list):
        return []
    scores: list[tuple[Move, int]] = []
    for face in candidate_faces:
        if not isinstance(face, str) or face not in FACES:
            continue
        for direction in DIRECTIONS:
            scores.append(((face, direction), _relative_flow_count(rollout.initial_puzzle, face, direction)))
    return scores


def _first_candidate_relative_flow_stats(state: vf.State) -> tuple[int, int, int, bool]:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0, 0, 0, False
    scores = _candidate_relative_flow_action_scores_from_state(state)
    if not scores:
        return 0, 0, 0, False
    score_map = dict(scores)
    chosen_score = score_map.get(rollout.first_rotate, 0)
    opposite = (rollout.first_rotate[0], _opposite_direction(rollout.first_rotate[1]))
    opposite_score = score_map.get(opposite, 0)
    max_score = max(score for _, score in scores)
    return chosen_score, opposite_score, chosen_score - opposite_score, chosen_score == max_score


async def first_candidate_relative_flow_count(state: vf.State) -> float:
    chosen_score, _, _, _ = _first_candidate_relative_flow_stats(state)
    return float(chosen_score)


async def first_candidate_relative_flow_margin(state: vf.State) -> float:
    _, _, margin, _ = _first_candidate_relative_flow_stats(state)
    return float(margin)


async def first_candidate_relative_flow_is_candidate_max(state: vf.State) -> float:
    _, _, _, is_max = _first_candidate_relative_flow_stats(state)
    return float(is_max)


async def candidate_relative_flow_oracle_unique(state: vf.State) -> float:
    scores = _candidate_relative_flow_action_scores_from_state(state)
    if not scores:
        return 0.0
    max_score = max(score for _, score in scores)
    return float(sum(score == max_score for _, score in scores) == 1)


def _second_candidate_relative_flow_stats(state: vf.State) -> tuple[int, int, int, bool]:
    rollout = _rollout_from_state(state)
    if (
        not rollout
        or len(rollout.move_history) < 2
        or not rollout.second_candidate_puzzle
        or not rollout.second_candidate_faces
    ):
        return 0, 0, 0, False
    chosen_move = rollout.move_history[1]
    scores: list[tuple[Move, int]] = []
    for face in rollout.second_candidate_faces:
        if face not in FACES:
            continue
        for direction in DIRECTIONS:
            scores.append(
                (
                    (face, direction),
                    _relative_flow_count(rollout.second_candidate_puzzle, face, direction),
                )
            )
    if not scores:
        return 0, 0, 0, False
    score_map = dict(scores)
    chosen_score = score_map.get(chosen_move, 0)
    opposite = (chosen_move[0], _opposite_direction(chosen_move[1]))
    opposite_score = score_map.get(opposite, 0)
    max_score = max(score for _, score in scores)
    return chosen_score, opposite_score, chosen_score - opposite_score, chosen_score == max_score


async def second_candidate_relative_flow_count(state: vf.State) -> float:
    chosen_score, _, _, _ = _second_candidate_relative_flow_stats(state)
    return float(chosen_score)


async def second_candidate_relative_flow_margin(state: vf.State) -> float:
    _, _, margin, _ = _second_candidate_relative_flow_stats(state)
    return float(margin)


async def second_candidate_relative_flow_is_candidate_max(state: vf.State) -> float:
    _, _, _, is_max = _second_candidate_relative_flow_stats(state)
    return float(is_max)


async def first_rotate_direction_correct(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate or not rollout.inverse_solution:
        return 0.0
    return float(rollout.first_rotate[1] == rollout.inverse_solution[0][1])


async def first_rotate_face_id(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return -1.0
    return float(FACES.index(rollout.first_rotate[0]))


async def first_rotate_direction_id(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return -1.0
    return float(DIRECTIONS.index(rollout.first_rotate[1]))


def _inverse_prefix_length(rollout: RolloutMegaminx) -> int:
    prefix = 0
    for actual, expected in zip(rollout.move_history, rollout.inverse_solution, strict=False):
        if actual != expected:
            break
        prefix += 1
    return prefix


async def inverse_prefix_length(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(_inverse_prefix_length(rollout) if rollout else 0)


async def second_rotate_correct(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or len(rollout.move_history) < 2 or len(rollout.inverse_solution) < 2:
        return 0.0
    return float(rollout.move_history[1] == rollout.inverse_solution[1])


async def second_rotate_face_correct(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or len(rollout.move_history) < 2 or len(rollout.inverse_solution) < 2:
        return 0.0
    return float(rollout.move_history[1][0] == rollout.inverse_solution[1][0])


async def second_rotate_direction_correct(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or len(rollout.move_history) < 2 or len(rollout.inverse_solution) < 2:
        return 0.0
    return float(rollout.move_history[1][1] == rollout.inverse_solution[1][1])


async def candidate_path_completed(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    required_moves = min(rollout.move_budget, len(rollout.inverse_solution))
    return float(rollout.move_count >= required_moves)


async def first_rotate_neighbor_overlap(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate or not rollout.initial_affected_faces:
        return 0.0
    ring = set(rollout.puzzle.topology.neighbor_rings[rollout.first_rotate[0]])
    affected = set(rollout.initial_affected_faces)
    return len(ring & affected) / len(affected)


async def first_rotate_mask_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0.0
    return float(rollout.initial_action_mask_counts.get(rollout.first_rotate, 0))


async def first_rotate_mask_is_max(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate or not rollout.initial_action_mask_counts:
        return 0.0
    count = rollout.initial_action_mask_counts.get(rollout.first_rotate, 0)
    return float(count == max(rollout.initial_action_mask_counts.values()))


async def initial_max_action_mask_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.initial_action_mask_counts:
        return 0.0
    return float(max(rollout.initial_action_mask_counts.values()))


async def first_rotate_frontier_tail_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0.0
    tail_count, _ = _frontier_after(rollout.initial_puzzle, rollout.first_rotate)
    return float(tail_count)


async def first_rotate_frontier_tail_unique(state: vf.State) -> float:
    return float(await first_rotate_frontier_tail_count(state) == 1.0)


async def first_rotate_frontier_tail_best_mask_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0.0
    _, tail_best_mask = _frontier_after(rollout.initial_puzzle, rollout.first_rotate)
    return float(tail_best_mask)


async def first_rotate_face_frontier_viable(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0.0
    return float(_face_frontier_viable(rollout.initial_puzzle, rollout.first_rotate[0]))


async def first_rotate_counterfactual_frontier(state: vf.State) -> float:
    return float(await first_rotate_frontier_tail_count(state) > 0.0)


async def action_reaches_one_turn_frontier(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0.0
    after = _copy_after(rollout.initial_puzzle, rollout.first_rotate)
    return float(not after.is_solved() and bool(_one_turn_solvers(after)))


async def frontier_equivalent(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0.0
    after = _copy_after(rollout.initial_puzzle, rollout.first_rotate)
    return float(after.is_solved() or bool(_one_turn_solvers(after)))


async def first_rotate_counterfactual_value(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0.0
    value, _, _ = _counterfactual_value_stats(rollout.initial_puzzle, rollout.first_rotate)
    return float(value)


async def first_rotate_counterfactual_value_norm(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0.0
    _, norm, _ = _counterfactual_value_stats(rollout.initial_puzzle, rollout.first_rotate)
    return float(norm)


async def first_rotate_counterfactual_value_rank(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0.0
    _, _, rank = _counterfactual_value_stats(rollout.initial_puzzle, rollout.first_rotate)
    return float(rank)


async def first_prediction_strip_accuracy(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate or not rollout.first_prediction:
        return 0.0
    expected = _move_touching_strips_after(rollout.initial_puzzle, rollout.first_rotate)
    if not expected:
        return 0.0
    correct = sum(
        rollout.first_prediction.get(face) == strip for face, strip in expected.items()
    )
    return correct / len(expected)


async def first_prediction_exact_strip_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate or not rollout.first_prediction:
        return 0.0
    expected = _move_touching_strips_after(rollout.initial_puzzle, rollout.first_rotate)
    return float(
        sum(rollout.first_prediction.get(face) == strip for face, strip in expected.items())
    )


async def first_prediction_char_accuracy(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate or not rollout.first_prediction:
        return 0.0
    expected = _move_touching_strips_after(rollout.initial_puzzle, rollout.first_rotate)
    correct = 0
    total = 0
    for face, expected_strip in expected.items():
        predicted_strip = rollout.first_prediction.get(face, "")
        for index, expected_label in enumerate(expected_strip):
            total += 1
            if index < len(predicted_strip) and predicted_strip[index] == expected_label:
                correct += 1
    return correct / total if total else 0.0


async def first_prediction_quality(state: vf.State) -> float:
    strip_accuracy = await first_prediction_strip_accuracy(state)
    char_accuracy = await first_prediction_char_accuracy(state)
    return 0.50 * strip_accuracy + 0.50 * char_accuracy


async def first_prediction_valid(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(bool(rollout and rollout.first_prediction_valid))


async def first_prediction_item_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.first_prediction_item_count if rollout else 0)


async def first_prediction_extra_count(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    return float(rollout.first_prediction_extra_count if rollout else 0)


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
        "action_gated_overlap_strict_shaped_direction": 8.0,
        "action_gated_mask_overlap_strict_shaped_direction": 9.0,
        "action_gated_counterfactual_frontier_strict": 10.0,
        "action_gated_counterfactual_frontier_value_strict": 11.0,
        "action_gated_predict_rotate_value_strict": 12.0,
        "action_gated_predict_rotate_transition": 13.0,
        "action_gated_face_discovery": 14.0,
        "action_gated_face_tournament": 15.0,
        "action_gated_candidate_tournament": 16.0,
        "action_gated_candidate_index": 17.0,
        "action_gated_candidate_mask_index_rank": 18.0,
        "action_gated_candidate_mask_frontier_equivalence": 19.0,
        "action_gated_candidate_geometry_frontier": 20.0,
        "action_gated_candidate_strict_frontier": 21.0,
        "action_gated_candidate_path_solve": 22.0,
        "action_gated_candidate_path_tail_solve": 23.0,
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


async def action_gated_overlap_strict_shaped_direction_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    if not _is_clean_single_rotate_attempt(rollout):
        return 0.0

    target_face, target_direction = rollout.inverse_solution[0]
    first_face, first_direction = rollout.first_rotate
    if rollout.solved() and rollout.first_rotate == rollout.inverse_solution[0]:
        return 1.0

    reward = 0.02
    reward += 0.25 * await first_rotate_neighbor_overlap(state)
    if first_face == target_face:
        reward += 0.25
        if first_direction == target_direction:
            reward += 0.30

    sticker_delta = rollout.puzzle.sticker_accuracy() - rollout.initial_sticker_accuracy
    piece_delta = rollout.puzzle.piece_accuracy() - rollout.initial_piece_accuracy
    progress_delta = max(0.0, 0.7 * sticker_delta + 0.3 * piece_delta)
    reward += min(0.13, progress_delta)
    return min(0.95, reward)


async def action_gated_mask_overlap_strict_shaped_direction_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    if not _is_clean_single_rotate_attempt(rollout):
        return 0.0
    if rollout.solved() and rollout.first_rotate == rollout.inverse_solution[0]:
        return 1.0

    base = await action_gated_overlap_strict_shaped_direction_reward(state)
    mask_count = rollout.initial_action_mask_counts.get(rollout.first_rotate, 0)
    max_mask_count = max(rollout.initial_action_mask_counts.values(), default=0)
    local_reward = 0.02 + 0.45 * (mask_count / 5.0)
    if mask_count == max_mask_count and max_mask_count > 0:
        local_reward += 0.10

    sticker_delta = rollout.puzzle.sticker_accuracy() - rollout.initial_sticker_accuracy
    piece_delta = rollout.puzzle.piece_accuracy() - rollout.initial_piece_accuracy
    progress_delta = max(0.0, 0.7 * sticker_delta + 0.3 * piece_delta)
    local_reward += min(0.13, progress_delta)
    return min(0.95, max(base, local_reward))


async def action_gated_counterfactual_frontier_strict_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0.0
    if not _is_clean_single_rotate_attempt(rollout):
        return 0.0
    if rollout.solved():
        return 1.0

    tail_count, tail_best_mask = _frontier_after(rollout.initial_puzzle, rollout.first_rotate)
    mask_norm = rollout.initial_action_mask_counts.get(rollout.first_rotate, 0) / 5.0
    if tail_count > 0:
        return min(
            0.95,
            0.78
            + (0.07 if tail_count == 1 else 0.0)
            + 0.05 * mask_norm
            + 0.05 * (tail_best_mask / 5.0),
        )

    face_viable = _face_frontier_viable(rollout.initial_puzzle, rollout.first_rotate[0])
    sticker_delta = rollout.puzzle.sticker_accuracy() - rollout.initial_sticker_accuracy
    piece_delta = rollout.puzzle.piece_accuracy() - rollout.initial_piece_accuracy
    progress_delta = max(0.0, 0.7 * sticker_delta + 0.3 * piece_delta)
    reward = 0.03
    reward += 0.25 * float(face_viable)
    reward += 0.20 * mask_norm
    reward += 0.07 * await first_rotate_neighbor_overlap(state)
    reward += min(0.05, progress_delta)
    return min(0.55, reward)


async def action_gated_counterfactual_frontier_value_strict_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0.0
    if not _is_clean_single_rotate_attempt(rollout):
        return 0.0
    if rollout.solved():
        return 1.0

    tail_count, _ = _frontier_after(rollout.initial_puzzle, rollout.first_rotate)
    _, value_norm, value_rank = _counterfactual_value_stats(
        rollout.initial_puzzle,
        rollout.first_rotate,
    )
    if tail_count > 0:
        return min(
            0.95,
            0.72
            + (0.10 if tail_count == 1 else 0.0)
            + 0.08 * value_norm
            + 0.05 * value_rank,
        )

    sticker_delta = rollout.puzzle.sticker_accuracy() - rollout.initial_sticker_accuracy
    piece_delta = rollout.puzzle.piece_accuracy() - rollout.initial_piece_accuracy
    progress_delta = max(0.0, 0.7 * sticker_delta + 0.3 * piece_delta)
    return min(0.62, 0.04 + 0.45 * value_norm + 0.08 * value_rank + min(0.05, progress_delta))


async def action_gated_predict_rotate_value_strict_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0.0
    if not _is_clean_single_predict_rotate_attempt(rollout):
        return 0.0

    prediction_accuracy = await first_prediction_strip_accuracy(state)
    prediction_quality = await first_prediction_quality(state)
    if rollout.solved():
        solved_value = min(1.0, 0.90 + 0.10 * prediction_accuracy)
        return solved_value * max(0.05, prediction_quality)

    tail_count, _ = _frontier_after(rollout.initial_puzzle, rollout.first_rotate)
    _, value_norm, value_rank = _counterfactual_value_stats(
        rollout.initial_puzzle,
        rollout.first_rotate,
    )
    if tail_count > 0:
        frontier_value = min(0.97, 0.62 + 0.10 * value_norm + 0.05 * value_rank)
        return frontier_value * max(0.05, prediction_quality)

    sticker_delta = rollout.puzzle.sticker_accuracy() - rollout.initial_sticker_accuracy
    piece_delta = rollout.puzzle.piece_accuracy() - rollout.initial_piece_accuracy
    progress_delta = max(0.0, 0.7 * sticker_delta + 0.3 * piece_delta)
    progress_value = min(
        0.78,
        0.03
        + 0.25 * value_norm
        + 0.08 * value_rank
        + min(0.05, progress_delta),
    )
    return progress_value * max(0.05, prediction_quality)


async def action_gated_predict_rotate_transition_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.first_rotate:
        return 0.0
    if not _is_clean_single_predict_rotate_attempt(rollout):
        return 0.0
    strip_accuracy = await first_prediction_strip_accuracy(state)
    char_accuracy = await first_prediction_char_accuracy(state)
    exact_strip_fraction = await first_prediction_exact_strip_count(state) / 5.0
    return min(1.0, 0.75 * char_accuracy + 0.15 * strip_accuracy + 0.10 * exact_strip_fraction)


async def action_gated_face_discovery_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    if not _is_clean_single_rotate_attempt(rollout):
        return 0.0

    target_face, _ = rollout.inverse_solution[0]
    chosen_face, _ = rollout.first_rotate
    if chosen_face == target_face:
        return 1.0

    overlap = await first_rotate_neighbor_overlap(state)
    face_viable = await first_rotate_face_frontier_viable(state)
    max_mask = await initial_max_action_mask_count(state)
    mask_norm = (await first_rotate_mask_count(state)) / max(1.0, max_mask)
    return min(0.35, 0.18 * overlap + 0.12 * face_viable + 0.05 * mask_norm)


async def action_gated_face_tournament_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    if not _is_clean_single_rotate_attempt(rollout):
        return 0.0
    try:
        task = _task_from_state(state)
    except (KeyError, TypeError, json.JSONDecodeError):
        return 0.0
    candidate_faces = task.get("candidate_faces")
    if not isinstance(candidate_faces, list) or rollout.first_rotate[0] not in candidate_faces:
        return 0.0

    target_face, _ = rollout.inverse_solution[0]
    chosen_face, _ = rollout.first_rotate
    if chosen_face == target_face:
        return 1.0

    overlap = await first_rotate_neighbor_overlap(state)
    face_viable = await first_rotate_face_frontier_viable(state)
    max_mask = await initial_max_action_mask_count(state)
    mask_norm = (await first_rotate_mask_count(state)) / max(1.0, max_mask)
    return min(0.25, 0.12 * overlap + 0.08 * face_viable + 0.05 * mask_norm)


async def action_gated_candidate_tournament_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    if not _is_clean_single_candidate_select_attempt(rollout):
        return 0.0

    target_face, _ = rollout.inverse_solution[0]
    chosen_face, _ = rollout.first_rotate
    if chosen_face == target_face:
        return 1.0

    overlap = await first_rotate_neighbor_overlap(state)
    face_viable = await first_rotate_face_frontier_viable(state)
    max_mask = await initial_max_action_mask_count(state)
    mask_norm = (await first_rotate_mask_count(state)) / max(1.0, max_mask)
    return min(0.25, 0.12 * overlap + 0.08 * face_viable + 0.05 * mask_norm)


async def action_gated_candidate_mask_frontier_equivalence_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    if not _is_clean_single_candidate_select_attempt(rollout):
        return 0.0

    if rollout.solved():
        return 1.0
    if await action_reaches_one_turn_frontier(state):
        return 0.90

    mask_norm = (await first_rotate_mask_count(state)) / 5.0
    return min(0.60, 0.02 + 0.58 * mask_norm)


async def action_gated_candidate_geometry_frontier_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    if not _is_clean_single_candidate_select_attempt(rollout):
        return 0.0

    if rollout.solved():
        return 1.0
    if await action_reaches_one_turn_frontier(state):
        return 0.75

    face_credit = await first_candidate_face_correct(state)
    value_rank = await first_rotate_counterfactual_value_rank(state)
    sticker_delta = rollout.puzzle.sticker_accuracy() - rollout.initial_sticker_accuracy
    piece_delta = rollout.puzzle.piece_accuracy() - rollout.initial_piece_accuracy
    progress_delta = max(0.0, 0.70 * sticker_delta + 0.30 * piece_delta)
    progress_credit = min(1.0, 8.0 * progress_delta)
    return min(0.45, 0.04 + 0.18 * face_credit + 0.18 * value_rank + 0.05 * progress_credit)


async def action_gated_candidate_strict_frontier_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    if not _is_clean_single_candidate_select_attempt(rollout):
        return 0.0

    if rollout.solved():
        return 1.0
    if await action_reaches_one_turn_frontier(state):
        return 0.75
    if await first_candidate_face_correct(state):
        return 0.20 + 0.10 * await first_rotate_direction_correct(state)
    if await first_candidate_relative_flow_is_candidate_max(state):
        return 0.05
    return 0.0


async def action_gated_candidate_path_solve_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    if not _is_clean_candidate_path_attempt(rollout):
        return 0.0

    if rollout.solved():
        return 1.0

    prefix = _inverse_prefix_length(rollout)
    if prefix >= 1:
        return 0.70
    if await action_reaches_one_turn_frontier(state):
        return 0.62
    if await first_candidate_face_correct(state):
        return 0.28 + 0.08 * await first_rotate_direction_correct(state)
    if await first_candidate_relative_flow_is_candidate_max(state):
        return 0.10
    return 0.0


async def action_gated_candidate_path_tail_solve_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    if not _is_clean_candidate_path_attempt(rollout):
        return 0.0

    if rollout.solved():
        return 1.0

    prefix = _inverse_prefix_length(rollout)
    required_moves = min(rollout.move_budget, len(rollout.inverse_solution))
    if prefix >= 1:
        if rollout.move_count < required_moves:
            return 0.25
        second_face = await second_rotate_face_correct(state)
        second_direction = await second_rotate_direction_correct(state)
        return min(0.74, 0.35 + 0.18 * second_face + 0.12 * second_direction)
    if await action_reaches_one_turn_frontier(state):
        return 0.30
    if await first_candidate_face_correct(state):
        return 0.16 + 0.04 * await first_rotate_direction_correct(state)
    if await first_candidate_relative_flow_is_candidate_max(state):
        return 0.05
    return 0.0


async def action_gated_candidate_index_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    if not _is_clean_single_candidate_index_attempt(rollout):
        return 0.0
    return await first_candidate_face_correct(state)


async def action_gated_candidate_mask_index_rank_reward(state: vf.State) -> float:
    rollout = _rollout_from_state(state)
    if not rollout or not rollout.inverse_solution:
        return 0.0
    if not _is_clean_single_candidate_index_attempt(rollout):
        return 0.0
    if await first_candidate_face_correct(state):
        return 1.0
    return 0.45 * await first_candidate_public_mask_rank_credit(state)


def _is_clean_single_candidate_index_attempt(rollout: RolloutMegaminx) -> bool:
    action_attempts = rollout.native_tool_call_count + rollout.text_tool_action_count
    return (
        rollout.illegal_moves == 0
        and rollout.protocol_violation_count == 0
        and rollout.tool_parse_error_count == 0
        and rollout.tool_call_error_count == 0
        and rollout.move_count == 0
        and rollout.rotate_call_count == 0
        and rollout.candidate_select_call_count == 1
        and rollout.inspect_call_count == 0
        and rollout.finish_call_count == 0
        and action_attempts == 1
        and rollout.first_rotate is None
        and rollout.first_candidate_index is not None
    )


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


def _is_clean_single_candidate_select_attempt(rollout: RolloutMegaminx) -> bool:
    action_attempts = rollout.native_tool_call_count + rollout.text_tool_action_count
    return (
        rollout.illegal_moves == 0
        and rollout.protocol_violation_count == 0
        and rollout.tool_parse_error_count == 0
        and rollout.tool_call_error_count == 0
        and rollout.move_count == 1
        and rollout.rotate_call_count == 0
        and rollout.candidate_select_call_count == 1
        and rollout.inspect_call_count == 0
        and rollout.finish_call_count == 0
        and action_attempts == 1
        and rollout.first_rotate is not None
        and rollout.first_candidate_index is not None
    )


def _is_clean_candidate_path_attempt(rollout: RolloutMegaminx) -> bool:
    action_attempts = rollout.native_tool_call_count + rollout.text_tool_action_count
    return (
        rollout.illegal_moves == 0
        and rollout.protocol_violation_count == 0
        and rollout.tool_parse_error_count == 0
        and rollout.tool_call_error_count == 0
        and 0 < rollout.move_count <= rollout.move_budget
        and rollout.rotate_call_count == 0
        and rollout.candidate_select_call_count == rollout.move_count
        and rollout.inspect_call_count == 0
        and rollout.finish_call_count == 0
        and action_attempts == rollout.candidate_select_call_count
        and len(rollout.move_history) == rollout.move_count
        and rollout.first_rotate is not None
        and rollout.first_candidate_index is not None
    )


def _is_clean_single_predict_rotate_attempt(rollout: RolloutMegaminx) -> bool:
    action_attempts = rollout.native_tool_call_count + rollout.text_tool_action_count
    return (
        rollout.illegal_moves == 0
        and rollout.protocol_violation_count == 0
        and rollout.tool_parse_error_count == 0
        and rollout.tool_call_error_count == 0
        and rollout.move_count == 1
        and rollout.rotate_call_count == 0
        and rollout.predict_rotate_call_count == 1
        and rollout.inspect_call_count == 0
        and rollout.finish_call_count == 0
        and action_attempts == 1
        and rollout.first_rotate is not None
        and rollout.first_prediction is not None
        and rollout.first_prediction_valid
    )


def _is_one_shot_candidate_rollout(rollout: RolloutMegaminx | None) -> bool:
    return bool(
        rollout
        and rollout.prompt_style in CANDIDATE_TOOL_PROMPT_STYLES
        and rollout.move_budget <= 1
    )


def _compact_terminal_observation(rollout: RolloutMegaminx, prefix: str) -> str:
    rollout.finished = True
    return (
        f"{prefix} Rollout finished. "
        f"solved={rollout.solved()} moves={rollout.move_count}/{rollout.move_budget} "
        f"illegal_moves={rollout.illegal_moves} last_move={rollout.last_move}"
    )


def _candidate_path_observation(rollout: RolloutMegaminx, prefix: str) -> str:
    return "\n".join(
        [
            (
                f"{prefix} solved={rollout.solved()} "
                f"moves={rollout.move_count}/{rollout.move_budget} "
                f"moves_left={rollout.moves_left()} last_move={rollout.last_move}"
            ),
            "Updated candidate relative-flow view:",
            _candidate_relative_flow_rule_section(multi_step=True),
            _candidate_relative_flow_section(rollout.puzzle, rollout.candidate_faces),
        ]
    )


class MegaminxEnv(vf.StatefulToolEnv):
    def __init__(
        self,
        topology: MegaminxTopology | None = None,
        allow_text_tool_actions: bool = False,
        exposed_tool_names: Sequence[str] | None = None,
        **kwargs: Any,
    ):
        self.topology = topology or DEFAULT_TOPOLOGY
        self.allow_text_tool_actions = allow_text_tool_actions
        self.exposed_tool_names = tuple(exposed_tool_names) if exposed_tool_names else ()
        super().__init__(tools=[], error_formatter=lambda error: f"Tool error: {error}", **kwargs)
        available_tools = {
            "rotate": self.rotate,
            "select_candidate": self.select_candidate,
            "select_candidate_index": self.select_candidate_index,
            "predict_rotate": self.predict_rotate,
            "inspect": self.inspect,
            "finish": self.finish,
        }
        tool_names = self.exposed_tool_names or tuple(available_tools)
        unknown_tools = sorted(set(tool_names) - set(available_tools))
        if unknown_tools:
            raise ValueError(f"Unknown exposed tool names: {', '.join(unknown_tools)}")
        for tool_name in tool_names:
            self.add_tool(available_tools[tool_name], args_to_skip=["rollout"])
        self._constrain_tool_schemas()

    def _constrain_tool_schemas(self) -> None:
        for tool in self.tool_defs:
            properties = tool.parameters.get("properties", {})
            if tool.name in {"rotate", "predict_rotate"}:
                properties.get("face", {})["enum"] = list(FACES)
                properties.get("direction", {})["enum"] = list(DIRECTIONS)
                if tool.name == "predict_rotate":
                    predicted_after = properties.get("predicted_after", {})
                    predicted_after["type"] = "array"
                    predicted_after["minItems"] = 5
                    predicted_after["maxItems"] = 5
                    predicted_after["items"] = {
                        "type": "string",
                        "pattern": "^[A-L]{3}$",
                    }
            elif tool.name == "select_candidate":
                properties.get("direction", {})["enum"] = list(DIRECTIONS)
                index_schema = properties.get("index", {})
                index_schema["type"] = "integer"
                index_schema["minimum"] = 1
                index_schema["maximum"] = 4
            elif tool.name == "select_candidate_index":
                index_schema = properties.get("index", {})
                index_schema["type"] = "integer"
                index_schema["minimum"] = 1
                index_schema["maximum"] = 4
            elif tool.name == "inspect":
                properties.get("face", {})["enum"] = [*FACES, "all"]

    async def setup_state(self, state: vf.State) -> None:
        await super().setup_state(state)
        task = _task_from_state(state)
        scramble = _decode_moves(task["scramble"])
        inverse_solution = _decode_moves(task["inverse_solution"])
        puzzle = MegaminxPuzzle.solved(self.topology)
        puzzle.apply_moves(scramble)
        initial_puzzle = puzzle.copy()
        initial_sticker = puzzle.sticker_accuracy()
        initial_piece = puzzle.piece_accuracy()
        initial_affected = _affected_faces(puzzle)
        initial_action_mask_counts = _action_mask_counts(puzzle)
        raw_candidate_faces = task.get("candidate_faces")
        candidate_faces = (
            tuple(face for face in raw_candidate_faces if face in FACES)
            if isinstance(raw_candidate_faces, list)
            else ()
        )
        raw_candidate_seed = task.get("candidate_seed", 0)
        candidate_seed = raw_candidate_seed if isinstance(raw_candidate_seed, int) else 0
        state["megaminx"] = RolloutMegaminx(
            puzzle=puzzle,
            scramble=scramble,
            inverse_solution=inverse_solution,
            scramble_depth=int(task["scramble_depth"]),
            move_budget=int(task["move_budget"]),
            reward_style=task.get("reward_style", "dense"),
            prompt_style=task.get("prompt_style", "default"),
            initial_puzzle=initial_puzzle,
            initial_sticker_accuracy=initial_sticker,
            initial_piece_accuracy=initial_piece,
            initial_affected_faces=initial_affected,
            initial_action_mask_counts=initial_action_mask_counts,
            candidate_faces=candidate_faces,
            candidate_seed=candidate_seed,
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
        updated_args = dict(tool_args)
        if rollout is not None:
            updated_args["rollout"] = rollout
            if tool_name not in self.tool_map:
                rollout.illegal_moves += 1
        return updated_args

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
        if _is_one_shot_candidate_rollout(rollout) and len(tool_calls) != 1:
            assert rollout is not None
            rollout.protocol_violation_count += 1
            rollout.illegal_moves += 1
            tool_call_id = _tool_call_field(tool_calls[0], "id") if tool_calls else "native-tool-0"
            return [
                ToolMessage(
                    role="tool",
                    content=_compact_terminal_observation(
                        rollout,
                        f"Protocol violation: expected exactly one native tool call, got {len(tool_calls)}.",
                    ),
                    tool_call_id=str(tool_call_id or "native-tool-0"),
                )
            ]
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
                    tool_args = dict(raw_args)
                else:
                    raise ValueError("native tool call arguments must be a JSON object")
                if not isinstance(tool_args, dict):
                    raise ValueError("native tool call arguments must decode to an object")
            except Exception as error:
                if rollout is not None:
                    rollout.tool_parse_error_count += 1
                    rollout.illegal_moves += 1
                    if _is_one_shot_candidate_rollout(rollout):
                        tool_messages.append(
                            ToolMessage(
                                role="tool",
                                content=_compact_terminal_observation(
                                    rollout,
                                    f"Tool parse error: {error}",
                                ),
                                tool_call_id=str(tool_call_id),
                            )
                        )
                        return tool_messages
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
                    if _is_one_shot_candidate_rollout(rollout):
                        tool_messages.append(
                            ToolMessage(
                                role="tool",
                                content=_compact_terminal_observation(
                                    rollout,
                                    f"Tool call error: {error}",
                                ),
                                tool_call_id=str(tool_call_id),
                            )
                        )
                        return tool_messages
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
        rollout.move_history.append((face, direction))
        if rollout.first_rotate is None:
            rollout.first_rotate = (face, direction)
        if rollout.solved() or rollout.exhausted():
            rollout.finished = True
        if (
            rollout.prompt_style in FRONTIER_SENSOR_PROMPT_STYLES
            and not rollout.finished
        ):
            return "\n".join(
                [
                    rollout.observation(),
                    "Updated frontier sensor view:",
                    _candidate_strips_section(rollout.puzzle),
                ]
        )
        return rollout.observation()

    async def select_candidate(
        self,
        index: int,
        direction: str,
        rollout: RolloutMegaminx,
    ) -> str:
        """Rotate one candidate face by slot index.

        Args:
            index: Candidate slot number, from 1 through 4.
            direction: cw advances neighbor strips one slot through the printed ring; ccw reverses them.

        Returns:
            Updated puzzle observation after the move.
        """
        rollout.tool_calls += 1
        rollout.candidate_select_call_count += 1
        if rollout.finished:
            return (
                _compact_terminal_observation(rollout, "The rollout is already finished.")
                if _is_one_shot_candidate_rollout(rollout)
                else "The rollout is already finished.\n" + rollout.observation()
            )
        if rollout.exhausted():
            rollout.finished = True
            return (
                _compact_terminal_observation(rollout, "Move budget exhausted.")
                if _is_one_shot_candidate_rollout(rollout)
                else "Move budget exhausted.\n" + rollout.observation()
            )
        if not rollout.candidate_faces:
            rollout.illegal_moves += 1
            if _is_one_shot_candidate_rollout(rollout):
                return _compact_terminal_observation(
                    rollout,
                    "Illegal candidate selection: no candidate set is available.",
                )
            return "Illegal candidate selection: no candidate set is available.\n" + rollout.observation()
        if not isinstance(index, int) or isinstance(index, bool):
            rollout.illegal_moves += 1
            if _is_one_shot_candidate_rollout(rollout):
                return _compact_terminal_observation(
                    rollout,
                    "Illegal candidate selection: index must be an integer.",
                )
            return "Illegal candidate selection: index must be an integer.\n" + rollout.observation()
        candidate_index = index
        if candidate_index < 1 or candidate_index > len(rollout.candidate_faces):
            rollout.illegal_moves += 1
            if _is_one_shot_candidate_rollout(rollout):
                return _compact_terminal_observation(
                    rollout,
                    "Illegal candidate selection: index is outside the candidate set.",
                )
            return "Illegal candidate selection: index is outside the candidate set.\n" + rollout.observation()
        try:
            direction = validate_move(rollout.candidate_faces[candidate_index - 1], direction)[1]
        except ValueError as error:
            rollout.illegal_moves += 1
            if _is_one_shot_candidate_rollout(rollout):
                return _compact_terminal_observation(rollout, f"Illegal move: {error}")
            return f"Illegal move: {error}\n" + rollout.observation()

        face = rollout.candidate_faces[candidate_index - 1]
        rollout.puzzle.apply_move(face, direction)
        rollout.move_count += 1
        rollout.last_move = f"select_candidate:{candidate_index}:{move_to_text((face, direction))}"
        rollout.move_history.append((face, direction))
        rollout.candidate_index_history.append(candidate_index)
        if rollout.first_candidate_index is None:
            rollout.first_candidate_index = candidate_index
        if rollout.first_rotate is None:
            rollout.first_rotate = (face, direction)
        if rollout.solved() or rollout.exhausted():
            rollout.finished = True
        if rollout.prompt_style in CANDIDATE_MULTISTEP_PROMPT_STYLES:
            if rollout.finished:
                return _compact_terminal_observation(rollout, "Action recorded.")
            next_target_face = (
                rollout.inverse_solution[rollout.move_count][0]
                if rollout.move_count < len(rollout.inverse_solution)
                else None
            )
            next_target_slot = (
                1 + ((rollout.candidate_seed + rollout.move_count - 1) % 4)
                if next_target_face
                else None
            )
            rollout.second_candidate_puzzle = rollout.puzzle.copy()
            rollout.candidate_faces = tuple(
                _candidate_geometry_faces(
                    rollout.puzzle,
                    rollout.candidate_seed + rollout.move_count,
                    required_face=next_target_face,
                    required_slot=next_target_slot,
                )
            )
            if rollout.move_count == 1:
                rollout.second_candidate_faces = rollout.candidate_faces
            return _candidate_path_observation(rollout, "Action recorded.")
        if _is_one_shot_candidate_rollout(rollout):
            return _compact_terminal_observation(rollout, "Action recorded.")
        return rollout.observation()

    async def select_candidate_index(
        self,
        index: int,
        rollout: RolloutMegaminx,
    ) -> str:
        """Select one candidate face by slot index without choosing direction.

        Args:
            index: Candidate slot number, from 1 through 4.

        Returns:
            Updated task observation after recording the selection.
        """
        rollout.tool_calls += 1
        rollout.candidate_select_call_count += 1
        if rollout.finished:
            return (
                _compact_terminal_observation(rollout, "The rollout is already finished.")
                if _is_one_shot_candidate_rollout(rollout)
                else "The rollout is already finished.\n" + rollout.observation()
            )
        if rollout.exhausted():
            rollout.finished = True
            return (
                _compact_terminal_observation(rollout, "Move budget exhausted.")
                if _is_one_shot_candidate_rollout(rollout)
                else "Move budget exhausted.\n" + rollout.observation()
            )
        if not rollout.candidate_faces:
            rollout.illegal_moves += 1
            if _is_one_shot_candidate_rollout(rollout):
                return _compact_terminal_observation(
                    rollout,
                    "Illegal candidate selection: no candidate set is available.",
                )
            return "Illegal candidate selection: no candidate set is available.\n" + rollout.observation()
        if not isinstance(index, int) or isinstance(index, bool):
            rollout.illegal_moves += 1
            if _is_one_shot_candidate_rollout(rollout):
                return _compact_terminal_observation(
                    rollout,
                    "Illegal candidate selection: index must be an integer.",
                )
            return "Illegal candidate selection: index must be an integer.\n" + rollout.observation()
        if index < 1 or index > len(rollout.candidate_faces):
            rollout.illegal_moves += 1
            if _is_one_shot_candidate_rollout(rollout):
                return _compact_terminal_observation(
                    rollout,
                    "Illegal candidate selection: index is outside the candidate set.",
                )
            return "Illegal candidate selection: index is outside the candidate set.\n" + rollout.observation()

        rollout.last_move = f"select_candidate_index:{index}"
        if rollout.first_candidate_index is None:
            rollout.first_candidate_index = index
        rollout.finished = True
        if _is_one_shot_candidate_rollout(rollout):
            return _compact_terminal_observation(rollout, "Recorded candidate selection.")
        return "Recorded candidate selection. Rollout finished.\n" + rollout.observation()

    async def predict_rotate(
        self,
        face: str,
        direction: str,
        predicted_after: list[str],
        rollout: RolloutMegaminx,
    ) -> str:
        """Predict local strips after a move, then rotate that face.

        Args:
            face: One face label from A through L.
            direction: cw advances neighbor strips one slot through the printed ring; ccw reverses them.
            predicted_after: Five predicted three-letter strips after the move, in the chosen face's printed ring order.

        Returns:
            Updated puzzle observation after the move.
        """
        rollout.tool_calls += 1
        rollout.predict_rotate_call_count += 1
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
        prediction, error, item_count, extra_count = _validate_predicted_after(predicted_after, face)
        rollout.first_prediction_item_count = item_count
        rollout.first_prediction_extra_count = extra_count
        if error or prediction is None:
            rollout.illegal_moves += 1
            rollout.finished = True
            return f"Illegal prediction: {error}\n" + rollout.observation()

        rollout.first_prediction = prediction
        rollout.first_prediction_valid = True
        rollout.puzzle.apply_move(face, direction)
        rollout.move_count += 1
        rollout.last_move = f"predict_rotate:{move_to_text((face, direction))}"
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
    exposed_tool_names: Sequence[str] | None = None,
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
    default_exposed_tool_names: tuple[str, ...] | None = (
        ("select_candidate_index",)
        if prompt_style in CANDIDATE_INDEX_PROMPT_STYLES
        else ("select_candidate",)
        if prompt_style in CANDIDATE_SELECT_PROMPT_STYLES
        else None
    )
    resolved_exposed_tool_names = (
        tuple(exposed_tool_names)
        if exposed_tool_names is not None
        else default_exposed_tool_names
    )
    return MegaminxEnv(
        dataset=dataset,
        system_prompt=(
            CANDIDATE_INDEX_SYSTEM_PROMPT
            if prompt_style in CANDIDATE_INDEX_PROMPT_STYLES
            else
            CANDIDATE_PATH_SYSTEM_PROMPT
            if prompt_style in CANDIDATE_MULTISTEP_PROMPT_STYLES
            else
            CANDIDATE_SYSTEM_PROMPT
            if prompt_style in CANDIDATE_SELECT_PROMPT_STYLES
            else SYSTEM_PROMPT
        ),
        rubric=rubric,
        max_turns=global_max_turns,
        allow_text_tool_actions=allow_text_tool_actions,
        exposed_tool_names=resolved_exposed_tool_names,
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
            "exposed_tool_names": resolved_exposed_tool_names,
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
    if prompt_style in SOLVE_DIRECTION_FLOW_V2_PROMPT_STYLES and max_depth > 2:
        raise ValueError("stage solve-direction v2 prompt styles are only defined for depths 1-2")
    if prompt_style in SOLVE_ACTION_TABLE_PROMPT_STYLES and max_depth > 2:
        raise ValueError("stage solve-action table prompt styles are only defined for depths 1-2")
    if prompt_style in SOLVE_ACTION_MASK_PROMPT_STYLES and max_depth > 2:
        raise ValueError("stage solve-action mask prompt styles are only defined for depths 1-2")
    if prompt_style in FRONTIER_SENSOR_PROMPT_STYLES and max_depth > 2:
        raise ValueError("stage frontier sensor prompt styles are only defined for depths 1-2")
    if prompt_style in FACE_DISCOVERY_PROMPT_STYLES and max_depth > 2:
        raise ValueError("stage face-discovery prompt styles are only defined for depths 1-2")
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
    candidate_faces = (
        _candidate_frontier_equivalence_faces(puzzle, inverse_solution, index)
        if prompt_style in CANDIDATE_FRONTIER_EQUIVALENCE_PROMPT_STYLES
        else _candidate_geometry_faces(puzzle, index)
        if prompt_style in CANDIDATE_GEOMETRY_FRONTIER_PROMPT_STYLES
        else _candidate_tournament_faces(puzzle, inverse_solution, index)
        if prompt_style in FACE_TOURNAMENT_PROMPT_STYLES
        else None
    )
    prompt = [
        {
            "role": "user",
            "content": _build_prompt(
                index,
                split,
                depth,
                move_budget,
                puzzle,
                prompt_style,
                candidate_faces=candidate_faces,
            ),
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
        "candidate_seed": index,
    }
    if candidate_faces is not None:
        task["candidate_faces"] = candidate_faces
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
        selected_move_count,
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
        candidate_select_call_count,
        predict_rotate_call_count,
        inspect_call_count,
        finish_call_count,
        action_taken,
        first_rotate_correct,
        first_rotate_face_correct,
        first_rotate_in_candidate_set,
        target_face_in_candidate_set,
        target_candidate_index,
        first_candidate_index,
        second_candidate_index,
        first_candidate_face_correct,
        second_target_candidate_index,
        second_candidate_face_correct,
        first_candidate_public_mask_count,
        first_candidate_public_mask_is_max,
        first_candidate_public_mask_rank_credit,
        first_candidate_relative_flow_count,
        first_candidate_relative_flow_margin,
        first_candidate_relative_flow_is_candidate_max,
        candidate_relative_flow_oracle_unique,
        second_candidate_relative_flow_count,
        second_candidate_relative_flow_margin,
        second_candidate_relative_flow_is_candidate_max,
        first_rotate_direction_correct,
        first_rotate_face_id,
        first_rotate_direction_id,
        second_rotate_correct,
        second_rotate_face_correct,
        second_rotate_direction_correct,
        candidate_path_completed,
        inverse_prefix_length,
        first_rotate_neighbor_overlap,
        first_rotate_mask_count,
        first_rotate_mask_is_max,
        initial_max_action_mask_count,
        first_rotate_frontier_tail_count,
        first_rotate_frontier_tail_unique,
        first_rotate_frontier_tail_best_mask_count,
        first_rotate_face_frontier_viable,
        first_rotate_counterfactual_frontier,
        action_reaches_one_turn_frontier,
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
        else action_gated_mask_overlap_strict_shaped_direction_reward
        if reward_style_name == "action_gated_mask_overlap_strict_shaped_direction"
        else action_gated_counterfactual_frontier_strict_reward
        if reward_style_name == "action_gated_counterfactual_frontier_strict"
        else action_gated_counterfactual_frontier_value_strict_reward
        if reward_style_name == "action_gated_counterfactual_frontier_value_strict"
        else action_gated_predict_rotate_value_strict_reward
        if reward_style_name == "action_gated_predict_rotate_value_strict"
        else action_gated_predict_rotate_transition_reward
        if reward_style_name == "action_gated_predict_rotate_transition"
        else action_gated_face_discovery_reward
        if reward_style_name == "action_gated_face_discovery"
        else action_gated_face_tournament_reward
        if reward_style_name == "action_gated_face_tournament"
        else action_gated_candidate_tournament_reward
        if reward_style_name == "action_gated_candidate_tournament"
        else action_gated_candidate_mask_index_rank_reward
        if reward_style_name == "action_gated_candidate_mask_index_rank"
        else action_gated_candidate_mask_frontier_equivalence_reward
        if reward_style_name == "action_gated_candidate_mask_frontier_equivalence"
        else action_gated_candidate_strict_frontier_reward
        if reward_style_name == "action_gated_candidate_strict_frontier"
        else action_gated_candidate_path_solve_reward
        if reward_style_name == "action_gated_candidate_path_solve"
        else action_gated_candidate_path_tail_solve_reward
        if reward_style_name == "action_gated_candidate_path_tail_solve"
        else action_gated_candidate_geometry_frontier_reward
        if reward_style_name == "action_gated_candidate_geometry_frontier"
        else action_gated_candidate_index_reward
        if reward_style_name == "action_gated_candidate_index"
        else action_gated_overlap_strict_shaped_direction_reward
        if reward_style_name == "action_gated_overlap_strict_shaped_direction"
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
    if (
        reward_style in {
            "action_gated_predict_rotate_value_strict",
            "action_gated_predict_rotate_transition",
        }
        and prompt_style not in PREDICT_ROTATE_PROMPT_STYLES
    ):
        expected = ", ".join(sorted(PREDICT_ROTATE_PROMPT_STYLES))
        raise ValueError(
            "action_gated_predict_rotate_value_strict requires a predict-rotate prompt "
            f"style ({expected}), got {prompt_style!r}"
        )
    if (
        reward_style == "action_gated_face_discovery"
        and prompt_style != "stage_face_discovery_native_tool"
    ):
        raise ValueError(
            "action_gated_face_discovery requires prompt_style "
            "'stage_face_discovery_native_tool'"
        )
    if (
        reward_style == "action_gated_face_tournament"
        and prompt_style != "stage_face_tournament_native_tool"
    ):
        raise ValueError(
            "action_gated_face_tournament requires prompt_style "
            "'stage_face_tournament_native_tool'"
        )
    if (
        reward_style == "action_gated_candidate_tournament"
        and prompt_style not in CANDIDATE_SELECT_PROMPT_STYLES
    ):
        expected = ", ".join(sorted(CANDIDATE_SELECT_PROMPT_STYLES))
        raise ValueError(
            "action_gated_candidate_tournament requires a candidate-select prompt "
            f"style ({expected}), got {prompt_style!r}"
        )
    if (
        reward_style == "action_gated_candidate_index"
        and prompt_style not in CANDIDATE_INDEX_PROMPT_STYLES
    ):
        expected = ", ".join(sorted(CANDIDATE_INDEX_PROMPT_STYLES))
        raise ValueError(
            "action_gated_candidate_index requires a candidate-index prompt "
            f"style ({expected}), got {prompt_style!r}"
        )
    if (
        reward_style == "action_gated_candidate_mask_index_rank"
        and prompt_style != "stage_candidate_scorecard_mask_index_native_tool"
    ):
        raise ValueError(
            "action_gated_candidate_mask_index_rank requires prompt_style "
            "'stage_candidate_scorecard_mask_index_native_tool'"
        )
    if (
        reward_style == "action_gated_candidate_mask_frontier_equivalence"
        and prompt_style not in CANDIDATE_FRONTIER_EQUIVALENCE_PROMPT_STYLES
    ):
        expected = ", ".join(sorted(CANDIDATE_FRONTIER_EQUIVALENCE_PROMPT_STYLES))
        raise ValueError(
            "action_gated_candidate_mask_frontier_equivalence requires a frontier-equivalence "
            f"candidate prompt style ({expected}), got {prompt_style!r}"
        )
    if (
        reward_style
        in {
            "action_gated_candidate_geometry_frontier",
            "action_gated_candidate_strict_frontier",
            "action_gated_candidate_path_solve",
            "action_gated_candidate_path_tail_solve",
        }
        and prompt_style not in CANDIDATE_GEOMETRY_FRONTIER_PROMPT_STYLES
    ):
        expected = ", ".join(sorted(CANDIDATE_GEOMETRY_FRONTIER_PROMPT_STYLES))
        raise ValueError(
            f"{reward_style} requires a geometry-frontier "
            f"candidate prompt style ({expected}), got {prompt_style!r}"
        )


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
        return name, dict(args) if isinstance(args, dict) else {}
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
    candidate_faces: Sequence[str] | None = None,
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
        "Goal: solve within the move budget by choosing candidate slots and directions."
        if prompt_style in CANDIDATE_MULTISTEP_PROMPT_STYLES
        else "Goal: choose a candidate slot and direction that solves now or leaves a one-turn solve."
        if prompt_style
        in CANDIDATE_FRONTIER_EQUIVALENCE_PROMPT_STYLES
        | CANDIDATE_GEOMETRY_FRONTIER_PROMPT_STYLES
        else "Goal: choose the correct candidate slot from the candidate set and call select_candidate_index."
        if prompt_style in CANDIDATE_INDEX_PROMPT_STYLES
        else "Goal: choose the correct candidate slot from the candidate set and call select_candidate."
        if prompt_style in CANDIDATE_SELECT_PROMPT_STYLES
        else "Goal: choose the correct face from the candidate set and call rotate on that face."
        if prompt_style in FACE_TOURNAMENT_PROMPT_STYLES
        else "Goal: identify the face whose rotation best explains the disturbed sticker geometry."
        if prompt_style in FACE_DISCOVERY_PROMPT_STYLES
        else
        "Goal: choose one legal face turn and predict the five moved strips after that turn."
        if prompt_style == "stage_predict_transition_native_tool"
        else "Goal: solve the Megaminx so every face contains only its own label.",
        (
            "Legal action: select one printed candidate slot 1-4."
            if prompt_style in CANDIDATE_INDEX_PROMPT_STYLES
            else "Legal action: select one printed candidate slot 1-4 with direction cw or ccw."
            if prompt_style in CANDIDATE_SELECT_PROMPT_STYLES
            else "Legal moves: rotate any face A-L with direction cw or ccw."
        ),
        "Use select_candidate_index(index). Do not use any other tool in this prompt style."
        if prompt_style in CANDIDATE_INDEX_PROMPT_STYLES
        else "Use select_candidate(index, direction). Do not use rotate, inspect, or finish in this prompt style."
        if prompt_style in CANDIDATE_SELECT_PROMPT_STYLES
        else "Use predict_rotate(face, direction, predicted_after). Do not use rotate, inspect, or finish in this prompt style."
        if prompt_style in PREDICT_ROTATE_PROMPT_STYLES
        else "Use rotate(face, direction), inspect(face), and finish().",
        "Transition curriculum: the reward is based on predicting the simulator's local after-state for your chosen move."
        if prompt_style == "stage_predict_transition_native_tool"
        else _curriculum_hint(depth, prompt_style),
    ]
    if prompt_style in ACTION_FIRST_PROMPT_STYLES:
        action_name = (
            "select_candidate_index"
            if prompt_style in CANDIDATE_INDEX_PROMPT_STYLES
            else "select_candidate"
            if prompt_style in CANDIDATE_SELECT_PROMPT_STYLES
            else "predict_rotate"
            if prompt_style in PREDICT_ROTATE_PROMPT_STYLES
            else "rotate"
        )
        lines.extend(
            [
                f"Action-first instruction: the first assistant turn must be a {action_name} tool call.",
                f"Plain text before a {action_name} tool call receives zero reward and ends the rollout.",
                f"Output no explanation before the first {action_name} tool call.",
            ]
        )
        if prompt_style in PREDICT_ROTATE_PROMPT_STYLES:
            lines.append(f"For this prediction task, choose exactly one {action_name} call; do not inspect or finish.")
        elif prompt_style in CANDIDATE_MULTISTEP_PROMPT_STYLES:
            lines.append(
                f"For this candidate-path task, make a {action_name} call immediately. "
                "After the tool observation, keep calling select_candidate until solved "
                "or the move budget is exhausted; do not inspect or finish."
            )
        elif prompt_style in CANDIDATE_TOOL_PROMPT_STYLES:
            lines.append(
                f"For this candidate-tournament task, choose exactly one {action_name} call; do not inspect or finish."
            )
        elif prompt_style in FACE_DISCOVERY_PROMPT_STYLES:
            lines.append(f"For this face-discovery task, choose exactly one {action_name} call; do not inspect or finish.")
        elif depth == 1:
            lines.append(
                f"For this one-turn scramble, inspect is unnecessary; choose exactly one {action_name} call."
            )
        elif prompt_style in FRONTIER_SENSOR_PROMPT_STYLES:
            lines.append(f"For this frontier task, choose exactly one {action_name} call; do not inspect or finish.")
        else:
            lines.append("Inspect only after making at least one rotate call.")
    if prompt_style in NATIVE_TOOL_PROMPT_STYLES:
        if prompt_style in CANDIDATE_INDEX_PROMPT_STYLES:
            lines.extend(
                [
                    "Native candidate-index mode: call the select_candidate_index tool directly.",
                    "Do not write JSON, prose, or analysis before the tool call.",
                    "The tool schema constrains index to 1-4.",
                ]
            )
        elif prompt_style in CANDIDATE_SELECT_PROMPT_STYLES:
            lines.extend(
                [
                    "Native candidate mode: call the select_candidate tool directly.",
                    "Do not write JSON, prose, or analysis before the tool call.",
                    "The tool schema constrains index to 1-4 and direction to cw or ccw.",
                ]
            )
            if prompt_style in CANDIDATE_MULTISTEP_PROMPT_STYLES:
                lines.append(
                    "This is a multi-call candidate path: use the refreshed candidate table after each tool observation."
                )
        elif prompt_style in PREDICT_ROTATE_PROMPT_STYLES:
            lines.extend(
                [
                    "Native prediction mode: call the predict_rotate tool directly.",
                    "Do not write JSON, prose, or analysis before the tool call.",
                    "The tool schema constrains face to A-L and direction to cw or ccw.",
                ]
            )
        elif prompt_style in REASONED_NATIVE_TOOL_PROMPT_STYLES:
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
    if prompt_style in FRONTIER_SENSOR_PROMPT_STYLES:
        lines.append(_frontier_sensor_objective_section())
    if prompt_style in FACE_DISCOVERY_PROMPT_STYLES:
        lines.append(
            _face_discovery_objective_section(
                candidate_faces,
                index_tool_name=(
                    "select_candidate_index"
                    if prompt_style in CANDIDATE_INDEX_PROMPT_STYLES
                    else "select_candidate"
                    if prompt_style in CANDIDATE_SELECT_PROMPT_STYLES
                    else None
                ),
                frontier_equivalence=prompt_style
                in CANDIDATE_FRONTIER_EQUIVALENCE_PROMPT_STYLES
                | CANDIDATE_GEOMETRY_FRONTIER_PROMPT_STYLES,
                multi_step=prompt_style in CANDIDATE_MULTISTEP_PROMPT_STYLES,
            )
        )
    if prompt_style == "stage_predict_transition_native_tool":
        lines.append(_predict_transition_objective_section())
    elif prompt_style in PREDICT_ROTATE_PROMPT_STYLES:
        lines.append(_predict_rotate_objective_section())
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
    if prompt_style in {
        "stage_candidate_scorecard_native_tool",
        "stage_candidate_scorecard_no_frontier_native_tool",
        "stage_candidate_scorecard_mask_native_tool",
        "stage_candidate_scorecard_mask_index_native_tool",
        "stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
    }:
        lines.append(
            _candidate_scorecard_section(
                puzzle,
                candidate_faces,
                include_frontier=prompt_style == "stage_candidate_scorecard_native_tool",
                include_support=prompt_style
                not in {
                    "stage_candidate_scorecard_mask_native_tool",
                    "stage_candidate_scorecard_mask_index_native_tool",
                    "stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
                },
            )
        )
    if prompt_style in {
        "stage_candidate_relative_flow_rule_frontier_native_tool",
        "stage_candidate_relative_flow_rule_solve2_native_tool",
    }:
        lines.append(
            _candidate_relative_flow_rule_section(
                multi_step=prompt_style in CANDIDATE_MULTISTEP_PROMPT_STYLES
            )
        )
    if prompt_style in {
        "stage_candidate_relative_flow_frontier_native_tool",
        "stage_candidate_relative_flow_rule_frontier_native_tool",
        "stage_candidate_relative_flow_rule_solve2_native_tool",
    }:
        lines.append(_candidate_relative_flow_section(puzzle, candidate_faces))
    if prompt_style in CANDIDATE_STRIPS_PROMPT_STYLES:
        lines.append(_candidate_strips_section(puzzle))
    if prompt_style in FACE_HINT_PROMPT_STYLES:
        lines.append(_face_hint_section(face_hint))
    if prompt_style in DIRECTION_FLOW_PROMPT_STYLES:
        lines.append(_direction_flow_section(puzzle, face_hint))
    if prompt_style in SOLVE_DIRECTION_FLOW_PROMPT_STYLES:
        lines.append(_solve_direction_flow_section(puzzle, face_hint))
    if prompt_style in SOLVE_DIRECTION_FLOW_V2_PROMPT_STYLES:
        lines.append(_all_candidate_solve_direction_flow_section(puzzle))
    if prompt_style in SOLVE_ACTION_TABLE_PROMPT_STYLES:
        lines.append(_solve_action_table_section(puzzle))
    if prompt_style in SOLVE_ACTION_MASK_PROMPT_STYLES:
        lines.append(_solve_action_mask_table_section(puzzle))
    if prompt_style in CHOICE_JSON_PROMPT_STYLES:
        action_menu = _json_action_menu_for_face(face_hint) if face_hint else _json_action_menu()
        lines.extend(
            [
                "Choose exactly one legal action from this menu and copy it exactly with no bullet:",
                action_menu,
            ]
        )
    if prompt_style in {
        "stage_frontier_sensor_compact_native_tool",
        "stage_predict_rotate_native_tool",
        "stage_predict_transition_native_tool",
        "stage_face_discovery_native_tool",
        "stage_face_tournament_native_tool",
        "stage_candidate_tournament_native_tool",
        "stage_candidate_index_native_tool",
        "stage_candidate_scorecard_native_tool",
        "stage_candidate_scorecard_no_frontier_native_tool",
        "stage_candidate_scorecard_mask_native_tool",
        "stage_candidate_scorecard_mask_index_native_tool",
        "stage_candidate_scorecard_mask_frontier_equivalence_native_tool",
        "stage_candidate_geometry_frontier_native_tool",
        "stage_candidate_relative_flow_frontier_native_tool",
        "stage_candidate_relative_flow_rule_frontier_native_tool",
        "stage_candidate_relative_flow_rule_solve2_native_tool",
    }:
        lines.append("Initial compact observation: use the sensor sections above.")
    else:
        lines.extend(["Initial observation:", puzzle.render_net()])
    return "\n".join(lines)


def _curriculum_hint(depth: int, prompt_style: str = "default") -> str:
    if prompt_style in CANDIDATE_MULTISTEP_PROMPT_STYLES:
        if depth == 1:
            return (
                "Curriculum hint: this is a one-turn scramble. One clean "
                "select_candidate call should solve immediately."
            )
        return (
            "Curriculum hint: this is a two-turn scramble. The first clean "
            "select_candidate call should leave a one-turn solve; then use the "
            "refreshed relative-flow table to make the final solving call."
        )
    if (
        prompt_style
        in CANDIDATE_FRONTIER_EQUIVALENCE_PROMPT_STYLES
        | CANDIDATE_GEOMETRY_FRONTIER_PROMPT_STYLES
    ):
        if depth == 1:
            return (
                "Curriculum hint: this is a one-turn scramble. A clean select_candidate "
                "call should choose the slot and direction that solves immediately."
            )
        return (
            "Curriculum hint: the puzzle was scrambled by two short face turns. "
            "A clean select_candidate call gets full credit for solving now and high "
            "credit for any equivalent first move that leaves a one-turn solve."
        )
    if prompt_style in CANDIDATE_INDEX_PROMPT_STYLES:
        if depth == 1:
            return (
                "Curriculum hint: this is a one-turn scramble. Exactly one "
                "select_candidate_index call can choose the slot whose face explains the disturbance."
            )
        return (
            "Curriculum hint: the puzzle was scrambled by a short sequence of face turns. "
            "Choose the candidate slot that best explains the visible disturbance."
        )
    if prompt_style in CANDIDATE_SELECT_PROMPT_STYLES:
        if depth == 1:
            return (
                "Curriculum hint: this is a one-turn scramble. Exactly one "
                "select_candidate call can choose the slot whose face explains the disturbance."
            )
        return (
            "Curriculum hint: the puzzle was scrambled by a short sequence of face turns. "
            "Choose the candidate slot that best explains the first reverse step."
        )
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


def _frontier_sensor_objective_section() -> str:
    return "\n".join(
        [
            "Frontier sensor objective:",
            (
                "Use only the affected-face summaries and candidate-local moved strips "
                "shown below."
            ),
            (
                "Choose one action that either solves now or moves the puzzle "
                "toward a state that can be solved by one later rotate."
            ),
            (
                "Decision rule: for candidate X with ring n0 n1 n2 n3 n4, X:cw is "
                "supported at n_i when n_i's strip is n_{i+1} repeated three times; "
                "X:ccw is supported when it is n_{i-1} repeated three times."
            ),
            (
                "Choose an action with the most supported ring positions. "
                "Five supported positions solves now; for depth 2, a best-supported "
                "action should expose a one-turn solve."
            ),
            (
                "Do not output text. Call the required native tool once with the face and direction "
                "supported by the sticker geometry."
            ),
        ]
    )


def _face_discovery_objective_section(
    candidate_faces: Sequence[str] | None = None,
    *,
    index_tool_name: str | None = None,
    frontier_equivalence: bool = False,
    multi_step: bool = False,
) -> str:
    rows = ["Face-discovery objective:"]
    if candidate_faces:
        if index_tool_name:
            rows.extend(
                f"Candidate {index}: {face}"
                for index, face in enumerate(candidate_faces, start=1)
            )
            if multi_step:
                rows.append(
                    "Choose one printed candidate slot per tool call; non-candidate slots or actions after the move budget receive zero reward."
                )
            else:
                rows.append(
                    "Choose only one candidate slot; non-candidate slots or extra actions receive zero reward."
                )
        else:
            rows.append("Candidate faces: " + " ".join(candidate_faces))
            rows.append("Choose only one of these candidate faces; other faces receive zero reward.")
    rows.extend(
        [
            (
                "Use the affected-face summaries and candidate-local moved strips below "
                "to identify which face turn produced the visible disturbance."
            ),
            (
                (
                    "Call select_candidate_index exactly once with the chosen slot index. "
                    "This curriculum scores whether the selected slot maps to the correct face."
                )
                if index_tool_name == "select_candidate_index"
                else (
                    "Call select_candidate with the chosen slot index and direction. "
                    "After the tool refreshes the candidate table, call select_candidate again "
                    "if the puzzle is not solved and moves remain."
                )
                if multi_step and index_tool_name == "select_candidate"
                else (
                    "Call select_candidate exactly once with the chosen slot index and direction. "
                    "This curriculum rewards any clean action that solves now or leaves the puzzle "
                    "one rotate from solved; equivalent first moves are acceptable."
                )
                if frontier_equivalence and index_tool_name == "select_candidate"
                else (
                    "Call select_candidate exactly once with the chosen slot index and direction. "
                    "This curriculum scores whether the selected slot maps to the correct face."
                )
                if index_tool_name == "select_candidate"
                else (
                    "Call rotate exactly once. Direction is still part of the tool call, but "
                    "this curriculum focuses on discovering the correct face from geometry."
                )
            ),
            (
                "Prefer a face whose five moved strips align with affected stickers around "
                "its printed neighbor ring. Do not use inspect or finish."
            ),
        ]
    )
    return "\n".join(rows)


def _predict_rotate_objective_section() -> str:
    return "\n".join(
        [
            "Predict-then-rotate objective:",
            (
                "Call predict_rotate(face, direction, predicted_after) exactly once. "
                "predicted_after must be five three-letter strips after your chosen move, "
                "listed in the chosen face's printed ring order."
            ),
            (
                "Example predicted_after shape for a face with ring B C D E F: "
                "[\"CCC\",\"DDD\",\"EEE\",\"FFF\",\"BBB\"]."
            ),
            (
                "The prediction is scored against the simulator for your chosen move, "
                "and the chosen move is scored by hidden frontier progress."
            ),
        ]
    )


def _predict_transition_objective_section() -> str:
    return "\n".join(
        [
            "Transition-prediction objective:",
            (
                "Call predict_rotate(face, direction, predicted_after) exactly once. "
                "predicted_after must be five three-letter strips after your chosen move, "
                "listed in the chosen face's printed ring order."
            ),
            (
                "Use the candidate-local moved strips below. If a chosen face has printed ring "
                "n0 n1 n2 n3 n4, then cw moves the strip from n4 to n0 and shifts the other "
                "neighbor strips one slot forward; ccw moves the strip from n0 to n4 and shifts "
                "the other strips one slot backward."
            ),
            (
                "This stage rewards only the accuracy of your local after-state prediction for "
                "the move you choose. It is a physics transition curriculum."
            ),
        ]
    )


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


def _relative_flow_delta(puzzle: MegaminxPuzzle, candidate: str, side: int) -> tuple[str, str]:
    ring = DEFAULT_TOPOLOGY.neighbor_rings[candidate]
    destination = ring[side]
    strip = "".join(
        puzzle.stickers[position]
        for position in DEFAULT_TOPOLOGY.side_strip(candidate, destination)
    )
    if len(set(strip)) != 1 or strip[0] not in ring:
        return "mixed", strip

    source_index = ring.index(strip[0])
    raw_delta = (source_index - side) % len(ring)
    delta = raw_delta if raw_delta <= len(ring) // 2 else raw_delta - len(ring)
    return f"{delta:+d}" if delta else "0", strip


def _relative_flow_count(puzzle: MegaminxPuzzle, candidate: str, direction: str) -> int:
    expected = "+1" if direction == "cw" else "-1"
    return sum(
        _relative_flow_delta(puzzle, candidate, side)[0] == expected
        for side in range(len(DEFAULT_TOPOLOGY.neighbor_rings[candidate]))
    )


def _opposite_direction(direction: str) -> str:
    return "ccw" if direction == "cw" else "cw"


def _candidate_relative_flow_rule_section(*, multi_step: bool = False) -> str:
    return "\n".join(
        [
            "Relative-flow counting rule:",
            (
                "For every printed row, count how many tokens have delta +1 and how "
                "many have delta -1."
            ),
            (
                "Choose direction cw for the row with the largest +1 count, or ccw "
                "for the row with the largest -1 count."
            ),
            (
                "On a one-turn scramble, five matching tokens means that slot and "
                "direction solves immediately. On a two-turn scramble, the largest "
                "count is the local reverse step to try."
            ),
            (
                "Use only the printed flow tokens and call select_candidate once for this observation."
                if multi_step
                else "Use only the printed flow tokens and call select_candidate exactly once."
            ),
        ]
    )


def _candidate_relative_flow_section(
    puzzle: MegaminxPuzzle,
    candidate_faces: Sequence[str] | None,
) -> str:
    lines = [
        "Candidate relative-flow table:",
        (
            "For each printed candidate, ring lists the five touching neighbors in order. "
            "Each flow token is destination:delta(strip), where delta is the source offset "
            "inside that candidate ring: +1 means next neighbor, -1 means previous neighbor, "
            "0 means unchanged, and mixed means the three touching stickers are not a repeated ring label."
        ),
        (
            "A cw solve matches +1 tokens; a ccw solve matches -1 tokens. "
            "Use only this visible coordinate transform plus the printed candidates."
        ),
        "slot | face | ring | relative_flow",
    ]
    faces = list(candidate_faces) if candidate_faces else list(FACES)
    for slot, face in enumerate(faces, start=1):
        if face not in FACES:
            continue
        ring = DEFAULT_TOPOLOGY.neighbor_rings[face]
        tokens = []
        for side, destination in enumerate(ring):
            delta, strip = _relative_flow_delta(puzzle, face, side)
            tokens.append(f"{destination}:{delta}({strip})")
        lines.append(f"{slot} | {face} | {' '.join(ring)} | {' '.join(tokens)}")
    return "\n".join(lines)


def _candidate_scorecard_section(
    puzzle: MegaminxPuzzle,
    candidate_faces: Sequence[str] | None,
    *,
    include_frontier: bool = True,
    include_support: bool = True,
) -> str:
    columns = ["slot", "face"]
    if include_support:
        columns.append("support")
    columns.append("affected_neighbors")
    if include_frontier:
        columns.append("frontier")
    columns.extend(["cw_mask", "ccw_mask"])
    lines = [
        "Candidate scorecard:",
        (
            "Rows list only the printed candidate slots. affected_neighbors counts how many "
            "affected faces lie on the candidate ring. cw_mask and ccw_mask are five-bit "
            "local strip-equality masks."
        ),
        " | ".join(columns),
    ]
    affected = set(_affected_faces(puzzle))
    mask_counts = _action_mask_counts(puzzle)
    faces = list(candidate_faces) if candidate_faces else list(FACES)
    for slot, face in enumerate(faces, start=1):
        if face not in FACES:
            continue
        ring = DEFAULT_TOPOLOGY.neighbor_rings[face]
        cw_mask = "".join(str(bit) for bit in _action_mask_bits(puzzle, face, "cw"))
        ccw_mask = "".join(str(bit) for bit in _action_mask_bits(puzzle, face, "ccw"))
        support = max(mask_counts[(face, "cw")], mask_counts[(face, "ccw")])
        affected_neighbors = len(set(ring) & affected)
        row_parts = [str(slot), face]
        if include_support:
            row_parts.append(str(support))
        row_parts.append(str(affected_neighbors))
        if include_frontier:
            frontier = int(_face_frontier_viable(puzzle, face))
            row_parts.append(str(frontier))
        row_parts.extend([cw_mask, ccw_mask])
        lines.append(" | ".join(row_parts))
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


def _all_candidate_solve_direction_flow_section(puzzle: MegaminxPuzzle) -> str:
    lines = [
        "All-candidate solve-direction flow table:",
        (
            "Rows are local one-turn solve hypotheses for every candidate face. "
            "For depth 1, one candidate direction should match all five rows. "
            "For depth 2, no candidate may match perfectly; choose the first rotate "
            "that restores the most local consistency."
        ),
        (
            "candidate | destination | solved_target | current | "
            "expected_current_if_solve_cw | expected_current_if_solve_ccw"
        ),
    ]
    for candidate in FACES:
        ring = DEFAULT_TOPOLOGY.neighbor_rings[candidate]
        lines.append(f"{candidate} ring: " + " ".join(ring))
        for side, destination in enumerate(ring):
            previous = ring[(side - 1) % len(ring)]
            next_face = ring[(side + 1) % len(ring)]
            strip = [
                puzzle.stickers[position]
                for position in DEFAULT_TOPOLOGY.side_strip(candidate, destination)
            ]
            lines.append(
                f"{candidate} | {destination} | solved_target={destination * 3} | "
                f"current={''.join(strip)} | expected_current_if_solve_cw={next_face * 3} | "
                f"expected_current_if_solve_ccw={previous * 3}"
            )
    return "\n".join(lines)


def _solve_action_table_section(puzzle: MegaminxPuzzle) -> str:
    lines = [
        "Solve-action evidence table:",
        (
            "Each row is one rotate action. Compare each current strip to that "
            "row's expected strip; for depth 1, exactly one action should match "
            "all five destination strips."
        ),
        "action | evidence",
    ]
    for candidate in FACES:
        ring = DEFAULT_TOPOLOGY.neighbor_rings[candidate]
        for direction in DIRECTIONS:
            parts = []
            for side, destination in enumerate(ring):
                previous = ring[(side - 1) % len(ring)]
                next_face = ring[(side + 1) % len(ring)]
                expected = next_face if direction == "cw" else previous
                strip = [
                    puzzle.stickers[position]
                    for position in DEFAULT_TOPOLOGY.side_strip(candidate, destination)
                ]
                parts.append(
                    f"{destination} current={''.join(strip)} expected={expected * 3}"
                )
            lines.append(f"{candidate}:{direction} | " + " | ".join(parts))
    return "\n".join(lines)


def _solve_action_mask_table_section(puzzle: MegaminxPuzzle) -> str:
    lines = [
        "Solve-action equality-mask table:",
        (
            "Each row lists one candidate face and two five-bit masks. "
            "A bit is 1 when that local strip equality holds for the action; 0 otherwise."
        ),
        (
            "For depth 1, one action should have mask 11111. "
            "For depth 2, choose a rotate action with many 1 bits; ties are possible."
        ),
        "face | ring | cw_mask | ccw_mask",
    ]
    for candidate in FACES:
        ring = DEFAULT_TOPOLOGY.neighbor_rings[candidate]
        cw_mask = "".join(str(bit) for bit in _action_mask_bits(puzzle, candidate, "cw"))
        ccw_mask = "".join(str(bit) for bit in _action_mask_bits(puzzle, candidate, "ccw"))
        lines.append(f"{candidate} | {' '.join(ring)} | {cw_mask} | {ccw_mask}")
    return "\n".join(lines)


def _action_mask_counts(puzzle: MegaminxPuzzle) -> dict[Move, int]:
    return {
        (face, direction): sum(_action_mask_bits(puzzle, face, direction))
        for face in FACES
        for direction in DIRECTIONS
    }


def _action_mask_bits(
    puzzle: MegaminxPuzzle,
    candidate: str,
    direction: str,
) -> tuple[int, ...]:
    ring = DEFAULT_TOPOLOGY.neighbor_rings[candidate]
    bits: list[int] = []
    for side, destination in enumerate(ring):
        previous = ring[(side - 1) % len(ring)]
        next_face = ring[(side + 1) % len(ring)]
        expected = next_face if direction == "cw" else previous
        strip = [
            puzzle.stickers[position]
            for position in DEFAULT_TOPOLOGY.side_strip(candidate, destination)
        ]
        bits.append(int("".join(strip) == expected * 3))
    return tuple(bits)


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
