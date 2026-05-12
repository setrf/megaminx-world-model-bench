from __future__ import annotations

from dataclasses import dataclass
from math import atan2, sqrt
from random import Random
from typing import Iterable, Sequence

FACES: tuple[str, ...] = tuple("ABCDEFGHIJKL")
DIRECTIONS: tuple[str, ...] = ("cw", "ccw")
CENTER_INDEX = 0
CORNER_OFFSET = 1
EDGE_OFFSET = 6
POSITIONS_PER_FACE = 11
STICKERS_PER_PUZZLE = len(FACES) * POSITIONS_PER_FACE
EDGE_COUNT = 30
CORNER_COUNT = 20
PIECE_COUNT = EDGE_COUNT + CORNER_COUNT

Move = tuple[str, str]
StickerPos = tuple[str, int]
Vec3 = tuple[float, float, float]


def flip_direction(direction: str) -> str:
    if direction == "cw":
        return "ccw"
    if direction == "ccw":
        return "cw"
    raise ValueError(f"Unknown direction: {direction}")


def inverse_moves(moves: Sequence[Move]) -> list[Move]:
    return [(face, flip_direction(direction)) for face, direction in reversed(moves)]


def move_to_text(move: Move) -> str:
    return f"{move[0]}:{move[1]}"


def moves_to_text(moves: Sequence[Move]) -> str:
    return " ".join(move_to_text(move) for move in moves)


def parse_move_text(text: str) -> Move:
    if ":" not in text:
        raise ValueError(f"Move must be FACE:DIRECTION, got {text!r}")
    face, direction = text.split(":", 1)
    return validate_move(face, direction)


def validate_move(face: str, direction: str) -> Move:
    face = face.strip().upper()
    direction = direction.strip().lower()
    if face not in FACES:
        raise ValueError(f"Face must be one of {''.join(FACES)}, got {face!r}")
    if direction not in DIRECTIONS:
        raise ValueError(f"Direction must be 'cw' or 'ccw', got {direction!r}")
    return face, direction


def generate_scramble(depth: int, rng: Random) -> list[Move]:
    if depth < 0:
        raise ValueError("Scramble depth must be non-negative")
    moves: list[Move] = []
    previous: Move | None = None
    while len(moves) < depth:
        move = (rng.choice(FACES), rng.choice(DIRECTIONS))
        if previous and move[0] == previous[0] and move[1] != previous[1]:
            continue
        moves.append(move)
        previous = move
    return moves


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _mul(a: Vec3, scalar: float) -> Vec3:
    return a[0] * scalar, a[1] * scalar, a[2] * scalar


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vec3) -> float:
    return sqrt(_dot(a, a))


def _normalize(a: Vec3) -> Vec3:
    length = _norm(a)
    return a[0] / length, a[1] / length, a[2] / length


def _distance_squared(a: Vec3, b: Vec3) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))


def _icosahedron_vertices() -> list[Vec3]:
    phi = (1.0 + sqrt(5.0)) / 2.0
    vertices: list[Vec3] = []
    for y in (-1.0, 1.0):
        for z in (-phi, phi):
            vertices.append((0.0, y, z))
    for x in (-1.0, 1.0):
        for y in (-phi, phi):
            vertices.append((x, y, 0.0))
    for x in (-phi, phi):
        for z in (-1.0, 1.0):
            vertices.append((x, 0.0, z))
    return sorted(vertices, key=lambda p: (-p[2], -p[1], p[0]))


@dataclass(frozen=True)
class MegaminxTopology:
    face_vectors: dict[str, Vec3]
    neighbor_rings: dict[str, tuple[str, ...]]
    edge_pieces: tuple[tuple[StickerPos, StickerPos], ...]
    corner_pieces: tuple[tuple[StickerPos, StickerPos, StickerPos], ...]

    @classmethod
    def build(cls) -> "MegaminxTopology":
        vectors = dict(zip(FACES, _icosahedron_vertices(), strict=True))
        closest: dict[str, list[str]] = {}
        for face, vector in vectors.items():
            nearest = sorted(
                (
                    (_distance_squared(vector, other_vector), other_face)
                    for other_face, other_vector in vectors.items()
                    if other_face != face
                ),
                key=lambda item: (round(item[0], 12), item[1]),
            )
            closest[face] = [other_face for _, other_face in nearest[:5]]

        neighbor_rings: dict[str, tuple[str, ...]] = {}
        for face, neighbors in closest.items():
            normal = _normalize(vectors[face])
            reference = (0.0, 0.0, 1.0)
            if abs(_dot(normal, reference)) > 0.9:
                reference = (0.0, 1.0, 0.0)
            u_axis = _normalize(_cross(reference, normal))
            v_axis = _cross(normal, u_axis)

            def angle(neighbor: str) -> float:
                projected = _sub(vectors[neighbor], _mul(normal, _dot(vectors[neighbor], normal)))
                return atan2(_dot(projected, v_axis), _dot(projected, u_axis))

            neighbor_rings[face] = tuple(sorted(neighbors, key=angle))

        for face, ring in neighbor_rings.items():
            if len(ring) != 5:
                raise ValueError(f"Face {face} should have five neighbors")
            for neighbor in ring:
                if face not in neighbor_rings[neighbor]:
                    raise ValueError(f"Neighbor relation {face}-{neighbor} is not reciprocal")

        edges = cls._build_edge_pieces(neighbor_rings)
        corners = cls._build_corner_pieces(neighbor_rings)
        if len(edges) != EDGE_COUNT:
            raise ValueError(f"Expected {EDGE_COUNT} edges, got {len(edges)}")
        if len(corners) != CORNER_COUNT:
            raise ValueError(f"Expected {CORNER_COUNT} corners, got {len(corners)}")
        return cls(
            face_vectors=vectors,
            neighbor_rings=neighbor_rings,
            edge_pieces=tuple(edges),
            corner_pieces=tuple(corners),
        )

    @staticmethod
    def _build_edge_pieces(
        neighbor_rings: dict[str, tuple[str, ...]]
    ) -> list[tuple[StickerPos, StickerPos]]:
        pieces: list[tuple[StickerPos, StickerPos]] = []
        seen: set[tuple[str, str]] = set()
        for face in FACES:
            for side, neighbor in enumerate(neighbor_rings[face]):
                key = tuple(sorted((face, neighbor)))
                if key in seen:
                    continue
                seen.add(key)
                neighbor_side = neighbor_rings[neighbor].index(face)
                pieces.append(
                    (
                        (face, edge_index(side)),
                        (neighbor, edge_index(neighbor_side)),
                    )
                )
        return pieces

    @staticmethod
    def _build_corner_pieces(
        neighbor_rings: dict[str, tuple[str, ...]]
    ) -> list[tuple[StickerPos, StickerPos, StickerPos]]:
        triangle_faces: set[tuple[str, str, str]] = set()
        for face in FACES:
            ring = neighbor_rings[face]
            for side, neighbor in enumerate(ring):
                next_neighbor = ring[(side + 1) % 5]
                if next_neighbor in neighbor_rings[neighbor]:
                    triangle_faces.add(tuple(sorted((face, neighbor, next_neighbor))))

        pieces: list[tuple[StickerPos, StickerPos, StickerPos]] = []
        for triangle in sorted(triangle_faces):
            stickers: list[StickerPos] = []
            for face in triangle:
                others = [other for other in triangle if other != face]
                ring = neighbor_rings[face]
                first = ring.index(others[0])
                second = ring.index(others[1])
                if (first + 1) % 5 == second:
                    corner_side = first
                elif (second + 1) % 5 == first:
                    corner_side = second
                else:
                    raise ValueError(f"Faces {triangle} do not meet at one corner")
                stickers.append((face, corner_index(corner_side)))
            pieces.append(tuple(stickers))  # type: ignore[arg-type]
        return pieces

    def side_strip(self, turned_face: str, neighbor: str) -> tuple[StickerPos, StickerPos, StickerPos]:
        side = self.neighbor_rings[neighbor].index(turned_face)
        return (
            (neighbor, corner_index((side - 1) % 5)),
            (neighbor, edge_index(side)),
            (neighbor, corner_index(side)),
        )


def corner_index(side: int) -> int:
    return CORNER_OFFSET + (side % 5)


def edge_index(side: int) -> int:
    return EDGE_OFFSET + (side % 5)


@dataclass
class MegaminxPuzzle:
    topology: MegaminxTopology
    stickers: dict[StickerPos, str]

    @classmethod
    def solved(cls, topology: MegaminxTopology | None = None) -> "MegaminxPuzzle":
        topology = topology or DEFAULT_TOPOLOGY
        stickers = {
            (face, index): face
            for face in FACES
            for index in range(POSITIONS_PER_FACE)
        }
        return cls(topology=topology, stickers=stickers)

    def copy(self) -> "MegaminxPuzzle":
        return MegaminxPuzzle(topology=self.topology, stickers=dict(self.stickers))

    def apply_moves(self, moves: Iterable[Move]) -> None:
        for face, direction in moves:
            self.apply_move(face, direction)

    def apply_move(self, face: str, direction: str) -> None:
        face, direction = validate_move(face, direction)
        turns = 1 if direction == "cw" else 4
        for _ in range(turns):
            self._apply_clockwise(face)

    def _apply_clockwise(self, face: str) -> None:
        previous = dict(self.stickers)
        for side in range(5):
            self.stickers[(face, corner_index((side + 1) % 5))] = previous[
                (face, corner_index(side))
            ]
            self.stickers[(face, edge_index((side + 1) % 5))] = previous[
                (face, edge_index(side))
            ]

        ring = self.topology.neighbor_rings[face]
        strips = [self.topology.side_strip(face, neighbor) for neighbor in ring]
        for side, destination_strip in enumerate(strips):
            source_strip = strips[(side - 1) % 5]
            for destination, source in zip(destination_strip, source_strip, strict=True):
                self.stickers[destination] = previous[source]

    def is_solved(self) -> bool:
        return all(color == face for (face, _), color in self.stickers.items())

    def sticker_accuracy(self) -> float:
        correct = sum(1 for (face, _), color in self.stickers.items() if color == face)
        return correct / STICKERS_PER_PUZZLE

    def piece_accuracy(self) -> float:
        correct = 0
        for piece in self.topology.edge_pieces:
            expected = {face for face, _ in piece}
            observed = {self.stickers[position] for position in piece}
            correct += int(observed == expected)
        for piece in self.topology.corner_pieces:
            expected = {face for face, _ in piece}
            observed = {self.stickers[position] for position in piece}
            correct += int(observed == expected)
        return correct / PIECE_COUNT

    def face_line(self, face: str) -> str:
        center = self.stickers[(face, CENTER_INDEX)]
        corners = "".join(self.stickers[(face, corner_index(side))] for side in range(5))
        edges = "".join(self.stickers[(face, edge_index(side))] for side in range(5))
        return f"{face}: center={center} corners={corners} edges={edges}"

    def render_net(self) -> str:
        return "\n".join(self.face_line(face) for face in FACES)


DEFAULT_TOPOLOGY = MegaminxTopology.build()
