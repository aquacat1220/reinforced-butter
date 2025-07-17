import numpy as np
from typing import Any
from enum import Enum

# Each tile is represented as a single 8bit integer.
# The upper four bits are bit flags that mark if a attacker/defender/exit/decoy/wall is present at that tile.
# This is possible because there can be at max one attacker/defender/exit/decoy/wall on a single tile.

NONE = np.int8(0b0)
WALL = np.int8(0b1)
DECOY = np.int8(0b10)
EXIT = np.int8(0b100)
DEFENDER = np.int8(0b1000)
ATTACKER = np.int8(0b10000)


# fmt: off
TEMPLATE: np.ndarray[Any, np.dtype[np.int8]] = np.array(
    [
        [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],
        [WALL, NONE, NONE, NONE, WALL, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, WALL, NONE, NONE, NONE, WALL],
        [WALL, NONE, WALL, NONE, WALL, NONE, WALL, WALL, WALL, WALL, WALL, WALL, WALL, NONE, WALL, NONE, WALL, NONE, WALL],
        [WALL, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, WALL],
        [WALL, NONE, WALL, NONE, WALL, NONE, WALL, WALL, WALL, WALL, WALL, WALL, WALL, NONE, WALL, NONE, WALL, NONE, WALL],
        [WALL, NONE, WALL, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, WALL, NONE, WALL],
        [WALL, NONE, WALL, WALL, WALL, WALL, NONE, WALL, WALL, WALL, WALL, WALL, NONE, WALL, WALL, WALL, WALL, NONE, WALL],
        [WALL, NONE, NONE, NONE, NONE, NONE, NONE, WALL, WALL, WALL, WALL, WALL, NONE, NONE, NONE, NONE, NONE, NONE, WALL],
        [WALL, NONE, WALL, WALL, WALL, WALL, NONE, WALL, WALL, WALL, WALL, WALL, NONE, WALL, WALL, WALL, WALL, NONE, WALL],
        [WALL, NONE, WALL, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, WALL, NONE, WALL],
        [WALL, NONE, WALL, NONE, WALL, NONE, WALL, WALL, WALL, WALL, WALL, WALL, WALL, NONE, WALL, NONE, WALL, NONE, WALL],
        [WALL, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, WALL],
        [WALL, NONE, WALL, NONE, WALL, NONE, WALL, WALL, WALL, WALL, WALL, WALL, WALL, NONE, WALL, NONE, WALL, NONE, WALL],
        [WALL, NONE, NONE, NONE, WALL, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, WALL, NONE, NONE, NONE, WALL],
        [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],
    ]
)
PRESET: np.ndarray[Any, np.dtype[np.int8]] = np.array(
    [
        [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],
        [WALL, NONE, NONE, NONE, WALL, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, WALL, NONE, NONE, NONE, WALL],
        [WALL, NONE, WALL, NONE, WALL, NONE, WALL, WALL, WALL, WALL, WALL, WALL, WALL, NONE, WALL, NONE, WALL, NONE, WALL],
        [WALL, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, DEFENDER, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, WALL],
        [WALL, NONE, WALL, NONE, WALL, NONE, WALL, WALL, WALL, WALL, WALL, WALL, WALL, NONE, WALL, NONE, WALL, NONE, WALL],
        [WALL, EXIT, WALL, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, WALL, EXIT, WALL],
        [WALL, NONE, WALL, WALL, WALL, WALL, NONE, WALL, WALL, WALL, WALL, WALL, NONE, WALL, WALL, WALL, WALL, NONE, WALL],
        [WALL, NONE, NONE, NONE, NONE, NONE, NONE, WALL, WALL, WALL, WALL, WALL, NONE, NONE, NONE, NONE, NONE, NONE, WALL],
        [WALL, NONE, WALL, WALL, WALL, WALL, NONE, WALL, WALL, WALL, WALL, WALL, NONE, WALL, WALL, WALL, WALL, NONE, WALL],
        [WALL, NONE, WALL, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, WALL, NONE, WALL],
        [WALL, NONE, WALL, NONE, WALL, NONE, WALL, WALL, WALL, WALL, WALL, WALL, WALL, NONE, WALL, NONE, WALL, NONE, WALL],
        [WALL, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE, WALL],
        [WALL, NONE, WALL, NONE, WALL, NONE, WALL, WALL, WALL, WALL, WALL, WALL, WALL, NONE, WALL, NONE, WALL, NONE, WALL],
        [WALL, NONE, NONE, NONE, WALL, NONE, NONE, NONE, NONE, ATTACKER, NONE, NONE, NONE, NONE, WALL, NONE, NONE, NONE, WALL],
        [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],
    ]
)

# TEMPLATE: np.ndarray[Any, np.dtype[np.int8]] = np.array(
#     [
#         [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],
#         [WALL, NONE, NONE, DEFENDER, WALL, NONE, NONE, NONE, NONE, NONE, NONE],
#         [WALL, NONE, WALL, NONE, WALL, NONE, WALL, WALL, WALL, WALL, WALL],
#         [WALL, EXIT, NONE, NONE, NONE, NONE, NONE, NONE, NONE, DECOY, NONE],
#         [WALL, NONE, WALL, NONE, WALL, NONE, WALL, WALL, WALL, WALL, WALL],
#         [WALL, NONE, WALL, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE],
#         [WALL, NONE, WALL, WALL, WALL, WALL, NONE, WALL, WALL, WALL, WALL],
#         [WALL, NONE, NONE, NONE, ATTACKER, NONE, NONE, WALL, WALL, WALL, WALL],
#     ]
# )
# fmt: on

HEIGHT = len(TEMPLATE)
WIDTH = len(TEMPLATE[0])

STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4


class Event(Enum):
    ATTACKER_CAPTURED = 0
    ATTACKER_EXITED = 1
    ATTACKER_DECOY = 2


class AgentType(Enum):
    ATTACKER = 0
    DEFENDER = 1


class ExitCore:
    def __init__(self, random_map: bool = True):
        self.attacker: tuple[int, int] | None = None
        self.defender: tuple[int, int] | None = None
        self._random_map = random_map

    def _get_random(self, filter: np.int8 = WALL) -> tuple[int, int]:
        """
        Fetch a random tile position that doesn't have any of the entities mentioned in `filter`.

        Args:
            filter (np.int8, optional): Filter of entities to avoid. Defaults to WALL.

        Returns:
            tuple[int, int]: The position of the selected tile.
        """
        while True:
            h = self._rng.integers(0, HEIGHT, dtype=int)
            w = self._rng.integers(0, WIDTH, dtype=int)
            if not self.map[h, w] & filter:
                return (h, w)

    def _get_adjacent(
        self, position: tuple[int, int], action: int
    ) -> tuple[int, int] | None:
        (pos_h, pos_w) = position
        if action == STAY:
            return (pos_h, pos_w)
        if action == UP:
            if pos_h >= 1:
                return (pos_h - 1, pos_w)
            else:
                return None
        if action == UP:
            if pos_h >= 1:
                return (pos_h - 1, pos_w)
            else:
                return None
        if action == DOWN:
            if pos_h < HEIGHT - 1:
                return (pos_h + 1, pos_w)
            else:
                return None
        if action == LEFT:
            if pos_w >= 1:
                return (pos_h, pos_w - 1)
            else:
                return None
        if action == RIGHT:
            if pos_w < WIDTH - 1:
                return (pos_h, pos_w + 1)
            else:
                return None

    def reset(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        if not self._random_map:
            self.map: np.ndarray[Any, np.dtype[np.int8]] = PRESET.copy()
            attacker_pos = np.argwhere(self.map & ATTACKER)[0]
            self.attacker = (attacker_pos[0], attacker_pos[1])
            defender_pos = np.argwhere(self.map & DEFENDER)[0]
            self.defender = (defender_pos[0], defender_pos[1])
            # `PRESET` has two exits in the map.
            exit_poss = np.argwhere(self.map & EXIT)
            # Convert one of them into a decoy.
            decoy_pos = self._rng.choice(exit_poss)
            decoy_pos: tuple[int, int] = (decoy_pos[0], decoy_pos[1])
            self.map[decoy_pos] ^= EXIT | DECOY
        else:
            self.map: np.ndarray[Any, np.dtype[np.int8]] = TEMPLATE.copy()
            exit_pos = self._get_random(WALL | EXIT | DECOY | ATTACKER | DEFENDER)
            self.map[exit_pos] ^= EXIT
            decoy_pos = self._get_random(WALL | EXIT | DECOY | ATTACKER | DEFENDER)
            self.map[decoy_pos] ^= DECOY
            attacker_pos = self._get_random(WALL | EXIT | DECOY | ATTACKER | DEFENDER)
            self.map[attacker_pos] ^= ATTACKER
            self.attacker = attacker_pos
            defender_pos = self._get_random(WALL | EXIT | DECOY | ATTACKER | DEFENDER)
            self.map[defender_pos] ^= DEFENDER
            self.defender = defender_pos
        self.events: list[Event] = []
        self.terminated: bool = False

    def perform_action(self, agent: AgentType, action: int):
        if self.terminated:
            return
        if agent == AgentType.ATTACKER:
            self.perform_attacker_action(action)
        elif agent == AgentType.DEFENDER:
            self.perform_defender_action(action)
        else:
            raise Exception("Unreachable")

    def perform_defender_action(self, action: int):
        assert self.defender is not None

        if action == STAY:
            return

        new_pos = self._get_adjacent(self.defender, action)
        if new_pos is None:
            # `action` moves defender out of map boundaries.
            return
        if self.map[new_pos] & WALL:
            # `action` moves defender into a  wall.
            return

        if self.map[new_pos] & ATTACKER:
            # `action` moves defender into an attacker. Capture the attacker, and end the game.
            # Remove the defender from old position.
            self.map[self.defender] ^= DEFENDER
            # Place the defender here, and remove the attacker.
            self.map[new_pos] ^= DEFENDER | ATTACKER
            # Update the defender position.
            self.defender = new_pos
            # Update the attacker position.
            self.attacker = None
            self.events.append(Event.ATTACKER_CAPTURED)
            self.terminated = True
            return

        self.map[self.defender] ^= DEFENDER
        self.map[new_pos] ^= DEFENDER
        self.defender = new_pos
        return

    def perform_attacker_action(self, action: int):
        assert self.attacker is not None

        if action == STAY:
            return

        new_pos = self._get_adjacent(self.attacker, action)
        if new_pos is None:
            # `action` moves the attacker out of map boundaries.
            return
        if self.map[new_pos] & WALL:
            # `action` moves the attacker into a wall.
            return

        if self.map[new_pos] & DEFENDER:
            # `action` moves attacker into a defender. Capture the attacker, and end the game.
            self.map[self.attacker] ^= ATTACKER
            self.attacker = None
            self.events.append(Event.ATTACKER_CAPTURED)
            self.terminated = True
            return

        if self.map[new_pos] & EXIT:
            # `action` moves attacker into an exit. Take the exit and end the game.
            self.map[self.attacker] ^= ATTACKER
            self.map[new_pos] ^= ATTACKER
            self.attacker = new_pos
            self.events.append(Event.ATTACKER_EXITED)
            self.terminated = True
            return

        if self.map[new_pos] & DECOY:
            # `action` moves attacker into a decoy.
            self.map[self.attacker] ^= ATTACKER
            self.map[new_pos] ^= ATTACKER
            self.attacker = new_pos
            self.events.append(Event.ATTACKER_DECOY)
            return

        # If player was not killed by a ghost, move player to new position.
        self.map[self.attacker] ^= ATTACKER
        self.map[new_pos] ^= ATTACKER
        self.attacker = new_pos
        return
