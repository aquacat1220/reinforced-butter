import numpy as np
from typing import Any
from abc import ABC, abstractmethod
from pathfinding.core.grid import Grid  # type: ignore
from .core import STAY, UP, DOWN, LEFT, RIGHT
from .env import find_path
from .env import WALL_IDX, POWER_IDX, PLAYER_IDX

DEFAULT_STUPIDITY = 3


class GhostAgentBase(ABC):
    @abstractmethod
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> int:
        return STAY


class IdleGhost(GhostAgentBase):
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> int:
        return STAY


class PursueGhost(GhostAgentBase):
    def __init__(self):
        self._grid: Grid | None = None

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> int:
        walls = observation[0][WALL_IDX]
        if self._grid is None:
            self._grid = Grid(matrix=1 - walls)

        player = observation[0][PLAYER_IDX]
        # Follow the first player to be found.
        # The environment contains only one player anyways, so no need to worry.
        player_pos: tuple[int, int] = np.argwhere(player)[0]
        my_pos = observation[1]
        actions = find_path(self._grid, my_pos, player_pos)
        # assert (
        #     len(actions) > 0
        # )  # Since we always have a single player in the game, and we must haven't catch it yet, we can always find a path to it.
        # The above was true until I added `PreviewWrapper`, where the ghost could preview into the future.
        if len(actions) == 0:
            return STAY
        return actions[0]


class StupidPursueGhost(PursueGhost):
    def __init__(self, stupidity: int = DEFAULT_STUPIDITY):
        super().__init__()
        self._counter = 0
        self._stupidity = stupidity

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> int:
        action: int = STAY
        if self._counter % self._stupidity == 0:
            action = super().get_action(observation)
        self._counter += 1
        return action


class PatrolPowerGhost(GhostAgentBase):
    def __init__(self):
        self._target_power_index: int = 0
        self._grid: Grid | None = None

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> int:
        walls = observation[0][WALL_IDX]
        if self._grid is None:
            self._grid = Grid(matrix=1 - walls)

        powers = observation[0][POWER_IDX]
        my_pos = observation[1]
        power_poss: list[tuple[int, int]] = np.argwhere(powers)  # type: ignore
        if len(power_poss) == 0:
            # If no more powers are left, just stay here.
            return STAY
        target_power_pos = power_poss[self._target_power_index % len(power_poss)]
        if np.all(my_pos == target_power_pos):
            # If we arrived at the target power, increment the target power index.
            self._target_power_index += 1
            target_power_pos = power_poss[self._target_power_index % len(power_poss)]
        actions = find_path(self._grid, my_pos, target_power_pos)
        if len(actions) == 0:
            return STAY
        return actions[0]


class DefendPowerGhost(GhostAgentBase):
    """Defends the first power"""

    def __init__(self):
        self._grid: Grid | None = None

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> int:
        walls = observation[0][WALL_IDX]
        if self._grid is None:
            self._grid = Grid(matrix=1 - walls)

        powers = observation[0][POWER_IDX]
        my_pos = observation[1]
        power_poss: list[tuple[int, int]] = np.argwhere(powers)  # type: ignore
        if len(power_poss) == 0:
            # If no more powers are left, just stay here.
            return STAY
        target_power_pos = power_poss[0]
        actions = find_path(self._grid, my_pos, target_power_pos)
        if len(actions) == 0:
            return STAY
        return actions[0]
