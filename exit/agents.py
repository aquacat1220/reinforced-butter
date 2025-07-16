import numpy as np
from typing import Any
from abc import ABC, abstractmethod
from pathfinding.core.grid import Grid  # type: ignore
from .core import STAY, UP, DOWN, LEFT, RIGHT
from .env import find_path, WALL_IDX, DECOY_IDX, EXIT_IDX

DEFAULT_STUPIDITY = 3


class AttackerAgentBase(ABC):
    @abstractmethod
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        return STAY

    @abstractmethod
    def get_mock_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        return self.get_action(observation=observation)


class IdleAttacker(AttackerAgentBase):
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        return STAY

    def get_mock_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        return self.get_action(observation=observation)


class NaiveExitAttacker(AttackerAgentBase):
    """Charges straight to the true exit."""

    def __init__(self):
        self._grid: Grid | None = None

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        walls = observation[0][WALL_IDX]
        if self._grid is None:
            self._grid = Grid(matrix=1 - walls)

        exits = observation[0][EXIT_IDX]
        exit_pos = np.argwhere(exits)[0]
        exit_pos: tuple[int, int] = (exit_pos[0], exit_pos[1])

        actions = find_path(
            self._grid,
            observation[1],
            exit_pos,
        )

        if len(actions) == 0:
            return STAY
        return actions[0]

    def get_mock_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        return self.get_action(observation=observation)


class StupidAttacker(AttackerAgentBase):
    def __init__(self, inner: AttackerAgentBase, stupidity: int):
        self._inner = inner
        self._stupidity = stupidity
        self._counter = 0
        self._mock_counter = 0

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        if self._counter % self._stupidity == 0:
            self._counter += 1
            self._mock_counter = self._counter
            return self._inner.get_action(observation=observation)
        self._counter += 1
        self._mock_counter = self._counter
        return STAY

    def get_mock_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        if self._mock_counter % self._stupidity == 0:
            self._mock_counter += 1
            return self._inner.get_mock_action(observation=observation)
        self._mock_counter += 1
        return STAY
