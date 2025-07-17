import numpy as np
from typing import Any, Callable
from abc import ABC, abstractmethod
from pathfinding.core.grid import Grid  # type: ignore
from PIL import Image
from .core import STAY, UP, DOWN, LEFT, RIGHT, HEIGHT, WIDTH
from .env import (
    find_path,
    WALL_IDX,
    EXIT_IDX,
    DEFENDER_IDX,
    ExitEnv,
)

DEFAULT_STUPIDITY = 3


class AttackerAgentBase(ABC):
    @abstractmethod
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        return [STAY, UP, DOWN, LEFT, RIGHT]

    @abstractmethod
    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        return [STAY, UP, DOWN, LEFT, RIGHT]

    @abstractmethod
    def reset_peek(self):
        return


class UserAttacker(AttackerAgentBase):
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        while True:
            Image.fromarray(ExitEnv.render_observation_rgb(observation[0])).save(
                "observation_user_attacker.png"
            )
            action = input("Select action for `UserAttacker`: ")

            if action == "s":
                action = STAY
            elif action == "u":
                action = UP
            elif action == "d":
                action = DOWN
            elif action == "l":
                action = LEFT
            elif action == "r":
                action = RIGHT
            else:
                continue
            return [action]

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        return self.get_action(observation=observation)

    def reset_peek(self):
        return


class IdleAttacker(AttackerAgentBase):
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        return [STAY]

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        return self.get_action(observation=observation)

    def reset_peek(self):
        return


class NaiveExitAttacker(AttackerAgentBase):
    """Charges straight to the true exit."""

    def __init__(self):
        self._grid: Grid | None = None

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        walls = observation[0][WALL_IDX]
        if self._grid is None:
            self._grid = Grid(matrix=1 - walls)

        exits = observation[0][EXIT_IDX]
        exit_pos = np.argwhere(exits)[0]
        exit_pos: tuple[int, int] = (exit_pos[0], exit_pos[1])

        attacker_pos = observation[1]

        candidate_positions: list[tuple[tuple[int, int], int]] = [
            (attacker_pos, STAY),
            ((attacker_pos[0] - 1, attacker_pos[1]), UP),
            ((attacker_pos[0] + 1, attacker_pos[1]), DOWN),
            ((attacker_pos[0], attacker_pos[1] - 1), LEFT),
            ((attacker_pos[0], attacker_pos[1] + 1), RIGHT),
        ]

        min_distance: int = 99999
        min_candidates: list[int] = []
        for candidate_position, candidate_action in candidate_positions:
            if candidate_position[0] < 0 or candidate_position[0] >= HEIGHT:
                continue
            if candidate_position[1] < 0 or candidate_position[1] >= WIDTH:
                continue
            if walls[candidate_position]:
                continue
            distance = len(find_path(self._grid, candidate_position, exit_pos))
            if distance < min_distance:
                min_candidates = [candidate_action]
                min_distance = distance
            elif distance == min_distance:
                min_candidates.append(candidate_action)

        return min_candidates

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        return self.get_action(observation=observation)

    def reset_peek(self):
        return


class EvadeAttacker(AttackerAgentBase):
    def __init__(self):
        self._grid: Grid | None = None

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        walls = observation[0][WALL_IDX]
        if self._grid is None:
            self._grid = Grid(matrix=1 - walls)

        defenders = observation[0][DEFENDER_IDX]
        defender_pos = np.argwhere(defenders)[0]
        defender_pos: tuple[int, int] = (defender_pos[0], defender_pos[1])

        attacker_pos = observation[1]

        candidate_positions: list[tuple[tuple[int, int], int]] = [
            (attacker_pos, STAY),
            ((attacker_pos[0] - 1, attacker_pos[1]), UP),
            ((attacker_pos[0] + 1, attacker_pos[1]), DOWN),
            ((attacker_pos[0], attacker_pos[1] - 1), LEFT),
            ((attacker_pos[0], attacker_pos[1] + 1), RIGHT),
        ]

        max_distance: int = -1
        max_candidates: list[int] = []
        for candidate_position, candidate_action in candidate_positions:
            if candidate_position[0] < 0 or candidate_position[0] >= HEIGHT:
                continue
            if candidate_position[1] < 0 or candidate_position[1] >= WIDTH:
                continue
            if walls[candidate_position]:
                continue
            distance = len(find_path(self._grid, candidate_position, defender_pos))
            if distance > max_distance:
                max_candidates = [candidate_action]
                max_distance = distance
            elif distance == max_distance:
                max_candidates.append(candidate_action)

        return max_candidates

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        return self.get_action(observation=observation)

    def reset_peek(self):
        return


class SwitchAttacker(AttackerAgentBase):
    def __init__(
        self,
        condition: Callable[
            [tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]]], bool
        ],
        true: AttackerAgentBase,
        false: AttackerAgentBase,
    ):
        self._condition = condition
        self._true = true
        self._false = false

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        # If `self._true` and `self._false` have a mutual action, prefer that.
        # Reset peek to ensure we are getting the immediate next action peek.
        self.reset_peek()
        true_candidates = self._true.peek_action(observation=observation)
        false_candidates = self._false.peek_action(observation=observation)
        # Reset peek again to ensure the next call to `peek_action()` will return the immediate next action peek.
        self.reset_peek()
        mutual_candidates: list[int] = []
        for candidate in true_candidates:
            if candidate in false_candidates:
                mutual_candidates.append(candidate)
        if len(mutual_candidates) > 0:
            if self._condition(observation):
                _ = self._true.get_action(observation=observation)
            else:
                _ = self._false.get_action(observation=observation)
            return mutual_candidates
        if self._condition(observation):
            return self._true.get_action(observation=observation)
        return self._false.get_action(observation=observation)

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        true_candidates = self._true.peek_action(observation=observation)
        false_candidates = self._false.peek_action(observation=observation)

        if self._condition(observation):
            self._false.reset_peek()
        else:
            self._true.reset_peek()

        mutual_candidates: list[int] = []
        for candidate in true_candidates:
            if candidate in false_candidates:
                mutual_candidates.append(candidate)

        if len(mutual_candidates) > 0:
            return mutual_candidates
        if self._condition(observation):
            return true_candidates
        return false_candidates

    def reset_peek(self):
        self._true.reset_peek()
        self._false.reset_peek()


class DistanceSwitchAttacker(SwitchAttacker):
    def __init__(
        self,
        trigger_distance: int,
        greater: AttackerAgentBase,
        lesser: AttackerAgentBase,
    ):
        self._trigger_distance = trigger_distance
        self._grid: Grid | None = None

        def _distance_condition(
            observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        ) -> bool:
            walls = observation[0][WALL_IDX]
            if self._grid is None:
                self._grid = Grid(matrix=1 - walls)

            defenders = observation[0][DEFENDER_IDX]
            defender_pos = np.argwhere(defenders)[0]
            defender_pos: tuple[int, int] = (defender_pos[0], defender_pos[1])

            attacker_pos = observation[1]

            distance = len(find_path(self._grid, defender_pos, attacker_pos))
            return distance >= self._trigger_distance

        super().__init__(condition=_distance_condition, true=greater, false=lesser)


class StupidAttacker(AttackerAgentBase):
    def __init__(self, inner: AttackerAgentBase, stupidity: int):
        self._inner = inner
        self._stupidity = stupidity
        self._counter = 0
        self._mock_counter = 0

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        if self._counter % self._stupidity == 0:
            self._counter += 1
            self._mock_counter = self._counter
            return self._inner.get_action(observation=observation)
        self._counter += 1
        self.reset_peek()
        return [STAY]

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        if self._mock_counter % self._stupidity == 0:
            self._mock_counter += 1
            return self._inner.peek_action(observation=observation)
        self._mock_counter += 1
        return [STAY]

    def reset_peek(self):
        self._mock_counter = self._counter
        self._inner.reset_peek()
