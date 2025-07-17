import numpy as np
from typing import Any, Callable
from abc import ABC, abstractmethod
from pathfinding.core.grid import Grid  # type: ignore
from PIL import Image
from .core import (
    STAY,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    HEIGHT,
    WIDTH,
    WALL,
    EXIT,
    DECOY,
    ATTACKER,
    DEFENDER,
)
from .env import (
    find_path,
    WALL_IDX,
    EXIT_IDX,
    DECOY_IDX,
    ATTACKER_IDX,
    DEFENDER_IDX,
    ExitEnv,
)

DEFAULT_STUPIDITY = 3


def get_target_position(
    observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    target: np.int8,
) -> tuple[int, int]:
    if target == WALL:
        raise ValueError("There are more than one wall in the game.")
    elif target == EXIT:
        target_idx = EXIT_IDX
    elif target == DECOY:
        target_idx = DECOY_IDX
    elif target == ATTACKER:
        target_idx = ATTACKER_IDX
    elif target == DEFENDER:
        target_idx = DEFENDER_IDX
    else:
        raise ValueError(
            "`target` should be one of `EXIT`, `DECOY`, `ATTACKER`, and `DEFENDER`."
        )
    targets = observation[0][target_idx]
    target_pos = np.argwhere(targets)[0]
    return (target_pos[0], target_pos[1])


def distance_between(
    grid: Grid,
    observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    start: np.int8,
    end: np.int8,
) -> int:
    start_pos = get_target_position(observation, start)
    end_pos = get_target_position(observation, end)
    return len(find_path(grid, start_pos, end_pos))


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


class PursueAttacker(AttackerAgentBase):
    """Charges straight to the target."""

    def __init__(self, target: np.int8 = EXIT):
        self._grid: Grid | None = None
        self._target = target

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        walls = observation[0][WALL_IDX]
        if self._grid is None:
            self._grid = Grid(matrix=1 - walls)

        target_pos: tuple[int, int] = get_target_position(observation, self._target)

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
            distance = len(find_path(self._grid, candidate_position, target_pos))
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
    """Runs away from the target."""

    def __init__(self, target: np.int8 = DEFENDER):
        self._grid: Grid | None = None
        self._target = target

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        walls = observation[0][WALL_IDX]
        if self._grid is None:
            self._grid = Grid(matrix=1 - walls)

        target_pos: tuple[int, int] = get_target_position(observation, self._target)

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
            distance = len(find_path(self._grid, candidate_position, target_pos))
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
            [tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], bool], bool
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
            if self._condition(observation, False):
                _ = self._true.get_action(observation=observation)
            else:
                _ = self._false.get_action(observation=observation)
            return mutual_candidates
        if self._condition(observation, False):
            return self._true.get_action(observation=observation)
        return self._false.get_action(observation=observation)

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        true_candidates = self._true.peek_action(observation=observation)
        false_candidates = self._false.peek_action(observation=observation)

        if self._condition(observation, True):
            self._false.reset_peek()
        else:
            self._true.reset_peek()

        mutual_candidates: list[int] = []
        for candidate in true_candidates:
            if candidate in false_candidates:
                mutual_candidates.append(candidate)

        if len(mutual_candidates) > 0:
            return mutual_candidates
        if self._condition(observation, True):
            return true_candidates
        return false_candidates

    def reset_peek(self):
        self._true.reset_peek()
        self._false.reset_peek()


class DistanceSwitchAttacker(SwitchAttacker):
    def __init__(
        self,
        trigger_distance: int,
        target: np.int8,
        greater: AttackerAgentBase,
        lesser: AttackerAgentBase,
    ):
        self._trigger_distance = trigger_distance
        self._grid: Grid | None = None
        self._target = target

        def _distance_condition(
            observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
            is_peeking: bool,
        ) -> bool:
            walls = observation[0][WALL_IDX]
            if self._grid is None:
                self._grid = Grid(matrix=1 - walls)

            distance = distance_between(self._grid, observation, ATTACKER, self._target)
            return distance >= self._trigger_distance

        super().__init__(condition=_distance_condition, true=greater, false=lesser)


class TimeSwitchAttacker(SwitchAttacker):
    def __init__(
        self, true_after: int, true: AttackerAgentBase, false: AttackerAgentBase
    ):
        self._true_after = true_after
        self._counter = 0
        self._peek_counter = 0

        def _time_condition(
            observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
            is_peeking: bool,
        ) -> bool:
            if is_peeking:
                return self._peek_counter >= self._true_after
            return self._counter >= self._true_after

        super().__init__(condition=_time_condition, true=true, false=false)

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        actions = super().get_action(observation)
        self._counter += 1
        self.reset_peek()
        return actions

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        actions = super().peek_action(observation)
        self._peek_counter += 1
        return actions

    def reset_peek(self):
        super().reset_peek()
        self._peek_counter = self._counter


class StupidAttacker(AttackerAgentBase):
    def __init__(self, inner: AttackerAgentBase, stupidity: int):
        self._inner = inner
        self._stupidity = stupidity
        self._counter = 0
        self._peek_counter = 0

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        if self._counter % self._stupidity == 0:
            self._counter += 1
            self._peek_counter = self._counter
            return self._inner.get_action(observation=observation)
        self._counter += 1
        self.reset_peek()
        return [STAY]

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> list[int]:
        if self._peek_counter % self._stupidity == 0:
            self._peek_counter += 1
            return self._inner.peek_action(observation=observation)
        self._peek_counter += 1
        return [STAY]

    def reset_peek(self):
        self._peek_counter = self._counter
        self._inner.reset_peek()


class NaiveAttacker(DistanceSwitchAttacker):
    """
    Moves towards the target, unless the defender is closer than `safety_distance`.
    """

    def __init__(
        self, safety_distance: int = 5, stupidity: int = 2, target: np.int8 = EXIT
    ):
        super().__init__(
            trigger_distance=safety_distance,
            target=DEFENDER,
            greater=StupidAttacker(PursueAttacker(target=target), stupidity=stupidity),
            lesser=EvadeAttacker(target=DEFENDER),
        )


class DecisiveNaiveAttacker(DistanceSwitchAttacker):
    """
    If `target` is closer than `commit_distance`, ignore the defender and moves to `target`.
    Otherwise moves towards the `target`, unless the defender is closer than `safety_distance`.
    """

    def __init__(
        self,
        safety_distance: int = 5,
        commit_distance: int = 3,
        stupidity: int = 2,
        target: np.int8 = EXIT,
    ):
        super().__init__(
            trigger_distance=commit_distance,
            target=target,
            greater=NaiveAttacker(
                safety_distance=safety_distance, stupidity=stupidity, target=target
            ),
            lesser=StupidAttacker(PursueAttacker(target=target), stupidity=stupidity),
        )


class DeceptiveAttacker(TimeSwitchAttacker):
    """
    Targets the decoy for first before `stop_deception_after`, then targets the exit.
    """

    def __init__(
        self,
        safety_distance: int = 5,
        commit_distance: int = 3,
        stupidity: int = 2,
        stop_deception_after: int = 32,
    ):
        super().__init__(
            true_after=stop_deception_after,
            true=DecisiveNaiveAttacker(
                safety_distance=safety_distance,
                commit_distance=commit_distance,
                stupidity=stupidity,
                target=EXIT,
            ),
            false=DecisiveNaiveAttacker(
                safety_distance=safety_distance,
                commit_distance=commit_distance,
                stupidity=stupidity,
                target=DECOY,
            ),
        )
