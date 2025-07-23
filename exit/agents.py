import numpy as np
from typing import Any, Callable
from abc import ABC, abstractmethod
from pathfinding.core.grid import Grid, GridNode  # type: ignore
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
    return len(find_path(grid, start_pos, end_pos)[1])


class AttackerAgentBase(ABC):
    @abstractmethod
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        return STAY

    @abstractmethod
    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        return STAY

    @abstractmethod
    def reset_peek(self):
        return


class UserAttacker(AttackerAgentBase):
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
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
            return action

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        return self.get_action(observation=observation)

    def reset_peek(self):
        return


class IdleAttacker(AttackerAgentBase):
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        return STAY

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        return self.get_action(observation=observation)

    def reset_peek(self):
        return


class PursueAttacker(AttackerAgentBase):
    """Charges straight to the target."""

    def __init__(
        self,
        seed: int | None = None,
        target: np.int8 = EXIT,
        ignore_defender: bool = True,
    ):
        self._grid: Grid | None = None
        # `self._rng` is the only state that should be kept track for determinism.
        # Since `choice()`-ing from the rng will update the rng in both get and peek,
        # we are effectively sharing the same state in get and peek.
        # Thus we store the inner state of the rng for the get-timeline, and restore it before `get_action()`.
        self._rng = np.random.default_rng(seed)
        self._rng_saved_state: dict[str, Any] | None = None
        self._target = target
        self._ignore_defender = ignore_defender

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        # Load the saved rng state.
        self._load_saved_state()
        walls = observation[0][WALL_IDX]
        if self._grid is None:
            self._grid = Grid(matrix=1 - walls)

        target_pos: tuple[int, int] = get_target_position(observation, self._target)
        defender_pos: tuple[int, int] = get_target_position(observation, DEFENDER)

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
            if self._ignore_defender:
                distance = len(find_path(self._grid, candidate_position, target_pos)[1])
            else:
                # Temporarily consider the defender to be an obstacle for pathfinding.
                # And also raise the weights of defender-reachable tiles.
                defender_node = self._grid.node(x=defender_pos[1], y=defender_pos[0])  # type: ignore
                for node in self._grid.neighbors(defender_node):
                    node.weight = 100
                defender_node.walkable = False
                distance = len(find_path(self._grid, candidate_position, target_pos)[1])
                for node in self._grid.neighbors(defender_node):
                    node.weight = 1
                defender_node.walkable = True
            if distance < min_distance:
                min_candidates = [candidate_action]
                min_distance = distance
            elif distance == min_distance:
                min_candidates.append(candidate_action)

        action = self._rng.choice(min_candidates)
        # `reset_peek()` to ensure the next peek is correct.
        # `PursueAttacker` doesn't need this, as `get_action()` and `peek_action()` are sharing the same state and thus doesn't need syncing.
        # But I'm doing this to keep things consistent.
        self.reset_peek()
        return action

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        if self._rng_saved_state is None:
            # If this is the first peek after a get, save the state to restore it later.
            self._rng_saved_state = self._rng.bit_generator.state  # type: ignore
        return self.get_action(observation=observation)

    def _load_saved_state(self):
        if self._rng_saved_state is None:
            return
        self._rng.bit_generator.state = self._rng_saved_state
        self._rng_saved_state = None

    def reset_peek(self):
        self._load_saved_state()
        return


class RandomPursueAttacker(AttackerAgentBase):
    """Chooses between multiple paths to a target."""

    def __init__(
        self,
        seed: int | None = None,
        target: np.int8 = EXIT,
        ignore_defender: bool = True,
        num_obstacles: int = 2,
        max_obstacle_weight: float = 10,
    ):
        self._grid: Grid | None = None
        # `self._rng` is the only state that should be kept track for determinism.
        # Since `choice()`-ing from the rng will update the rng in both get and peek,
        # we are effectively sharing the same state in get and peek.
        # Thus we store the inner state of the rng for the get-timeline, and restore it before `get_action()`.
        self._rng = np.random.default_rng(seed)
        self._rng_saved_state: dict[str, Any] | None = None
        self._target = target
        self._selected_path: tuple[list[GridNode], list[int]] | None = None
        self._saved_selected_path: tuple[list[GridNode], list[int]] | None = None
        self._ignore_defender = ignore_defender
        self._num_obstacles = num_obstacles
        self._obstacle_weight = max_obstacle_weight / num_obstacles

    def _select_new_path(
        self, observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]]
    ):
        assert self._grid is not None
        target_pos: tuple[int, int] = get_target_position(observation, self._target)
        defender_pos: tuple[int, int] = get_target_position(observation, DEFENDER)
        attacker_pos = observation[1]

        # First fetch the optimal route.
        if self._ignore_defender:
            self._selected_path = find_path(self._grid, attacker_pos, target_pos)
        else:
            # Temporarily consider the defender to be an obstacle for pathfinding.
            # And also raise the weights of defender-reachable tiles.
            defender_node = self._grid.node(x=defender_pos[1], y=defender_pos[0])  # type: ignore
            for node in self._grid.neighbors(defender_node):
                node.weight = 100
            defender_node.walkable = False
            self._selected_path = find_path(self._grid, attacker_pos, target_pos)
            for node in self._grid.neighbors(defender_node):
                node.weight = 1
            defender_node.walkable = True
            if len(self._selected_path[0]) == 0:
                # This means the defender is blocking all possible paths between the attacker and the target.
                # In such case, we have no other choice than to ignore the defender.
                self._selected_path = find_path(self._grid, attacker_pos, target_pos)

        # Then randomly raise the cost of some tiles in the path.
        nodes = self._selected_path[0]
        assert len(nodes) > 0

        obstacle_nodes: list[GridNode] = self._rng.choice(nodes, self._num_obstacles)  # type: ignore
        for obstacle_node in obstacle_nodes:  # type: ignore
            obstacle_node.weight = self._obstacle_weight
            print(f"{obstacle_node.y}, {obstacle_node.x}")

        # Then fetch the route again.
        if self._ignore_defender:
            self._selected_path = find_path(self._grid, attacker_pos, target_pos)
        else:
            # Temporarily consider the defender to be an obstacle for pathfinding.
            # And also raise the weights of defender-reachable tiles.
            defender_node = self._grid.node(x=defender_pos[1], y=defender_pos[0])  # type: ignore
            for node in self._grid.neighbors(defender_node):
                node.weight = 100
            defender_node.walkable = False
            self._selected_path = find_path(self._grid, attacker_pos, target_pos)
            for node in self._grid.neighbors(defender_node):
                node.weight = 1
            defender_node.walkable = True
            if len(self._selected_path[0]) == 0:
                # This means the defender is blocking all possible paths between the attacker and the target.
                # In such case, we have no other choice than to ignore the defender.
                self._selected_path = find_path(self._grid, attacker_pos, target_pos)

        for obstacle_node in obstacle_nodes:  # type: ignore
            obstacle_node.weight = 1

        return

    def _ensure_path(
        self, observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]]
    ):
        if self._selected_path is None:
            self._select_new_path(observation=observation)
            return
        head_node = self._selected_path[0][0]
        attacker_pos = observation[1]
        if attacker_pos != (head_node.y, head_node.x):
            self._select_new_path(observation=observation)
            return
        defender_pos: tuple[int, int] = get_target_position(observation, DEFENDER)
        for node in self._selected_path[0]:
            if defender_pos == (node.y, node.x):
                self._select_new_path(observation=observation)
                return
        return

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        # Load the saved rng state.
        self._load_saved_state()
        walls = observation[0][WALL_IDX]
        if self._grid is None:
            self._grid = Grid(matrix=1 - walls)

        self._ensure_path(observation=observation)
        assert self._selected_path is not None
        path: list[GridNode] = self._selected_path[0]
        actions: list[int] = self._selected_path[1]
        assert path[0].x == observation[1][1] and path[0].y == observation[1][0]
        action = actions[0]
        self._selected_path = (path[1:], actions[1:])
        self.reset_peek()
        return action

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        if self._rng_saved_state is None:
            # If this is the first peek after a get, save the state to restore it later.
            self._rng_saved_state = self._rng.bit_generator.state  # type: ignore
            self._saved_selected_path = self._selected_path
        return self.get_action(observation=observation)

    def _load_saved_state(self):
        if self._rng_saved_state is not None:
            self._rng.bit_generator.state = self._rng_saved_state
            self._rng_saved_state = None
        if self._saved_selected_path is not None:
            self._selected_path = self._saved_selected_path
            self._saved_selected_path = None

    def reset_peek(self):
        self._load_saved_state()
        return


class EvadeAttacker(AttackerAgentBase):
    """Runs away from the target."""

    def __init__(self, seed: int | None = None, target: np.int8 = DEFENDER):
        self._grid: Grid | None = None
        self._rng = np.random.default_rng(seed)
        self._rng_saved_state: dict[str, Any] | None = None
        self._target = target

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        self.reset_peek()
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
            distance = len(find_path(self._grid, candidate_position, target_pos)[1])
            if distance > max_distance:
                max_candidates = [candidate_action]
                max_distance = distance
            elif distance == max_distance:
                max_candidates.append(candidate_action)

        return self._rng.choice(max_candidates)

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        if self._rng_saved_state is None:
            # If this is the first peek after a get, save the state to restore it later.
            self._rng_saved_state = self._rng.bit_generator.state  # type: ignore
        return self.get_action(observation=observation)

    def reset_peek(self):
        if self._rng_saved_state is None:
            return
        self._rng.bit_generator.state = self._rng_saved_state
        self._rng_saved_state = None
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
    ) -> int:
        # If `self._true` and `self._false` have a mutual action, prefer that.
        # Reset peek to ensure we are getting the immediate next action peek.
        # self.reset_peek()
        # true_candidates = self._true.peek_action(observation=observation)
        # false_candidates = self._false.peek_action(observation=observation)
        # # Reset peek again to ensure the next call to `peek_action()` will return the immediate next action peek.
        # self.reset_peek()
        # mutual_candidates: list[int] = []
        # for candidate in true_candidates:
        #     if candidate in false_candidates:
        #         mutual_candidates.append(candidate)
        # if len(mutual_candidates) > 0:
        #     if self._condition(observation, False):
        #         _ = self._true.get_action(observation=observation)
        #     else:
        #         _ = self._false.get_action(observation=observation)
        #     return mutual_candidates
        if self._condition(observation, False):
            action = self._true.get_action(observation=observation)
        else:
            action = self._false.get_action(observation=observation)
        self.reset_peek()
        return action

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        # true_candidates = self._true.peek_action(observation=observation)
        # false_candidates = self._false.peek_action(observation=observation)

        # if self._condition(observation, True):
        #     self._false.reset_peek()
        # else:
        #     self._true.reset_peek()

        # mutual_candidates: list[int] = []
        # for candidate in true_candidates:
        #     if candidate in false_candidates:
        #         mutual_candidates.append(candidate)

        # if len(mutual_candidates) > 0:
        #     return mutual_candidates
        # if self._condition(observation, True):
        #     return true_candidates
        # return false_candidates
        if self._condition(observation, True):
            return self._true.peek_action(observation=observation)
        return self._false.peek_action(observation=observation)

    def reset_peek(self):
        self._true.reset_peek()
        self._false.reset_peek()


class DistanceSwitchAttacker(SwitchAttacker):
    def __init__(
        self,
        trigger_distance: int,
        target: np.int8,
        greater_or_eq: AttackerAgentBase,
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

        super().__init__(
            condition=_distance_condition, true=greater_or_eq, false=lesser
        )


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
    ) -> int:
        actions = super().get_action(observation)
        self._counter += 1
        self.reset_peek()
        return actions

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        action = super().peek_action(observation)
        self._peek_counter += 1
        return action

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
    ) -> int:
        if self._counter % self._stupidity == 0:
            action = self._inner.get_action(observation=observation)
        else:
            action = STAY
        self._counter += 1
        self.reset_peek()
        return action

    def peek_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        if self._peek_counter % self._stupidity == 0:
            action = self._inner.peek_action(observation=observation)
        else:
            action = STAY
        self._peek_counter += 1
        return action

    def reset_peek(self):
        self._peek_counter = self._counter
        self._inner.reset_peek()


class NaiveAttacker(DistanceSwitchAttacker):
    """
    Moves towards the target, unless the defender is closer than `safety_distance`.
    """

    def __init__(
        self,
        seed: int | None = None,
        min_safe_distance: int = 5,
        stupidity: int = 2,
        target: np.int8 = EXIT,
        ignore_defender: bool = False,
    ):
        super().__init__(
            trigger_distance=min_safe_distance,
            target=DEFENDER,
            greater_or_eq=StupidAttacker(
                PursueAttacker(
                    seed=seed, target=target, ignore_defender=ignore_defender
                ),
                stupidity=stupidity,
            ),
            lesser=EvadeAttacker(seed=seed, target=DEFENDER),
        )


class DecisiveNaiveAttacker(DistanceSwitchAttacker):
    """
    If `target` is closer than `commit_distance`, ignore the defender and moves to `target`.
    Otherwise moves towards the `target`, unless the defender is closer than `safety_distance`.
    """

    def __init__(
        self,
        seed: int | None = None,
        min_safe_distance: int = 5,
        max_commit_distance: int = 3,
        stupidity: int = 2,
        target: np.int8 = EXIT,
        ignore_defender: bool = False,
    ):
        super().__init__(
            trigger_distance=max_commit_distance,
            target=target,
            greater_or_eq=NaiveAttacker(
                seed=seed,
                min_safe_distance=min_safe_distance,
                stupidity=stupidity,
                target=target,
                ignore_defender=ignore_defender,
            ),
            lesser=StupidAttacker(
                PursueAttacker(seed=seed, target=target, ignore_defender=False),
                stupidity=stupidity,
            ),
        )


class DeceptiveAttacker(TimeSwitchAttacker):
    """
    Targets the decoy for first before `stop_deception_after`, then targets the exit.
    """

    def __init__(
        self,
        seed: int | None = None,
        min_safe_distance: int = 5,
        max_commit_distance: int = 3,
        stupidity: int = 2,
        ignore_defender: bool = False,
        stop_deception_after: int = 32,
    ):
        super().__init__(
            true_after=stop_deception_after,
            true=DecisiveNaiveAttacker(
                seed=seed,
                min_safe_distance=min_safe_distance,
                max_commit_distance=max_commit_distance,
                stupidity=stupidity,
                target=EXIT,
                ignore_defender=ignore_defender,
            ),
            # Don't be decisive at a decoy.
            false=NaiveAttacker(
                seed=seed,
                min_safe_distance=min_safe_distance,
                stupidity=stupidity,
                target=DECOY,
                ignore_defender=ignore_defender,
            ),
        )
