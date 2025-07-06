import numpy as np
from typing import Any
from abc import ABC, abstractmethod
from pathfinding.core.grid import Grid, GridNode  # type: ignore
from pathfinding.finder.a_star import AStarFinder  # type: ignore
from . import STAY, UP, DOWN, LEFT, RIGHT

STUPIDITY = 3


class GhostAgentBase(ABC):
    @abstractmethod
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> int:
        return STAY

    @classmethod
    def find_path(
        cls,
        walls: np.ndarray[Any, np.dtype[np.int8]],
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[int]:
        grid = Grid(matrix=1 - walls)
        start_node = grid.node(x=start[1], y=start[0])  # type: ignore
        end_node = grid.node(x=end[1], y=end[0])  # type: ignore
        finder = AStarFinder()
        path: list[GridNode] = finder.find_path(start_node, end_node, grid)[0]  # type: ignore
        actions: list[int] = []
        curr_node = start_node
        for path_node in path[1:]:  # type: ignore
            if curr_node.x == path_node.x:  # type: ignore
                if curr_node.y - 1 == path_node.y:  # type: ignore
                    action = UP
                elif curr_node.y + 1 == path_node.y:  # type: ignore
                    action = DOWN
                else:
                    raise Exception("Unreachable")
            elif curr_node.y == path_node.y:  # type: ignore
                if curr_node.x - 1 == path_node.x:  # type: ignore
                    action = LEFT
                elif curr_node.x + 1 == path_node.x:  # type: ignore
                    action = RIGHT
                else:
                    raise Exception("Unreachable")
            else:
                raise Exception("Unreachable")
            actions.append(action)
            curr_node = path_node  # type: ignore

        return actions


class IdleGhost(GhostAgentBase):
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> int:
        return STAY


class PursueGhost(GhostAgentBase):
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> int:
        walls = observation[0][0]
        player = observation[0][3]
        # Follow the first player to be found.
        # The environment contains only one player anyways, so no need to worry.
        player_pos: tuple[int, int] = np.argwhere(player)[0]
        my_pos = observation[1]
        actions = GhostAgentBase.find_path(walls, my_pos, player_pos)
        assert (
            len(actions) > 0
        )  # Since we always have a single player in the game, and we must haven't catch it yet, we can always find a path to it.
        return actions[0]


class StupidPursueGhost(PursueGhost):
    def __init__(self):
        self._counter = 0

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> int:
        action: int = STAY
        if self._counter % STUPIDITY == 0:
            action = super().get_action(observation)
        self._counter += 1
        return action


class PatrolPowerGhost(GhostAgentBase):
    def __init__(self):
        self._target_power_index: int = 0

    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> int:
        walls = observation[0][0]
        powers = observation[0][2]
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
        actions = GhostAgentBase.find_path(walls, my_pos, target_power_pos)
        if len(actions) == 0:
            return STAY
        return actions[0]
