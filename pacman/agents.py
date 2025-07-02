import numpy as np
from typing import Any
from abc import ABC, abstractmethod
from pathfinding.core.grid import Grid, GridNode  # type: ignore
from pathfinding.finder.a_star import AStarFinder  # type: ignore
from . import STAY, UP, DOWN, LEFT, RIGHT


class GhostAgentBase(ABC):
    @abstractmethod
    def get_action(
        self, observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]]
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


class PursueGhost(GhostAgentBase):
    def get_action(
        self, observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]]
    ) -> int:
        walls = observation[0][0]
        player = observation[0][1]
        # Follow the first player to be found.
        # The environment contains only one player anyways, so no need to worry.
        player_pos: tuple[int, int] = np.argwhere(player)[0]
        my_pos = observation[1]
        actions = GhostAgentBase.find_path(walls, my_pos, player_pos)
        assert (
            len(actions) > 0
        )  # Since we always have a single player in the game, and we must haven't catch it yet, we can always find a path to it.
        return actions[0]
