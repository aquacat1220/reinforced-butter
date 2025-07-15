from pettingzoo import ParallelEnv  # type: ignore
from gymnasium.spaces import MultiBinary, Discrete, Tuple
import numpy as np
from typing import Any
from colorama import Fore
from pathfinding.core.grid import Grid, GridNode  # type: ignore
from pathfinding.finder.a_star import AStarFinder  # type: ignore
from .core import STAY, UP, DOWN, LEFT, RIGHT
from .core import PacmanCore, Event
from .core import HEIGHT, WIDTH
from .core import POWER_DURATION
from .core import WALL, DOT, POWER, PLAYER, GHOST, ONE_GHOST

WALL_IDX = 0
DOT_IDX = 1
POWER_IDX = 2
PLAYER_IDX = 3
GHOST_IDX = 4

DOT_SCORE = 1
POWER_SCORE = 5
GHOST_KILL_SCORE = 20
WIN_SCORE = 50
LOSE_SCORE = -50


def find_path(
    grid: Grid,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[int]:
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


class PacmanEnv(
    ParallelEnv[
        str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int], int
    ]
):
    metadata = {"name": "pacman_env_v0", "render_modes": ["ansi", "rgb_array"]}

    def __init__(
        self,
        render_mode: str = "ansi",
        ghost_count: int = 2,
        use_distance_reward: bool = False,
    ):
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                '`render_mode` should be one of `PacmanEnv.metadata["render_modes"].'
            )
        self.render_mode = render_mode
        self._use_distance_reward = use_distance_reward
        self.player: str = "player"
        self.ghosts: list[str] = []
        for i in range(ghost_count):
            self.ghosts.append(f"ghost_{i}")
        self._core = PacmanCore(player=self.player, ghosts=self.ghosts)

    def _valid_agent(self, agent: str) -> bool:
        if agent == self.player or agent in self.ghosts:
            return True
        return False

    def observation_space(self, agent: str):
        if not self._valid_agent(agent):
            raise ValueError("`agent` is not recognized.")
        return Tuple(
            (
                MultiBinary((5, HEIGHT, WIDTH)),
                Tuple((Discrete(HEIGHT), Discrete(WIDTH))),
                Discrete(POWER_DURATION + 1),
            )
        )

    def action_space(self, agent: str):
        if not self._valid_agent(agent):
            raise ValueError("`agent` is not recognized.")
        return Discrete(5)

    def _get_observation(
        self,
    ) -> dict[str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int]]:
        map = self._core.map
        stack: list[np.ndarray[Any, np.dtype[np.int8]] | None] = [
            None,
            None,
            None,
            None,
            None,
        ]
        stack[WALL_IDX] = (map & WALL) != 0
        stack[DOT_IDX] = (map & DOT) != 0
        stack[POWER_IDX] = (map & POWER) != 0
        stack[PLAYER_IDX] = (map & PLAYER) != 0
        stack[GHOST_IDX] = map & GHOST
        stack_: list[np.ndarray[Any, np.dtype[np.int8]]] = stack  # type: ignore
        full_observation = np.stack(
            stack_,
            axis=0,
        )

        observation: dict[
            str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int]
        ] = {}
        player_pos: tuple[int, int] | None = self._core.player[self.player]
        if player_pos is None:
            player_pos = (-1, -1)
        observation[self.player] = (
            full_observation,
            player_pos,
            self._core.player_power_remaining,
        )
        for ghost in self.ghosts:
            ghost_pos: tuple[int, int] | None = self._core.ghosts[ghost]
            if ghost_pos is None:
                ghost_pos = (-1, -1)
            observation[ghost] = (
                full_observation,
                ghost_pos,
                self._core.player_power_remaining,
            )
        return observation

    def _get_empty_infos(self) -> dict[str, dict[Any, Any]]:
        empty_infos: dict[str, dict[Any, Any]] = {}
        empty_infos[self.player] = {}
        for ghost in self.ghosts:
            empty_infos[ghost] = {}
        return empty_infos

    def _compute_score(self, events: list[Event]) -> float:
        score: float = 0
        for event in events:
            if event == Event.CONSUME_DOT:
                score += DOT_SCORE
            elif event == Event.CONSUME_POWER:
                score += POWER_SCORE
            elif event == Event.KILL_GHOST:
                score += GHOST_KILL_SCORE
            elif event == Event.LOSE:
                score += LOSE_SCORE
            elif event == Event.WIN:
                score += WIN_SCORE
            else:
                raise Exception("Unreachable")
        return score

    def _compute_min_distance(self) -> int:
        min_distance: int | None = None
        player_pos = self._core.player[self.player]
        if player_pos is not None:
            # Then compute min distance between ghost and player.
            for ghost_name in self.ghosts:
                ghost_pos = self._core.ghosts[ghost_name]
                if ghost_pos is None:
                    continue
                distance = len(find_path(self._grid, ghost_pos, player_pos))
                if min_distance is None:
                    min_distance = distance
                else:
                    min_distance = min(distance, min_distance)
        if min_distance is None:
            return 0
        return min_distance

    def reset(
        self, seed: int | None = None, options: dict[Any, Any] | None = None
    ) -> tuple[
        dict[str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int]],
        dict[str, dict[Any, Any]],
    ]:
        self._core.reset(seed)
        walls = (self._core.map & WALL) != 0
        self._grid: Grid = Grid(matrix=1 - walls)

        self._last_min_distance: int = self._compute_min_distance()
        return self._get_observation(), self._get_empty_infos()

    def step(self, actions: dict[str, int]) -> tuple[
        dict[str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        if self.player in actions:
            action = actions[self.player]
            self._core.perform_action(self.player, action)
        for ghost in self.ghosts:
            if ghost in actions:
                action = actions[ghost]
                self._core.perform_action(ghost, action)

        score = self._compute_score(self._core.events)
        # Consume the event queue to compute score.
        self._core.events.clear()

        reward: float = score

        if self._use_distance_reward:
            min_distance = self._compute_min_distance()
            delta_min_distance = min_distance - self._last_min_distance
            self._last_min_distance = min_distance

            if self._core.player_power_remaining > 0:
                # If player has power, the closer we are to the ghosts the better it is.
                delta_min_distance = -delta_min_distance

            reward += delta_min_distance

        rewards: dict[str, float] = {ghost: -reward for ghost in self.ghosts}
        rewards[self.player] = reward

        t: dict[str, bool] = {
            ghost: (self._core.ghosts[ghost] is None) or (self._core.terminated)
            for ghost in self.ghosts
        }
        t[self.player] = (
            self._core.player[self.player] is None
        ) or self._core.terminated

        return self._get_observation(), rewards, t, t, self._get_empty_infos()

    def render(self) -> str | np.ndarray[Any, np.dtype[np.uint8]]:
        if self.render_mode == "ansi":
            return PacmanEnv.render_observation_ansi(
                self._get_observation()[self.player]
            )
        elif self.render_mode == "rgb_array":
            return PacmanEnv.render_observation_rgb(
                self._get_observation()[self.player]
            )
        else:
            raise Exception("Unreachable")

    @classmethod
    def render_observation_ansi(
        cls,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> str:
        tiles = observation[0]
        player_power_remaining = observation[2]
        reconstructed_map = (
            tiles[WALL_IDX] * WALL
            + tiles[DOT_IDX] * DOT
            + tiles[POWER_IDX] * POWER
            + tiles[PLAYER_IDX] * PLAYER
            + tiles[GHOST_IDX] * ONE_GHOST
        )
        text: str = ""
        for h in range(HEIGHT):
            for w in range(WIDTH):
                tile: np.int8 = reconstructed_map[h, w]
                if tile & WALL:
                    text += Fore.WHITE + "██"
                elif tile & PLAYER:
                    if player_power_remaining > 0:
                        text += Fore.LIGHTRED_EX + "🭪 "
                    else:
                        text += Fore.YELLOW + "🭪 "
                elif tile & GHOST:
                    text += Fore.BLUE + "ᙁ "
                elif tile & DOT:
                    text += Fore.YELLOW + "○ "
                elif tile & POWER:
                    text += Fore.YELLOW + "● "
                else:
                    text += "  "
            text += "\n"
        return text

    @classmethod
    def render_observation_rgb(
        cls,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> np.ndarray[Any, np.dtype[np.uint8]]:
        tiles = observation[0]
        player_power_remaining = observation[2]
        reconstructed_map = (
            tiles[WALL_IDX] * WALL
            + tiles[DOT_IDX] * DOT
            + tiles[POWER_IDX] * POWER
            + tiles[PLAYER_IDX] * PLAYER
            + tiles[GHOST_IDX] * ONE_GHOST
        )
        image = np.zeros(shape=(HEIGHT, WIDTH, 3, 3, 3), dtype=np.uint8)
        for h in range(HEIGHT):
            for w in range(WIDTH):
                tile: np.int8 = reconstructed_map[h, w]
                image_tile = np.zeros(shape=(3, 3, 3), dtype=np.uint8)
                if tile & WALL:
                    image_tile = image_tile + [255, 255, 255]
                elif tile & PLAYER:
                    if player_power_remaining > 0:
                        color = [255, 100, 100]
                    else:
                        color = [255, 255, 0]
                    image_tile = image_tile + color
                    image_tile[1, 2] = [0, 0, 0]
                elif tile & GHOST:
                    color = [0, 0, 255]
                    image_tile[0, 1] = color
                    image_tile[1, 0] = color
                    image_tile[1, 2] = color
                    image_tile[2, 0] = color
                    image_tile[2, 2] = color
                elif tile & DOT:
                    image_tile[1, 1] = [255, 255, 0]
                elif tile & POWER:
                    color = [255, 255, 0]
                    image_tile[0, 1] = color
                    image_tile[1, 0] = color
                    image_tile[1, 2] = color
                    image_tile[2, 1] = color
                else:
                    pass
                image[h, w] = image_tile
        image = image.transpose((0, 2, 1, 3, 4))
        image = image.reshape((HEIGHT * 3, WIDTH * 3, 3))
        return image
