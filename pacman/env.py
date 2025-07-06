from pettingzoo import ParallelEnv  # type: ignore
from gymnasium.spaces import MultiBinary, Discrete, Tuple
import numpy as np
from typing import Any
from colorama import Fore
from .core import PacmanCore
from .core import HEIGHT, WIDTH
from .core import POWER_DURATION
from .core import WALL, DOT, POWER, PLAYER, GHOST, ONE_GHOST


class PacmanEnv(
    ParallelEnv[
        str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int], int
    ]
):
    metadata = {"name": "pacman_env_v0", "render_modes": ["ansi", "rgb_array"]}

    def __init__(
        self,
        render_mode: str = "ansi",
        agent_sight_limit: int = 5,
        ghost_count: int = 2,
    ):
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                '`render_mode` should be one of `PacmanEnv.metadata["render_modes"].'
            )
        self.render_mode = render_mode
        self._player_sight_limit = agent_sight_limit
        self.player: str = "player"
        self.ghosts: list[str] = []
        for i in range(ghost_count):
            self.ghosts.append(f"ghost_{i}")
        self._core = PacmanCore(player=self.player, ghosts=self.ghosts)
        self._last_score = 0

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
        full_observation = np.stack(
            [
                (map & WALL) != 0,
                (map & DOT) != 0,
                (map & POWER) != 0,
                (map & PLAYER) != 0,
                # The ghost map contains the number of ghosts on that tile.
                (map & GHOST),
            ],
            axis=0,
        )

        observation: dict[
            str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int]
        ] = {}
        observation[self.player] = (full_observation, self._core.player[self.player], self._core.player_power_remaining)  # type: ignore
        for ghost in self.ghosts:
            observation[ghost] = (full_observation, self._core.ghosts[ghost], self._core.player_power_remaining)  # type: ignore
        return observation

    def _get_empty_infos(self) -> dict[str, dict[Any, Any]]:
        empty_infos: dict[str, dict[Any, Any]] = {}
        empty_infos[self.player] = {}
        for ghost in self.ghosts:
            empty_infos[ghost] = {}
        return empty_infos

    def reset(
        self, seed: int | None = None, options: dict[Any, Any] | None = None
    ) -> tuple[
        dict[str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int]],
        dict[str, dict[Any, Any]],
    ]:
        self._core.reset(seed)
        self._last_score = 0
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

        score_delta = self._core.score - self._last_score
        self._last_score = self._core.score

        rewards: dict[str, float] = {ghost: -score_delta for ghost in self.ghosts}
        rewards[self.player] = score_delta

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
            tiles[0] * WALL
            + tiles[1] * DOT
            + tiles[2] * POWER
            + tiles[3] * PLAYER
            + tiles[4] * ONE_GHOST
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
            tiles[0] * WALL
            + tiles[1] * DOT
            + tiles[2] * POWER
            + tiles[3] * PLAYER
            + tiles[4] * ONE_GHOST
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
