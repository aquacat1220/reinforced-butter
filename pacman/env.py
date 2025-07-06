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
    metadata = {"name": "pacman_env_v0", "render_modes": ["ansi"]}

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
        self._render_mode = render_mode
        self._player_sight_limit = agent_sight_limit
        self.player: str = "player"
        self.ghosts: list[str] = []
        for i in range(ghost_count):
            self.ghosts.append(f"ghost_{i}")
        self._core = PacmanCore(player=self.player, ghosts=self.ghosts)
        self._last_score = 0
        self._infos: dict[str, dict[Any, Any]] | None = None

    def _valid_agent(self, agent: str) -> bool:
        if agent == self.player or agent in self.ghosts:
            return True
        return False

    def observation_space(self, agent: str):
        if not self._valid_agent(agent):
            raise ValueError("`agent` is not recognized.")
        return Tuple(
            (
                MultiBinary((HEIGHT, WIDTH, 5)),
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
        if self._infos is not None:
            return self._infos
        self._infos = {}
        self._infos[self.player] = {}
        for ghost in self.ghosts:
            self._infos[ghost] = {}
        return self._infos

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

    def render(self):
        assert self._render_mode == "ansi"
        PacmanEnv.render_observation(self._get_observation()[self.player])

    @classmethod
    def render_observation(
        cls,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ):
        tiles = observation[0]
        player_power_remaining = observation[2]
        reconstructed_map = (
            tiles[0] * WALL
            + tiles[1] * DOT
            + tiles[2] * POWER
            + tiles[3] * PLAYER
            + tiles[4] * ONE_GHOST
        )
        for h in range(HEIGHT):
            for w in range(WIDTH):
                tile: np.int8 = reconstructed_map[h, w]
                if tile & WALL:
                    print(Fore.WHITE + "██", end="")
                elif tile & PLAYER:
                    if player_power_remaining > 0:
                        print(Fore.LIGHTRED_EX + "🭪 ", end="")
                    else:
                        print(Fore.YELLOW + "🭪 ", end="")
                elif tile & GHOST:
                    print(Fore.BLUE + "ᙁ ", end="")
                elif tile & DOT:
                    print(Fore.YELLOW + "○ ", end="")
                elif tile & POWER:
                    print(Fore.YELLOW + "● ", end="")
                else:
                    print("  ", end="")
            print()
