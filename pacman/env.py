from pettingzoo import ParallelEnv  # type: ignore
from gymnasium.spaces import MultiBinary, Discrete, Tuple
import numpy as np
from typing import Any
from colorama import Fore
from .core import PacmanCore
from .core import HEIGHT, WIDTH
from .core import WALL, PLAYER, GHOST, DOT, POWER


class PacmanEnv(
    ParallelEnv[str, tuple[np.ndarray[Any, np.dtype[np.uint8]], tuple[int, int]], int]
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
            )
        )

    def action_space(self, agent: str):
        if not self._valid_agent(agent):
            raise ValueError("`agent` is not recognized.")
        return Discrete(5)

    def _get_observation(
        self,
    ) -> dict[str, tuple[np.ndarray[Any, np.dtype[np.uint8]], tuple[int, int]]]:
        map = self._core.map
        full_observation = np.stack(
            [
                (map & WALL) != 0,
                (map & PLAYER) != 0,
                (map & GHOST) != 0,
                (map & DOT) != 0,
                (map & POWER) != 0,
            ],
            axis=0,
        )

        observation: dict[
            str, tuple[np.ndarray[Any, np.dtype[np.uint8]], tuple[int, int]]
        ] = {}
        observation[self.player] = (full_observation, self._core.player[self.player])  # type: ignore
        for ghost in self.ghosts:
            observation[ghost] = (full_observation, self._core.ghosts[ghost])  # type: ignore
        return observation

    def reset(
        self, seed: int | None = None, options: dict[Any, Any] | None = None
    ) -> tuple[
        dict[str, tuple[np.ndarray[Any, np.dtype[np.uint8]], tuple[int, int]]],
        dict[Any, Any],
    ]:
        self._core.reset()
        self._last_score = 0
        return self._get_observation(), {}

    def step(self, actions: dict[str, int]) -> tuple[
        dict[str, tuple[np.ndarray[Any, np.dtype[np.uint8]], tuple[int, int]]],
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

        return self._get_observation(), rewards, t, t, {}

    def render(self):
        assert self._render_mode == "ansi"
        for h in range(HEIGHT):
            for w in range(WIDTH):
                tile: np.uint8 = self._core.map[h, w]
                if tile & WALL:
                    print(Fore.WHITE + "██", end="")
                elif tile & PLAYER:
                    if self._core.player_power_remaining > 0:
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

    def is_terminated(self) -> bool:
        return self._core.terminated
