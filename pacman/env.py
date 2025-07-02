from pettingzoo import ParallelEnv  # type: ignore
from gymnasium.spaces import MultiBinary, Discrete
import numpy as np
from typing import Any
from colorama import Fore, Back, Style
from .core import PacmanCore
from .core import HEIGHT, WIDTH
from .core import WALL, PLAYER, GHOST, DOT, POWER


class PacmanEnv(ParallelEnv[str, np.ndarray[Any, np.dtype[np.int8]], int]):
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
        self._player: str = "player"
        self._ghosts: list[str] = []
        for i in range(ghost_count):
            self._ghosts.append(f"ghost_{i}")
        self._core = PacmanCore(player=self._player, ghosts=self._ghosts)
        self._last_score = 0

    def _valid_agent(self, agent: str) -> bool:
        if agent == self._player or agent in self._ghosts:
            return True
        return False

    def observation_space(self, agent: str):
        if not self._valid_agent(agent):
            raise ValueError("`agent` is not recognized.")
        return MultiBinary((HEIGHT, WIDTH, 5))

    def action_space(self, agent: str):
        if not self._valid_agent(agent):
            raise ValueError("`agent` is not recognized.")
        return Discrete(5)

    def _get_observation(self) -> dict[str, np.ndarray[Any, np.dtype[np.int8]]]:
        map = self._core.map
        full_observation = np.stack(
            [
                (map & WALL) != 0,
                (map & PLAYER) != 0,
                (map & GHOST) != 0,
                (map & DOT) != 0,
                (map & POWER) != 0,
            ],
            axis=-1,
        )

        observation: dict[str, np.ndarray[Any, np.dtype[np.int8]]] = {}
        observation[self._player] = full_observation
        for ghost in self._ghosts:
            observation[ghost] = full_observation
        return observation

    def reset(
        self, seed: int | None = None, options: dict[Any, Any] | None = None
    ) -> tuple[dict[str, np.ndarray[Any, np.dtype[np.int8]]], dict[Any, Any]]:
        self._core.reset()
        self._last_score = 0
        return self._get_observation(), {}

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray[Any, np.dtype[np.int8]]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        if self._player in actions:
            action = actions[self._player]
            self._core.perform_action(self._player, action)
        for ghost in self._ghosts:
            if ghost in actions:
                action = actions[ghost]
                self._core.perform_action(ghost, action)

        score_delta = self._core.score - self._last_score
        self._last_score = self._core.score

        rewards: dict[str, float] = {ghost: -score_delta for ghost in self._ghosts}
        rewards[self._player] = score_delta

        t: dict[str, bool] = {
            ghost: (self._core.ghosts[ghost] is None) or (self._core.terminated)
            for ghost in self._ghosts
        }
        t[self._player] = (
            self._core.player[self._player] is None
        ) or self._core.terminated

        return self._get_observation(), rewards, t, t, {}

    def render(self):
        assert self._render_mode == "ansi"
        for h in range(HEIGHT):
            for w in range(WIDTH):
                tile: np.int8 = self._core.map[h, w]
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
