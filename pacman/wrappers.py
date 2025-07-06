from gymnasium import Env
import numpy as np
from typing import Any, Callable
from .env import PacmanEnv
from .agents import GhostAgentBase


class GymWrapper(
    Env[tuple[np.ndarray[Any, np.dtype[np.uint8]], tuple[int, int], int], int]
):
    def __init__(self, env: PacmanEnv, ghost_builder: Callable[[str], GhostAgentBase]):
        self.env = env
        self._last_observations: (
            dict[str, tuple[np.ndarray[Any, np.dtype[np.uint8]], tuple[int, int], int]]
            | None
        ) = None
        self._ghosts: dict[str, GhostAgentBase] = {}
        for ghost_name in self.env.ghosts:
            ghost = ghost_builder(ghost_name)
            self._ghosts[ghost_name] = ghost

    def step(self, action: int) -> tuple[
        tuple[np.ndarray[Any, np.dtype[np.uint8]], tuple[int, int], int],
        float,
        bool,
        bool,
        dict[Any, Any],
    ]:
        if self._last_observations is None:
            raise Exception(
                "`GymWrapper.step()` was called before `GymWrapper.reset()` was ever called."
            )
        actions: dict[str, int] = {}
        actions[self.env.player] = action
        for ghost_name, ghost in self._ghosts.items():
            actions[ghost_name] = ghost.get_action(self._last_observations[ghost_name])
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self._last_observations = observations
        return (
            observations[self.env.player],
            rewards[self.env.player],
            terminations[self.env.player],
            truncations[self.env.player],
            infos[self.env.player],
        )

    def reset(
        self, seed: int | None = None, options: dict[Any, Any] | None = None
    ) -> tuple[
        tuple[np.ndarray[Any, np.dtype[np.uint8]], tuple[int, int], int], dict[Any, Any]
    ]:
        observations, infos = self.env.reset()
        self._last_observations = observations
        return (
            observations[self.env.player],
            infos[self.env.player],
        )
