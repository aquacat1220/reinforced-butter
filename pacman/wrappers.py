from gymnasium import Env
import numpy as np
from typing import Any, Callable
from gymnasium.spaces import MultiBinary, Tuple, Discrete
from .env import PacmanEnv
from .agents import GhostAgentBase
from .core import HEIGHT, WIDTH, POWER_DURATION


class GymWrapper(
    Env[tuple[np.ndarray[Any, np.dtype[np.uint8]], tuple[int, int], int], np.int64]
):
    def __init__(self, env: PacmanEnv, ghost_builder: Callable[[str], GhostAgentBase]):
        self.env = env
        self.observation_space = Tuple(
            (
                MultiBinary((HEIGHT, WIDTH, 5)),
                Tuple((Discrete(HEIGHT), Discrete(WIDTH))),
                Discrete(POWER_DURATION + 1),
            )
        )
        self.action_space = Discrete(5)
        self._last_observations: (
            dict[str, tuple[np.ndarray[Any, np.dtype[np.uint8]], tuple[int, int], int]]
            | None
        ) = None
        self._ghost_builder = ghost_builder

    def step(self, action: np.int64) -> tuple[
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
        actions[self.env.player] = int(action)
        for ghost_name, ghost in self._ghosts.items():
            if self._ghost_dones[ghost_name]:
                # If `ghost` is already done, skip fetching actions.
                continue
            actions[ghost_name] = ghost.get_action(self._last_observations[ghost_name])
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        for ghost_name in self._ghost_dones:
            if self._ghost_dones[ghost_name]:
                continue
            self._ghost_dones[ghost_name] = (
                terminations[ghost_name] or truncations[ghost_name]
            )
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

        self._ghosts: dict[str, GhostAgentBase] = {}
        self._ghost_dones: dict[str, bool] = {}
        for ghost_name in self.env.ghosts:
            ghost = self._ghost_builder(ghost_name)
            self._ghosts[ghost_name] = ghost
            self._ghost_dones[ghost_name] = False

        return (
            observations[self.env.player],
            infos[self.env.player],
        )

    def render(self):
        self.env.render()
