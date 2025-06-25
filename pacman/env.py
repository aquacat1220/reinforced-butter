from pettingzoo import ParallelEnv  # type: ignore
from gymnasium.spaces import Box, Discrete
import numpy as np
from typing import Any
from pacman import PacmanCore


class PacmanEnv(ParallelEnv[str, np.ndarray[Any, np.dtype[np.float32]], int]):
    metadata = {"name": "pacman_env_v0", "render_modes": ["human"]}

    def __init__(
        self,
        render_mode: str = "human",
        agent_sight_limit: int = 5,
        ghost_count: int = 3,
    ):
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                '`render_mode` should be one of `PacmanEnv.metadata["render_modes"].'
            )
        self._render_mode = render_mode
        self._agent_sight_limit = agent_sight_limit
        self._agents: list[str] = ["player"]
        for i in range(ghost_count):
            self._agents.append(f"ghost_{i}")
        self._core = PacmanCore(agents=self._agents)

    def observation_space(self, agent: str):
        if agent not in self._agents:
            raise ValueError("`agent` is not recognized.")
        return Box(
            low=0.0,
            high=1.0,
            shape=(self._agent_sight_limit, self._agent_sight_limit, 5),
            dtype=np.float32,
        )

    def action_space(self, agent: str):
        if agent not in self._agents:
            raise ValueError("`agent` is not recognized.")
        return Discrete(5)

    def _get_observation(self) -> dict[str, np.ndarray[Any, np.dtype[np.float32]]]:
        observation: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
        for agent in self._agents:
            observation[agent] = np.zeros(
                (self._agent_sight_limit, self._agent_sight_limit, 5), dtype=np.float32
            )
        return observation

    def reset(
        self, seed: int | None = None, options: dict[Any, Any] | None = None
    ) -> tuple[dict[str, np.ndarray[Any, np.dtype[np.float32]]], dict[Any, Any]]:
        return self._get_observation(), {}

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray[Any, np.dtype[np.float32]]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        rewards: dict[str, float] = {agent: 0.0 for agent in self._agents}
        t: dict[str, bool] = {agent: False for agent in self._agents}
        return self._get_observation(), rewards, t, t, {}

    def render(self):
        pass
