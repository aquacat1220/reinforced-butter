from gymnasium import Env, ObservationWrapper
import numpy as np
from typing import Any, Callable
from gymnasium.spaces import MultiBinary, Tuple, Discrete
from .env import PacmanEnv
from .agents import GhostAgentBase
from .core import HEIGHT, WIDTH, POWER_DURATION
from .env import WALL_IDX, DOT_IDX, POWER_IDX, PLAYER_IDX, GHOST_IDX


class GymWrapper(
    Env[tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int], np.int64]
):
    def __init__(self, env: PacmanEnv, ghost_builder: Callable[[str], GhostAgentBase]):
        self.env = env
        self.metadata = env.metadata
        self.render_mode = env.render_mode
        self.observation_space = Tuple(
            (
                MultiBinary((5, HEIGHT, WIDTH)),
                Tuple((Discrete(HEIGHT), Discrete(WIDTH))),
                Discrete(POWER_DURATION + 1),
            )
        )
        self.action_space = Discrete(5)
        self._ghost_builder = ghost_builder

    def step(self, action: np.int64) -> tuple[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
        float,
        bool,
        bool,
        dict[Any, Any],
    ]:
        actions = self._ghost_next_actions
        actions[self.env.player] = int(action)
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        self._ghost_next_actions = {}
        # Effectively *moving* the dict from `self._ghost_next_actions` to `actions`.
        for ghost_name, ghost_done in self._ghost_dones.items():
            if ghost_done:
                continue
            if terminations[ghost_name] or truncations[ghost_name]:
                self._ghost_dones[ghost_name] = True
                continue
            # Fetch actions only from ghosts that are not done yet.
            ghost = self._ghosts[ghost_name]
            self._ghost_next_actions[ghost_name] = ghost.get_action(
                observations[ghost_name]
            )

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
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int], dict[Any, Any]
    ]:
        observations, infos = self.env.reset(seed, options)

        self._ghosts: dict[str, GhostAgentBase] = {}
        self._ghost_dones: dict[str, bool] = {}
        self._ghost_next_actions: dict[str, int] = {}
        for ghost_name in self.env.ghosts:
            ghost = self._ghost_builder(ghost_name)
            self._ghosts[ghost_name] = ghost
            self._ghost_dones[ghost_name] = False
            self._ghost_next_actions[ghost_name] = ghost.get_action(
                observations[ghost_name]
            )

        return (
            observations[self.env.player],
            infos[self.env.player],
        )

    def render(self) -> str | np.ndarray[Any, np.dtype[np.uint8]]:
        return self.env.render()


class StripWrapper(
    ObservationWrapper[
        np.ndarray[Any, np.dtype[np.int8]],
        np.int64,
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ]
):
    def __init__(
        self,
        env: Env[
            tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int], np.int64
        ],
    ):
        super().__init__(env)
        self.observation_space = MultiBinary((5, HEIGHT, WIDTH))

    def observation(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> np.ndarray[Any, np.dtype[np.int8]]:
        return observation[0]


class PartialObservabilityWrapper(
    ObservationWrapper[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
        np.int64,
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ]
):
    def __init__(
        self,
        env: Env[
            tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int], np.int64
        ],
        sight_limit: int = 7,
    ):
        super().__init__(env)
        self._sight_limit = sight_limit
        self._last_observation: (
            tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int] | None
        ) = None
        self._last_mask: np.ndarray[Any, np.dtype[np.bool]] | None = None

    def observation(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ) -> tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int]:
        self._last_observation = observation
        dots = observation[0][DOT_IDX]
        powers = observation[0][POWER_IDX]
        player_pos = np.array(observation[1])
        indices = np.indices((HEIGHT, WIDTH)).transpose((1, 2, 0))
        signed_dist = indices - player_pos
        unsigned_dist = np.abs(signed_dist)
        dist = np.sum(unsigned_dist, axis=-1)
        mask = dist <= self._sight_limit
        self._last_mask = mask
        masked_dots = np.where(mask, dots, -1)
        masked_powers = np.where(mask, powers, -1)
        observation[0][DOT_IDX] = masked_dots
        observation[0][POWER_IDX] = masked_powers
        return observation

    def render(self):
        if self.render_mode == "ansi":
            raise NotImplementedError(
                "I am too lazy to implement partial-observable text rendering."
            )
        elif self.render_mode == "rgb_array":
            image: np.ndarray[Any, np.dtype[np.uint8]] = self.env.render()  # type: ignore

            assert image.shape == (HEIGHT * 3, WIDTH * 3, 3)
            assert (self._last_observation is not None) and (
                self._last_mask is not None
            )  # This can happen if `reset()` wasn't called before calling `render()`.
            # Tiles within sight range, or holding walls/players/ghosts can be seen.
            # Strictly speaking, agent can never see dot/power status of tiles out of range.
            # But since players and ghosts render *over* dots and powers, we don't need to mask tiles with players or ghosts.
            mask = np.logical_or(
                np.logical_or(self._last_mask, self._last_observation[0][WALL_IDX]),
                np.logical_or(
                    self._last_observation[0][PLAYER_IDX],
                    self._last_observation[0][GHOST_IDX],
                ),
            )
            mask = mask[:, :, np.newaxis, np.newaxis, np.newaxis]
            tiled_image = image.reshape((HEIGHT, 3, WIDTH, 3, 3)).transpose(
                (0, 2, 1, 3, 4)
            )
            masked_tiled_image = np.where(
                mask, tiled_image, np.array([0, 0, 100], dtype=np.uint8)
            )
            masked_image = masked_tiled_image.transpose((0, 2, 1, 3, 4)).reshape(
                (HEIGHT * 3, WIDTH * 3, 3)
            )
            return masked_image
