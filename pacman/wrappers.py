from gymnasium import Env, ObservationWrapper, Wrapper
import numpy as np
from typing import Any, Callable
from gymnasium.spaces import MultiBinary, Tuple, Discrete
from .env import PacmanEnv
from .agents import GhostAgentBase
from .core import HEIGHT, WIDTH, POWER_DURATION, STAY, UP, DOWN, LEFT, RIGHT
from .env import WALL_IDX, DOT_IDX, POWER_IDX, PLAYER_IDX, GHOST_IDX


class GymWrapper(
    Env[tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int], np.int64]
):
    """
    Wrapper to convert `PacmanEnv` from a `pettingzoo.ParallelEnv` to a `gym.Env`.
    Uses predetermined `GhostAgentBase`s to produce ghost agent actions, and exposes observations and rewards for the player.
    Passes the inner observation in `info["gym_wrapper_inner_observations"]`.
    """

    def __init__(self, env: PacmanEnv, ghost_builder: Callable[[str], GhostAgentBase]):
        """
        Construct a `GymWrapper` instance wrapping a `PacmanEnv`, supplying ghost actions with ghost agents built from `ghost_builder`.

        Args:
            env (PacmanEnv): The `PacmanEnv` instance to wrap.
            ghost_builder (Callable[[str], GhostAgentBase]): A builder function that takes in the ghost agent name (`str`) and returns a `GhostAgentBase` instance.
        """
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

        info = infos[self.env.player]
        info["gym_wrapper_inner_observations"] = observations
        info["ghost_dones"] = self._ghost_dones

        return (
            observations[self.env.player],
            rewards[self.env.player],
            terminations[self.env.player],
            truncations[self.env.player],
            info,
        )

    # God knows why the line below is a type error.
    def reset(  # type: ignore
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

        info = infos[self.env.player]
        info["gym_wrapper_inner_observations"] = observations
        info["ghost_dones"] = self._ghost_dones

        return (observations[self.env.player], info)

    def render(self) -> str | np.ndarray[Any, np.dtype[np.uint8]]:
        return self.env.render()


class PreviewWrapper(
    Wrapper[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
        np.int64,
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
        np.int64,
    ]
):
    """
    Wrapper to add a preview of expected ghost behavior.
    What shoudl I do? Take the observation, run loops to create expected pos over time.
    I need to create ghost agents based on info, which can only be accessed in reset.
    Overriding reset must be the answer.
    """

    def __init__(
        self,
        env: Env[
            tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int], np.int64
        ],
        ghost_builder: Callable[[str], GhostAgentBase],
        preview_steps: int = 2,
    ):
        super().__init__(env)
        self._preview_steps = preview_steps
        self._ghost_builder = ghost_builder

    def _observation(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
        info: dict[Any, Any],
    ) -> tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int]:
        map_view, player_pos, remaining_power = observation

        if ("gym_wrapper_inner_observations" not in info) or (
            "ghost_dones" not in info
        ):
            raise Exception("`PreviewWrapper` is intended to wrap `GymWrapper`.")

        inner_observation: dict[  # type: ignore
            str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int]
        ] = info["gym_wrapper_inner_observations"]
        ghost_dones: dict[str, bool] = info["ghost_dones"]  # type: ignore

        # `ghost_positions` holds positions of valid (not killed) ghosts.
        ghost_positions: dict[str, tuple[int, int]] = {
            ghost_name: inner_observation[ghost_name][1]
            for (ghost_name, ghost_done) in ghost_dones.items()
            if not ghost_done
        }

        # Copy `map_view` to use it in preview generation loop.
        copied_map_view = map_view.copy()
        # And copy `remaining_power` too.
        copied_remaining_power = remaining_power
        preview_ghosts: list[np.ndarray[Any, np.dtype[np.int8]]] = []
        for _ in range(self._preview_steps):
            # First copy the current step ghosts.
            next_step_ghosts = copied_map_view[GHOST_IDX].copy()
            # Then fetch and apply actions based on current step.
            for ghost_name in ghost_positions:
                ghost_position = ghost_positions[ghost_name]
                ghost = self._ghosts[ghost_name]
                action = ghost.get_action(
                    (copied_map_view, ghost_position, copied_remaining_power)
                )
                if action == STAY:
                    continue
                if action == UP:
                    if ghost_position[0] <= 0:
                        continue
                    new_position = ghost_position[0] - 1, ghost_position[1]
                    if copied_map_view[WALL_IDX][new_position] != 0:
                        continue
                    next_step_ghosts[ghost_position] -= 1
                    next_step_ghosts[new_position] += 1
                    ghost_positions[ghost_name] = new_position
                    continue
                if action == DOWN:
                    if ghost_position[0] >= HEIGHT - 1:
                        continue
                    new_position = ghost_position[0] + 1, ghost_position[1]
                    if copied_map_view[WALL_IDX][new_position] != 0:
                        continue
                    next_step_ghosts[ghost_position] -= 1
                    next_step_ghosts[new_position] += 1
                    ghost_positions[ghost_name] = new_position
                    continue
                if action == LEFT:
                    if ghost_position[1] <= 0:
                        continue
                    new_position = ghost_position[0], ghost_position[1] - 1
                    if copied_map_view[WALL_IDX][new_position] != 0:
                        continue
                    next_step_ghosts[ghost_position] -= 1
                    next_step_ghosts[new_position] += 1
                    ghost_positions[ghost_name] = new_position
                    continue
                if action == RIGHT:
                    if ghost_position[1] >= WIDTH - 1:
                        continue
                    new_position = ghost_position[0], ghost_position[1] + 1
                    if copied_map_view[WALL_IDX][new_position] != 0:
                        continue
                    next_step_ghosts[ghost_position] -= 1
                    next_step_ghosts[new_position] += 1
                    ghost_positions[ghost_name] = new_position
                    continue
                raise Exception("Unreachable")
            # Update the current step, and save it to `preview_ghosts`.
            copied_map_view[GHOST_IDX] = next_step_ghosts
            preview_ghosts.append(next_step_ghosts[np.newaxis, ...])
            if copied_remaining_power > 0:
                copied_remaining_power -= 1

        augmented_map_view = np.concatenate([map_view] + preview_ghosts, axis=0)

        self._last_observation = augmented_map_view, player_pos, remaining_power

        return self._last_observation

    def step(self, action: np.int64) -> tuple[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
        float,
        bool,
        bool,
        dict[Any, Any],
    ]:
        observation, reward, truncation, termination, info = self.env.step(action)
        new_observation = self._observation(observation, info)
        return new_observation, float(reward), truncation, termination, info

    def reset(  # type: ignore
        self, seed: int | None = None, options: dict[Any, Any] | None = None
    ) -> tuple[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int], dict[Any, Any]
    ]:
        observation, info = self.env.reset(seed=seed, options=options)
        if ("gym_wrapper_inner_observations" not in info) or (
            "ghost_dones" not in info
        ):
            raise Exception("`PreviewWrapper` is intended to wrap `GymWrapper`.")

        # Initialize `self._ghosts`.
        ghost_dones: dict[str, bool] = info["ghost_dones"]  # type: ignore
        self._ghosts: dict[str, GhostAgentBase] = {
            ghost_name: self._ghost_builder(ghost_name)
            for ghost_name in ghost_dones.keys()
        }

        return self._observation(observation, info), info

    def render(self):
        """
        Overrided `render()` function that masks away non-observable portions of the full observation.

        Raises:
            NotImplementedError: ANSI render mode is not supported, and will raise an runtime exception.

        Returns:
            np.ndarray[Any, np.dtype[np.uint8]]: A color image of shape `(HEIGHT * 3, WIDTH * 3, 3)`.
        """
        if self.render_mode == "ansi":
            raise NotImplementedError(
                "I am too lazy to implement partial-observable text rendering."
            )
        elif self.render_mode == "rgb_array":
            image: np.ndarray[Any, np.dtype[np.uint8]] = self.env.render()  # type: ignore

            assert image.shape == (HEIGHT * 3, WIDTH * 3, 3)
            assert self._last_observation is not None
            tiled_image = image.reshape((HEIGHT, 3, WIDTH, 3, 3)).transpose(
                (0, 2, 1, 3, 4)
            )
            map_view = self._last_observation[0]
            for step in range(self._preview_steps):
                preview: np.ndarray[Any, np.dtype[np.uint8]] = map_view[5 + step]
                tiled_image_with_warning = (
                    np.array(
                        [
                            int(
                                255 * (self._preview_steps - step) / self._preview_steps
                            ),
                            0,
                            0,
                        ],
                        dtype=np.uint8,
                    )
                    + tiled_image
                ) // 2
                tiled_image = np.where(
                    preview[:, :, np.newaxis, np.newaxis, np.newaxis] > 0,
                    tiled_image_with_warning,
                    tiled_image,
                )
            image = tiled_image.transpose((0, 2, 1, 3, 4)).reshape(
                (HEIGHT * 3, WIDTH * 3, 3)
            )
            return image


class StripWrapper(
    ObservationWrapper[
        np.ndarray[Any, np.dtype[np.int8]],
        np.int64,
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
    ]
):
    """
    Wrapper to strip away player position / remaining power duration from the observation, leaving only the map info.
    """

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
    """
    Wrapper to apply a manhattan distance based visibility limit to the observation. Intended to wrap `GymWrapper`.

    Args:
        ObservationWrapper (_type_): _description_
    """

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
        """
        Overrided `render()` function that masks away non-observable portions of the full observation.

        Raises:
            NotImplementedError: ANSI render mode is not supported, and will raise an runtime exception.

        Returns:
            np.ndarray[Any, np.dtype[np.uint8]]: A color image of shape `(HEIGHT * 3, WIDTH * 3, 3)`.
        """
        if self.render_mode == "ansi":
            raise NotImplementedError(
                "I am too lazy to implement partial-observable text rendering."
            )
        elif self.render_mode == "rgb_array":
            image: np.ndarray[Any, np.dtype[np.uint8]] = super().render()  # type: ignore

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
