from gymnasium import Env, ObservationWrapper, Wrapper
import numpy as np
from typing import Any, Callable, Generic, TypeVar
from gymnasium.spaces import MultiBinary, Tuple, Discrete
from .env import ExitEnv
from .agents import AttackerAgentBase
from .core import HEIGHT, WIDTH, STAY, UP, DOWN, LEFT, RIGHT
from .env import WALL_IDX, EXIT_IDX, DECOY_IDX, ATTACKER_IDX, DEFENDER_IDX


class GymWrapper(
    Env[tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], np.int64]
):
    """
    Wrapper to convert `ExitEnv` from a `pettingzoo.ParallelEnv` to a `gym.Env`.
    Uses predetermined `AttackerAgentBase`s to produce attacker agent actions, and exposes observations and rewards for the player.
    Passes the inner observation in `info["gym_wrapper_inner_observations"]`.
    """

    def __init__(
        self, env: ExitEnv, attacker_builder: Callable[[int | None], AttackerAgentBase]
    ):
        """
        Construct a `GymWrapper` instance wrapping a `ExitEnv`, supplying attacker actions with attacker agent built from `attacker_builder`.

        Args:
            env (ExitEnv): The `ExitEnv` instance to wrap.
            attacker_builder (Callable[[int | None], AttackerAgentBase]): A builder function that takes a seed and returns a `AttackerAgentBase` instance.
        """
        self.env = env
        self.metadata = env.metadata
        self.render_mode = env.render_mode
        self.observation_space = Tuple(
            (
                MultiBinary((5, HEIGHT, WIDTH)),
                Tuple((Discrete(HEIGHT), Discrete(WIDTH))),
            )
        )
        self.action_space = Discrete(5)
        self._attacker_builder = attacker_builder

    def step(self, action: np.int64) -> tuple[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        float,
        bool,
        bool,
        dict[Any, Any],
    ]:
        actions: dict[str, int] = {}
        actions[self.env.attacker_name] = self._attacker_next_action
        actions[self.env.defender_name] = int(action)
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        # Fetch next actions if environment wasn't terminated.
        if (not terminations[self.env.attacker_name]) and (
            not truncations[self.env.attacker_name]
        ):
            self._attacker_next_action = self._attacker.get_action(
                observations[self.env.attacker_name]
            )

        info = infos[self.env.defender_name]
        info["attacker_observation"] = observations[self.env.attacker_name]
        info["defender_observation"] = observations[self.env.defender_name]

        return (
            observations[self.env.defender_name],
            rewards[self.env.defender_name],
            terminations[self.env.defender_name],
            truncations[self.env.defender_name],
            info,
        )

    # God knows why the line below is a type error.
    def reset(  # type: ignore
        self, seed: int | None = None, options: dict[Any, Any] | None = None
    ) -> tuple[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], dict[Any, Any]
    ]:
        self._rng: np.random.Generator = np.random.default_rng(seed)
        observations, infos = self.env.reset(seed, options)

        self._attacker = self._attacker_builder(seed)
        self._attacker_next_action = self._attacker.get_action(
            observations[self.env.attacker_name]
        )

        info = infos[self.env.defender_name]
        info["attacker_observation"] = observations[self.env.attacker_name]
        info["defender_observation"] = observations[self.env.defender_name]

        return (observations[self.env.defender_name], info)

    def render(self) -> np.ndarray[Any, np.dtype[np.uint8]]:
        return self.env.render()


class OraclePreviewWrapper(
    Wrapper[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        np.int64,
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        np.int64,
    ]
):
    """
    Wrapper to add a preview of expected attacker behavior.
    """

    def __init__(
        self,
        env: Env[tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], np.int64],
        attacker_builder: Callable[[int | None], AttackerAgentBase],
        preview_steps: int = 2,
    ):
        super().__init__(env)
        self.observation_space = Tuple(
            (
                MultiBinary((5 + preview_steps, HEIGHT, WIDTH)),
                Tuple((Discrete(HEIGHT), Discrete(WIDTH))),
            )
        )
        self.action_space = Discrete(5)
        self._preview_steps = preview_steps
        self._attacker_builder = attacker_builder

    def _observation(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        info: dict[Any, Any],
    ) -> tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]]:
        _, defender_pos = observation

        if ("attacker_observation" not in info) or ("defender_observation" not in info):
            raise Exception("`PreviewWrapper` is intended to wrap `GymWrapper`.")

        attacker_observation: tuple[  # type: ignore
            np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]
        ] = info["attacker_observation"]

        map_view, attacker_pos = attacker_observation
        # Copy `map_view` to use it in preview generation loop.
        copied_map_view = map_view.copy()
        preview_attackers: list[np.ndarray[Any, np.dtype[np.int8]]] = []
        for _ in range(self._preview_steps):
            # First copy the current step attacker.
            next_step_attacker = copied_map_view[ATTACKER_IDX].copy()
            # Then fetch and apply actions based on current step.
            action = self._attacker.peek_action((copied_map_view, attacker_pos))
            if action == STAY:
                pass
            elif action == UP:
                if attacker_pos[0] > 0:
                    new_position = attacker_pos[0] - 1, attacker_pos[1]
                    if copied_map_view[WALL_IDX][new_position] == 0:
                        next_step_attacker[attacker_pos] -= 1
                        next_step_attacker[new_position] += 1
                        attacker_pos = new_position
            elif action == DOWN:
                if attacker_pos[0] < HEIGHT - 1:
                    new_position = attacker_pos[0] + 1, attacker_pos[1]
                    if copied_map_view[WALL_IDX][new_position] == 0:
                        next_step_attacker[attacker_pos] -= 1
                        next_step_attacker[new_position] += 1
                        attacker_pos = new_position
            elif action == LEFT:
                if attacker_pos[1] > 0:
                    new_position = attacker_pos[0], attacker_pos[1] - 1
                    if copied_map_view[WALL_IDX][new_position] == 0:
                        next_step_attacker[attacker_pos] -= 1
                        next_step_attacker[new_position] += 1
                        attacker_pos = new_position
            elif action == RIGHT:
                if attacker_pos[1] < WIDTH - 1:
                    new_position = attacker_pos[0], attacker_pos[1] + 1
                    if copied_map_view[WALL_IDX][new_position] == 0:
                        next_step_attacker[attacker_pos] -= 1
                        next_step_attacker[new_position] += 1
                        attacker_pos = new_position
            else:
                raise Exception("Unreachable")
            # Update the current step, and save it to `preview_ghosts`.
            copied_map_view[ATTACKER_IDX] = next_step_attacker
            preview_attackers.append(next_step_attacker[np.newaxis, ...])

        # Get an action to step the attacker.
        _ = self._attacker.get_action(observation=attacker_observation)

        augmented_map_view = np.concatenate([map_view] + preview_attackers, axis=0)

        self._last_observation = augmented_map_view, defender_pos

        return self._last_observation

    def step(self, action: np.int64) -> tuple[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
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
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], dict[Any, Any]
    ]:
        self._attacker: AttackerAgentBase = self._attacker_builder(seed)
        observation, info = self.env.reset(seed=seed, options=options)

        if ("attacker_observation" not in info) or ("defender_observation" not in info):
            raise Exception("`PreviewWrapper` is intended to wrap `GymWrapper`.")

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
            previews = map_view[5:]
            steps = np.arange(self._preview_steps, 0, -1)[:, np.newaxis, np.newaxis]
            previews = previews * steps
            previews: np.ndarray[Any, np.dtype[np.int8]] = previews.max(axis=0)
            preview_intensities = previews / (self._preview_steps + 1)
            tiled_previews = preview_intensities[
                :, :, np.newaxis, np.newaxis, np.newaxis
            ] * np.array([255, 0, 0])
            tiled_overlay = (tiled_image + tiled_previews) // 2
            tiled_image = np.where(
                previews[:, :, np.newaxis, np.newaxis, np.newaxis] > 0,
                tiled_overlay,
                tiled_image,
            )
            image = (
                tiled_image.astype(np.uint8)
                .transpose((0, 2, 1, 3, 4))
                .reshape((HEIGHT * 3, WIDTH * 3, 3))
            )
            return image


class StupidPreviewWrapper(
    Wrapper[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        np.int64,
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        np.int64,
    ]
):
    """
    Wrapper to add a preview, but with complete random attacker movement.
    """

    def __init__(
        self,
        env: Env[tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], np.int64],
        preview_steps: int = 2,
    ):
        super().__init__(env)
        self.observation_space = Tuple(
            (
                MultiBinary((5 + preview_steps, HEIGHT, WIDTH)),
                Tuple((Discrete(HEIGHT), Discrete(WIDTH))),
            )
        )
        self.action_space = Discrete(5)
        self._preview_steps = preview_steps

    def _observation(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        info: dict[Any, Any],
    ) -> tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]]:
        _, defender_pos = observation

        if ("attacker_observation" not in info) or ("defender_observation" not in info):
            raise Exception("`PreviewWrapper` is intended to wrap `GymWrapper`.")

        attacker_observation: tuple[  # type: ignore
            np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]
        ] = info["attacker_observation"]

        map_view, attacker_pos = attacker_observation
        # Copy `map_view` to use it in preview generation loop.
        walls = map_view[WALL_IDX]
        zeros = np.zeros_like(walls)
        possible_attacker_positions = [attacker_pos]
        moves = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        preview_attackers: list[np.ndarray[Any, np.dtype[np.int8]]] = []
        for _ in range(self._preview_steps):
            new_attacker_positions: list[tuple[int, int]] = []
            for possible_attacker_position in possible_attacker_positions:
                for move in moves:
                    h = possible_attacker_position[0] + move[0]
                    w = possible_attacker_position[1] + move[1]
                    if (
                        0 <= h
                        and h < HEIGHT
                        and 0 <= w
                        and w < WIDTH
                        and walls[h, w] == 0
                    ):
                        new_attacker_positions.append((h, w))
            possible_attacker_positions = new_attacker_positions
            attackers = zeros.copy()
            for possible_attacker_position in possible_attacker_positions:
                attackers[possible_attacker_position] = 1

            preview_attackers.append(attackers[np.newaxis, ...])

        augmented_map_view = np.concatenate([map_view] + preview_attackers, axis=0)

        self._last_observation = augmented_map_view, defender_pos

        return self._last_observation

    def step(self, action: np.int64) -> tuple[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
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
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], dict[Any, Any]
    ]:
        observation, info = self.env.reset(seed=seed, options=options)

        if ("attacker_observation" not in info) or ("defender_observation" not in info):
            raise Exception("`PreviewWrapper` is intended to wrap `GymWrapper`.")

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
            previews = map_view[5:]
            steps = np.arange(self._preview_steps, 0, -1)[:, np.newaxis, np.newaxis]
            previews = previews * steps
            previews: np.ndarray[Any, np.dtype[np.int8]] = previews.max(axis=0)
            preview_intensities = previews / (self._preview_steps + 1)
            tiled_previews = preview_intensities[
                :, :, np.newaxis, np.newaxis, np.newaxis
            ] * np.array([255, 0, 0])
            tiled_overlay = (tiled_image + tiled_previews) // 2
            tiled_image = np.where(
                previews[:, :, np.newaxis, np.newaxis, np.newaxis] > 0,
                tiled_overlay,
                tiled_image,
            )
            image = (
                tiled_image.astype(np.uint8)
                .transpose((0, 2, 1, 3, 4))
                .reshape((HEIGHT * 3, WIDTH * 3, 3))
            )
            return image


class StripWrapper(
    ObservationWrapper[
        np.ndarray[Any, np.dtype[np.int8]],
        np.int64,
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ]
):
    """
    Wrapper to strip away player position from the observation, leaving only the map info.
    """

    def __init__(
        self,
        env: Env[tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], np.int64],
    ):
        super().__init__(env)
        self.observation_space = self.env.observation_space.spaces[0]  # type: ignore

    def observation(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> np.ndarray[Any, np.dtype[np.int8]]:
        return observation[0]


class PartialObservabilityWrapper(
    ObservationWrapper[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        np.int64,
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ]
):
    """
    Wrapper to merge obseravtion for decoy and exit. Intended to wrap `GymWrapper` or `PreviewWrapper`.
    """

    def __init__(
        self,
        env: Env[tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], np.int64],
        render_partial: bool = False,
    ):
        super().__init__(env)
        self._render_partial = render_partial
        n: int = self.env.observation_space.spaces[0].n[0]  # type: ignore
        self.observation_space = Tuple(
            (
                MultiBinary((n - 1, HEIGHT, WIDTH)),  # type: ignore
                Tuple((Discrete(HEIGHT), Discrete(WIDTH))),
            )
        )

    def observation(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]]:
        decoys = observation[0][DECOY_IDX]
        exits = observation[0][EXIT_IDX]
        merged_exits = decoys + exits

        return (
            np.concatenate(
                [
                    observation[0][0][np.newaxis, ...],
                    merged_exits[np.newaxis, ...],
                    observation[0][3:],
                ],
                axis=0,
            ),
            observation[1],
        )

    def render(self) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """
        Overrided `render()` function that renders the true exit and the decoy.

        Raises:
            NotImplementedError: ANSI render mode is not supported, and will raise an runtime exception.

        Returns:
            np.ndarray[Any, np.dtype[np.uint8]]: A color image of shape `(HEIGHT * 3, WIDTH * 3, 3)`.
        """
        if not self._render_partial:
            return self.env.render()  # type: ignore
        if self.render_mode == "ansi":
            raise NotImplementedError(
                "I am too lazy to implement partial-observable text rendering."
            )
        elif self.render_mode == "rgb_array":
            image: np.ndarray[Any, np.dtype[np.uint8]] = self.env.render()  # type: ignore

            assert image.shape == (HEIGHT * 3, WIDTH * 3, 3)
            yellow_pixels = np.all(
                image == np.array([200, 200, 0], dtype=np.uint8), axis=-1
            )
            image = np.where(
                yellow_pixels[:, :, np.newaxis],
                np.array([0, 200, 0], dtype=np.uint8),
                image,
            )
            return image


class FrameStackWrapper(
    ObservationWrapper[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        np.int64,
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ]
):
    """
    Wrapper to stack obseravtions of multiple past frames. Stacks only the mutable parts (attacker/defender) to save space.
    """

    def __init__(
        self,
        env: Env[tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], np.int64],
        history_length: int = 2,
    ):
        super().__init__(env)
        self._history_length: int = history_length
        self._history: list[np.ndarray[Any, np.dtype[np.int8]]] = []
        n: int = self.env.observation_space.spaces[0].n[0]  # type: ignore
        self.observation_space = Tuple(
            (
                MultiBinary((n + 2 * self._history_length, HEIGHT, WIDTH)),  # type: ignore
                Tuple((Discrete(HEIGHT), Discrete(WIDTH))),
            )
        )

    def observation(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]]:
        mutables = observation[0][2:4]
        if len(self._history) == 0:
            self._history = [mutables for _ in range(self._history_length)]
        assert len(self._history) == self._history_length

        concated = np.concatenate([observation[0]] + self._history, axis=0)
        self._history = [mutables] + self._history[:-1]
        return concated, observation[1]


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class DeterministicResetWrapper(
    Wrapper[ObsType, ActType, ObsType, ActType],
    Generic[ObsType, ActType],
):
    """Wrapper to support deterministic resets even when `seed == None` through the use of `options["increment_seed_by"]`."""

    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)
        self._last_seed: int | None = None
        self._increment_seed_by: int | None = None

    def reset(  # type: ignore
        self, seed: int | None = None, options: dict[Any, Any] | None = None
    ) -> tuple[
        ObsType,
        dict[Any, Any],
    ]:
        increment_seed_by: int | None = None
        if (options is not None) and ("increment_seed_by" in options):
            increment_seed_by = options["increment_seed_by"]

        if (seed is None) and (increment_seed_by is not None):
            raise ValueError(
                'key `"increment_seed_by"` can only be used in `options` with non-None `seed`.'
            )

        if seed is None:
            if self._increment_seed_by is not None:
                # `self._increment_seed_by` and `self._last_seed`'s None-ness are synced.
                assert self._last_seed is not None
                self._last_seed += self._increment_seed_by
                return self.env.reset(seed=self._last_seed, options=options)
            else:
                return self.env.reset(seed=seed, options=options)
        else:
            if increment_seed_by is not None:
                self._last_seed = seed
                self._increment_seed_by = increment_seed_by
                return self.env.reset(seed=self._last_seed, options=options)
            else:
                self._last_seed = None
                self._increment_seed_by = None
                return self.env.reset(seed=seed, options=options)
