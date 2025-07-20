from gymnasium import Env, ObservationWrapper, Wrapper
import numpy as np
from typing import Any, Callable, Generic, TypeVar
from gymnasium.spaces import MultiBinary, Tuple, Discrete
from .env import ExitEnv
from .agents import AttackerAgentBase
from .core import HEIGHT, WIDTH
from .env import WALL_IDX, EXIT_IDX, DECOY_IDX, ATTACKER_IDX, DEFENDER_IDX


class GymWrapper(
    Env[tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], np.int64]
):
    """
    Wrapper to convert `ExitEnv` from a `pettingzoo.ParallelEnv` to a `gym.Env`.
    Uses predetermined `AttackerAgentBase`s to produce attacker agent actions, and exposes observations and rewards for the player.
    Passes the inner observation in `info["gym_wrapper_inner_observations"]`.
    """

    def __init__(self, env: ExitEnv, attacker_builder: Callable[[], AttackerAgentBase]):
        """
        Construct a `GymWrapper` instance wrapping a `ExitEnv`, supplying attacker actions with attacker agent built from `attacker_builder`.

        Args:
            env (ExitEnv): The `ExitEnv` instance to wrap.
            attacker_builder (Callable[[], AttackerAgentBase]): A builder function that returns a `AttackerAgentBase` instance.
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
        attacker_action_candidates = self._attacker_next_action
        actions[self.env.attacker_name] = self._rng.choice(attacker_action_candidates)
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
        info["gym_wrapper_inner_observations"] = observations

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

        self._attacker = self._attacker_builder()
        self._attacker_next_action = self._attacker.get_action(
            observations[self.env.attacker_name]
        )

        info = infos[self.env.defender_name]
        info["gym_wrapper_inner_observations"] = observations

        return (observations[self.env.defender_name], info)

    def render(self) -> np.ndarray[Any, np.dtype[np.uint8]]:
        return self.env.render()


# class PreviewWrapper(
#     Wrapper[
#         tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
#         np.int64,
#         tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
#         np.int64,
#     ]
# ):
#     """
#     Wrapper to add a preview of expected ghost behavior.
#     What shoudl I do? Take the observation, run loops to create expected pos over time.
#     I need to create ghost agents based on info, which can only be accessed in reset.
#     Overriding reset must be the answer.
#     """

#     def __init__(
#         self,
#         env: Env[
#             tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int], np.int64
#         ],
#         ghost_builder: Callable[[str], GhostAgentBase],
#         preview_steps: int = 2,
#     ):
#         super().__init__(env)
#         self.observation_space = Tuple(
#             (
#                 MultiBinary((5 + preview_steps, HEIGHT, WIDTH)),
#                 Tuple((Discrete(HEIGHT), Discrete(WIDTH))),
#                 Discrete(POWER_DURATION + 1),
#             )
#         )
#         self.action_space = Discrete(5)
#         self._preview_steps = preview_steps
#         self._ghost_builder = ghost_builder

#     def _observation(
#         self,
#         observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
#         info: dict[Any, Any],
#     ) -> tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int]:
#         map_view, player_pos, remaining_power = observation

#         if ("gym_wrapper_inner_observations" not in info) or (
#             "ghost_dones" not in info
#         ):
#             raise Exception("`PreviewWrapper` is intended to wrap `GymWrapper`.")

#         inner_observation: dict[  # type: ignore
#             str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int]
#         ] = info["gym_wrapper_inner_observations"]
#         ghost_dones: dict[str, bool] = info["ghost_dones"]  # type: ignore

#         # `ghost_positions` holds positions of valid (not killed) ghosts.
#         ghost_positions: dict[str, tuple[int, int]] = {
#             ghost_name: inner_observation[ghost_name][1]
#             for (ghost_name, ghost_done) in ghost_dones.items()
#             if not ghost_done
#         }

#         # Copy `map_view` to use it in preview generation loop.
#         copied_map_view = map_view.copy()
#         # And copy `remaining_power` too.
#         copied_remaining_power = remaining_power
#         preview_ghosts: list[np.ndarray[Any, np.dtype[np.int8]]] = []
#         for _ in range(self._preview_steps):
#             # First copy the current step ghosts.
#             next_step_ghosts = copied_map_view[GHOST_IDX].copy()
#             # Then fetch and apply actions based on current step.
#             for ghost_name in ghost_positions:
#                 ghost_position = ghost_positions[ghost_name]
#                 ghost = self._ghosts[ghost_name]
#                 action = ghost.get_action(
#                     (copied_map_view, ghost_position, copied_remaining_power)
#                 )
#                 if action == STAY:
#                     continue
#                 if action == UP:
#                     if ghost_position[0] <= 0:
#                         continue
#                     new_position = ghost_position[0] - 1, ghost_position[1]
#                     if copied_map_view[WALL_IDX][new_position] != 0:
#                         continue
#                     next_step_ghosts[ghost_position] -= 1
#                     next_step_ghosts[new_position] += 1
#                     ghost_positions[ghost_name] = new_position
#                     continue
#                 if action == DOWN:
#                     if ghost_position[0] >= HEIGHT - 1:
#                         continue
#                     new_position = ghost_position[0] + 1, ghost_position[1]
#                     if copied_map_view[WALL_IDX][new_position] != 0:
#                         continue
#                     next_step_ghosts[ghost_position] -= 1
#                     next_step_ghosts[new_position] += 1
#                     ghost_positions[ghost_name] = new_position
#                     continue
#                 if action == LEFT:
#                     if ghost_position[1] <= 0:
#                         continue
#                     new_position = ghost_position[0], ghost_position[1] - 1
#                     if copied_map_view[WALL_IDX][new_position] != 0:
#                         continue
#                     next_step_ghosts[ghost_position] -= 1
#                     next_step_ghosts[new_position] += 1
#                     ghost_positions[ghost_name] = new_position
#                     continue
#                 if action == RIGHT:
#                     if ghost_position[1] >= WIDTH - 1:
#                         continue
#                     new_position = ghost_position[0], ghost_position[1] + 1
#                     if copied_map_view[WALL_IDX][new_position] != 0:
#                         continue
#                     next_step_ghosts[ghost_position] -= 1
#                     next_step_ghosts[new_position] += 1
#                     ghost_positions[ghost_name] = new_position
#                     continue
#                 raise Exception("Unreachable")
#             # Update the current step, and save it to `preview_ghosts`.
#             copied_map_view[GHOST_IDX] = next_step_ghosts
#             preview_ghosts.append(next_step_ghosts[np.newaxis, ...])
#             if copied_remaining_power > 0:
#                 copied_remaining_power -= 1

#         augmented_map_view = np.concatenate([map_view] + preview_ghosts, axis=0)

#         self._last_observation = augmented_map_view, player_pos, remaining_power

#         return self._last_observation

#     def step(self, action: np.int64) -> tuple[
#         tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int],
#         float,
#         bool,
#         bool,
#         dict[Any, Any],
#     ]:
#         observation, reward, truncation, termination, info = self.env.step(action)
#         new_observation = self._observation(observation, info)
#         return new_observation, float(reward), truncation, termination, info

#     def reset(  # type: ignore
#         self, seed: int | None = None, options: dict[Any, Any] | None = None
#     ) -> tuple[
#         tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int], int], dict[Any, Any]
#     ]:
#         observation, info = self.env.reset(seed=seed, options=options)
#         if ("gym_wrapper_inner_observations" not in info) or (
#             "ghost_dones" not in info
#         ):
#             raise Exception("`PreviewWrapper` is intended to wrap `GymWrapper`.")

#         # Initialize `self._ghosts`.
#         ghost_dones: dict[str, bool] = info["ghost_dones"]  # type: ignore
#         self._ghosts: dict[str, GhostAgentBase] = {
#             ghost_name: self._ghost_builder(ghost_name)
#             for ghost_name in ghost_dones.keys()
#         }

#         return self._observation(observation, info), info

#     def render(self):
#         """
#         Overrided `render()` function that masks away non-observable portions of the full observation.

#         Raises:
#             NotImplementedError: ANSI render mode is not supported, and will raise an runtime exception.

#         Returns:
#             np.ndarray[Any, np.dtype[np.uint8]]: A color image of shape `(HEIGHT * 3, WIDTH * 3, 3)`.
#         """
#         if self.render_mode == "ansi":
#             raise NotImplementedError(
#                 "I am too lazy to implement partial-observable text rendering."
#             )
#         elif self.render_mode == "rgb_array":
#             image: np.ndarray[Any, np.dtype[np.uint8]] = self.env.render()  # type: ignore

#             assert image.shape == (HEIGHT * 3, WIDTH * 3, 3)
#             assert self._last_observation is not None
#             tiled_image = image.reshape((HEIGHT, 3, WIDTH, 3, 3)).transpose(
#                 (0, 2, 1, 3, 4)
#             )
#             map_view = self._last_observation[0]
#             for step in range(self._preview_steps):
#                 preview: np.ndarray[Any, np.dtype[np.uint8]] = map_view[5 + step]
#                 tiled_image_with_warning = (
#                     np.array(
#                         [
#                             int(
#                                 255 * (self._preview_steps - step) / self._preview_steps
#                             ),
#                             0,
#                             0,
#                         ],
#                         dtype=np.uint8,
#                     )
#                     + tiled_image
#                 ) // 2
#                 tiled_image = np.where(
#                     preview[:, :, np.newaxis, np.newaxis, np.newaxis] > 0,
#                     tiled_image_with_warning,
#                     tiled_image,
#                 )
#             image = tiled_image.transpose((0, 2, 1, 3, 4)).reshape(
#                 (HEIGHT * 3, WIDTH * 3, 3)
#             )
#             return image


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
    Wrapper to merge obseravtion for decoy and exit. Intended to wrap `GymWrapper`.
    """

    def __init__(
        self,
        env: Env[tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], np.int64],
    ):
        super().__init__(env)
        self.observation_space = Tuple(
            (
                MultiBinary((4, HEIGHT, WIDTH)),
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

        stack: list[np.ndarray[Any, np.dtype[np.int8]] | None] = [
            None,
            None,
            None,
            None,
        ]
        stack[0] = observation[0][WALL_IDX]
        stack[1] = merged_exits
        stack[2] = observation[0][ATTACKER_IDX]
        stack[3] = observation[0][DEFENDER_IDX]
        stack_: list[np.ndarray[Any, np.dtype[np.int8]]] = stack  # type: ignore
        return (np.stack(stack_, axis=0), observation[1])


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
        self.observation_space = Tuple(
            (
                MultiBinary((4 + 2 * self._history_length, HEIGHT, WIDTH)),
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

    def __init__(self, env: Env[ObsType, ActType], is_enabled: bool = True):
        super().__init__(env)
        self._is_enabled = is_enabled
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
