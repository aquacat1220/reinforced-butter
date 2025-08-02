from gymnasium import Env, ObservationWrapper, Wrapper
import json
import numpy as np
from typing import Any, Callable, Generic, TypeVar
from gymnasium.spaces import MultiBinary, Tuple, Discrete
from openai import OpenAI
from .env import ExitEnv
from .agents import AttackerAgentBase
from .core import HEIGHT, WIDTH, STAY, UP, DOWN, LEFT, RIGHT
from .env import WALL_IDX, EXIT_IDX, DECOY_IDX, ATTACKER_IDX, DEFENDER_IDX

LLM_SYSTEM_PROMPT = """You are a strategic assistant for the DEFENDER in a two-player capture game.

GAME RULES:
- Game is played on a 15x19 grid maze.
- Each tile is either:
  - a wall: not walkable
  - not a wall: walkable
- There are two flags in the maze, but one of them is fake
- Players can move up/down/left/right or stay in place (1 tile per turn)
- Players CANNOT move diagonally
- Players CANNOT move more than one tile, or skip a wall
- Players CANNOT move outside the 15x19 boundaries

ROLES:
- ATTACKER: Knows which of the two flags are real, and tries to reach it before the time limit
- When the ATTACKER reaches the real flag, the attacker wins
- DEFENDER: Doesn't know which flag is real, and must stop the attacker from reaching it
- DEFENDER can capture the ATTACKER by landing on the same tile
- When the defender captures the attacker, or the time limit is reached, the defender wins
- ATTACKER may try to DECEIVE the defender by moving to the fake flag

MAP FORMAT:
- Sent as JSON with:
  - "walkable": list of (row, col) coordinates that are walkable
  - "flags": list of two (row, col) positions
  - "attacker": current attacker position as (row, col)
  - "defender": current defender position as (row, col)
  - "history": list of previous turns, each turn a dict with "attacker" and "defender" positions
  - "time_remaining": number of turns left

YOUR TASK: Predict the attacker's next k moves

GUIDELINE:
The guideline SHOULD NEVER BE DISRESPECTED; follow every single step without skipping
1. FIRST: Analyze movement history to understand attacker behavior
2. THEN: Repeat the following k times
2-1. Given the current attacker position, identify all adjacent tiles
2-2. Check the map to see which adjacent tiles are walkable
2-3. Choose the most likely next move
2-4. Update the current attacker position
3. FINALLY: Validate the prediction
3-1. All moves are 1-tile or stationary, and not diagonal
3-2. All moves are inside map boundaries
3-3. All moves are on "walkable" tiles

OUTPUT FORMAT:
First, explain your logic under the REASONING section.
Then, output the list of predicted coordinates under the PREDICTION section.

EXAMPLE OUTPUT:

REASONING:
- The attacker is currently at (13, 9).
- The two flags are at (5, 17) and (5, 1).
- The defender is at (3, 9), closer to the left flag (5, 1), so the attacker might try to reach the right flag (5, 17) to avoid the defender.
- The attacker can move up, down, left, right, or stay in place, but only on walkable tiles.
- From the walkable tiles, the attacker has options to move mostly upward and rightward to approach the right flag.
- The attacker is likely to move upward and rightward to reach the right flag at (5, 17).
- I will predict moves that move the attacker closer to (5, 17) by moving up and right when possible.

Step-by-step:
1. From (13, 9), possible moves: (12, 9), (13, 8), (13, 10), (14, 9) (14,9) is outside the map (max row 14?), but 14 is within 15 rows, so (14,9) is valid if walkable. Check walkable tiles for these positions:
   - (12, 9) is not in walkable list.
   - (13, 8) is walkable.
   - (13, 10) is walkable.
   - (14, 9) is not in walkable list.
   So options are (13, 8) or (13, 10).
   Moving right (13, 10) is closer to the right flag (column 17), so choose (13, 10).

2. From (13, 10), possible moves: (12, 10), (13, 9), (13, 11), (14, 10)
   - (12, 10) not walkable.
   - (13, 9) walkable.
   - (13, 11) walkable.
   - (14, 10) not walkable.
   Choose (13, 11) to continue moving right.

3. From (13, 11), possible moves: (12, 11), (13, 10), (13, 12), (14, 11)
   - (12, 11) not walkable.
   - (13, 10) walkable.
   - (13, 12) walkable.
   - (14, 11) not walkable.
   Choose (13, 12) to continue right.

4. From (13, 12), possible moves: (12, 12), (13, 11), (13, 13), (14, 12)
   - (12, 12) not walkable.
   - (13, 11) walkable.
   - (13, 13) walkable.
   - (14, 12) not walkable.
   Choose (13, 13) to continue right.

PREDICTION:
(13, 10)
(13, 11)
(13, 12)
(13, 13)
"""


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
        self.max_steps = env.max_steps
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

        self._steps_remaining -= 1

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
        info["steps_remaining"] = self._steps_remaining

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
        observations, infos = self.env.reset(seed, options)

        self._attacker = self._attacker_builder(seed)
        self._attacker_next_action = self._attacker.get_action(
            observations[self.env.attacker_name]
        )
        self._steps_remaining = self.max_steps

        info = infos[self.env.defender_name]
        info["attacker_observation"] = observations[self.env.attacker_name]
        info["defender_observation"] = observations[self.env.defender_name]
        info["steps_remaining"] = self._steps_remaining

        return (observations[self.env.defender_name], info)

    def render(self) -> np.ndarray[Any, np.dtype[np.uint8]]:
        return self.env.render()


class PreviewWrapper(
    Wrapper[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        np.int64,
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        np.int64,
    ]
):
    """
    Wrapper to add a preview. Base class for all preview wrappers.
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

    def generate_previews(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        info: dict[Any, Any],
    ) -> np.ndarray[Any, np.dtype[np.intp]]:
        return np.array([])

    def _observation(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        info: dict[Any, Any],
    ) -> tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]]:
        preview_positionss = self.generate_previews(observation=observation, info=info)
        preview_attackers = np.zeros(
            (self._preview_steps, HEIGHT, WIDTH), dtype=np.int8
        )

        preview_attackers[tuple(preview_positionss.T)] = 1

        augmented_map_view = np.concatenate([observation[0], preview_attackers], axis=0)

        self._last_observation = augmented_map_view, observation[1]

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


class OraclePreviewWrapper(PreviewWrapper):
    """
    Wrapper to add a preview of expected attacker behavior.
    """

    def __init__(
        self,
        env: Env[tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], np.int64],
        attacker_builder: Callable[[int | None], AttackerAgentBase],
        preview_steps: int = 2,
    ):
        super().__init__(env, preview_steps=preview_steps)
        self._attacker_builder = attacker_builder

    def generate_previews(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        info: dict[Any, Any],
    ) -> np.ndarray[Any, np.dtype[np.intp]]:
        _, defender_pos = observation

        if ("attacker_observation" not in info) or ("defender_observation" not in info):
            raise Exception("`PreviewWrapper` is intended to wrap `GymWrapper`.")

        attacker_observation: tuple[  # type: ignore
            np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]
        ] = info["attacker_observation"]

        map_view, attacker_pos = attacker_observation
        exit_pos = np.argwhere(map_view[EXIT_IDX])[0]
        exit_pos: tuple[int, int] = (exit_pos[0], exit_pos[1])
        # Copy `map_view` to use it in preview generation loop.
        copied_map_view = map_view.copy()
        preview_attackers: list[np.ndarray[Any, np.dtype[np.int8]]] = []
        peeked_once: bool = False
        for _ in range(self._preview_steps):
            # First copy the current step attacker.
            next_step_attacker = copied_map_view[ATTACKER_IDX].copy()
            # Then fetch and apply actions based on current step.
            if attacker_pos == (-1, -1):
                # The last performed action ended the game. Don't peek into a null future.
                action = STAY
            else:
                peeked_once = True
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
            preview_attackers.append(next_step_attacker)
            if attacker_pos == defender_pos or attacker_pos == exit_pos:
                # The game must have ended here. Set `attacker_pos` to `(-1, -1)`.
                attacker_pos = (-1, -1)

        if peeked_once:
            # Get an action to step the attacker.
            _ = self._attacker.get_action(observation=attacker_observation)

        preview_attackers_np = np.array(preview_attackers)
        return np.argwhere(preview_attackers_np == 1)

    def reset(  # type: ignore
        self, seed: int | None = None, options: dict[Any, Any] | None = None
    ) -> tuple[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], dict[Any, Any]
    ]:
        self._attacker: AttackerAgentBase = self._attacker_builder(seed)
        return super().reset(seed=seed, options=options)


class StupidPreviewWrapper(PreviewWrapper):
    """
    Wrapper to add a preview, but with complete random attacker movement.
    """

    def generate_previews(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        info: dict[Any, Any],
    ) -> np.ndarray[Any, np.dtype[np.intp]]:
        _, defender_pos = observation

        if ("attacker_observation" not in info) or ("defender_observation" not in info):
            raise Exception("`PreviewWrapper` is intended to wrap `GymWrapper`.")

        attacker_observation: tuple[  # type: ignore
            np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]
        ] = info["attacker_observation"]

        map_view, attacker_pos = attacker_observation
        # Copy `map_view` to use it in preview generation loop.
        walls = map_view[WALL_IDX]
        possible_attacker_positions = [attacker_pos]
        moves = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        previews: list[tuple[int, int, int]] = []
        preview_attackers: list[np.ndarray[Any, np.dtype[np.int8]]] = []
        for i in range(self._preview_steps):
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
                        previews.append((i, h, w))
            possible_attacker_positions = new_attacker_positions

        return np.array(previews)


class LLMPreviewWrapper(PreviewWrapper):
    _openai_client = None

    def __init__(
        self,
        env: Env[tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], np.int64],
        preview_steps: int = 2,
        model_name: str = "gpt-4.1-mini",
        max_history: int = 8,
        log_file: str | None = None,
    ):
        super().__init__(env=env, preview_steps=preview_steps)
        if self._openai_client is None:
            self._openai_client = OpenAI()
        self._model_name = model_name
        self._max_history = max_history
        if log_file is not None:
            self._log_file = open(log_file, mode="a")
        else:
            self._log_file = None

    def generate_previews(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        info: dict[Any, Any],
    ) -> np.ndarray[Any, np.dtype[np.intp]]:

        user_prompt = self._observation_to_user_prompt(
            observation=observation, info=info
        )

        #         user_prompt = """{
        #     "walkable": [[1, 1], [1, 2], [1, 3], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12], [1, 13], [1, 15], [1, 16], [1, 17],
        #         [2, 1], [2, 3], [2, 5], [2, 13], [2, 15], [2, 17],
        #         [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [3, 14], [3, 15], [3, 16], [3, 17],
        #         [4, 1], [4, 3], [4, 5], [4, 13], [4, 15], [4, 17],
        #         [5, 1], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12], [5, 13], [5, 14], [5, 15], [5, 17],
        #         [6, 1], [6, 6], [6, 12], [6, 17],
        #         [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 12], [7, 13], [7, 14], [7, 15], [7, 16], [7, 17],
        #         [8, 1], [8, 6], [8, 12], [8, 17],
        #         [9, 1], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [9, 11], [9, 12], [9, 13], [9, 14], [9, 15], [9, 17],
        #         [10, 1], [10, 3], [10, 5], [10, 13], [10, 15], [10, 17],
        #         [11, 1], [11, 2], [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10], [11, 11], [11, 12], [11, 13], [11, 14], [11, 15], [11, 16], [11, 17],
        #         [12, 1], [12, 3], [12, 5], [12, 13], [12, 15], [12, 17],
        #         [13, 1], [13, 2], [13, 3], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10], [13, 11], [13, 12], [13, 13], [13, 15], [13, 16], [13, 17]
        #     ],
        #     "flags": [[5, 1], [5, 17]],
        #     "attacker": [13, 10],
        #     "defender": [3, 9],
        #     "history": [
        #         {"attacker": [13, 9], "defender": [3, 9]},
        #         {"attacker": [13, 10], "defender": [3, 9]}
        #     ],
        #     "time_remaining": 30
        # }

        # Predict the attacker's next 4 moves:"""
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": LLM_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        try:
            response = self._openai_client.responses.create(
                model=self._model_name, input=messages, temperature=0  # type: ignore
            )
            print("Received LLM response")

            reasoning, predictions = self._extract_reasoning_and_prediction(
                response.output_text
            )

            assert reasoning is not None and predictions is not None
            if self._log_file is not None:
                self._log_file.write("\nInput:\n")
                self._log_file.write(user_prompt)
                self._log_file.write("\nOutput:\n")
                self._log_file.write(response.output_text)
                self._log_file.flush()
        except:
            print("API Error")
            reasoning = "OpenAI API returned an error."
            predictions = []

        previews = np.array(
            [
                (i, h, w)
                for (i, (h, w)) in enumerate(predictions)
                if self._in_range(i, h, w)
            ]
        )
        return np.array(previews)

    def _in_range(self, i: int, h: int, w: int):
        return (
            0 <= i
            and i < self._preview_steps
            and 0 <= h
            and h < HEIGHT
            and 0 <= w
            and w < WIDTH
        )

    def reset(  # type: ignore
        self, seed: int | None = None, options: dict[Any, Any] | None = None
    ) -> tuple[
        tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], dict[Any, Any]
    ]:
        self._history: list[dict[str, Any]] = []
        if self._log_file is not None:
            self._log_file.write("\nReset\n\n\n\n")
            self._log_file.flush()
        return super().reset(seed=seed, options=options)

    def _observation_to_user_prompt(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
        info: dict[Any, Any],
    ) -> str:
        user_prompt: dict[str, Any] = {}

        walls = observation[0][WALL_IDX]
        walkable: list[list[int]] = np.argwhere(walls == 0).tolist()

        decoys = observation[0][DECOY_IDX]
        decoy_pos: list[int] = np.argwhere(decoys)[0].tolist()

        exits = observation[0][EXIT_IDX]
        exit_pos: list[int] = np.argwhere(exits)[0].tolist()

        attackers = observation[0][ATTACKER_IDX]
        attacker_pos: list[int] = np.argwhere(attackers)[0].tolist()

        defender_pos: list[int] = list(observation[1])

        steps_remaining: int = info["steps_remaining"]  # type: ignore

        if len(self._history) < self._max_history:
            self._history.append({"attacker": attacker_pos, "defender": defender_pos})
        else:
            self._history = self._history[1:] + [
                {"attacker": attacker_pos, "defender": defender_pos}
            ]
        user_prompt["walkable"] = walkable
        user_prompt["flags"] = [exit_pos, decoy_pos]
        user_prompt["attacker"] = attacker_pos
        user_prompt["defender"] = defender_pos
        user_prompt["history"] = self._history
        user_prompt["steps_remaining"] = steps_remaining
        return f"{json.dumps(user_prompt, indent=4, sort_keys=True)}\n\nPredict the attacker's next {self._preview_steps} moves:"

    def _extract_reasoning_and_prediction(
        self, text: str
    ) -> tuple[str | None, list[tuple[int, int]] | None]:
        if "REASONING:" in text and "PREDICTION:" in text:
            reasoning_part = text.split("REASONING:")[1].split("PREDICTION:")[0].strip()
            prediction_part = text.split("PREDICTION:")[1].strip()

            # Manually parse tuples from lines like "(13, 10)"
            predictions: list[tuple[int, int]] = []
            for line in prediction_part.splitlines():
                line = line.strip()
                if line.startswith("(") and line.endswith(")"):
                    # Remove parentheses and split by comma
                    coords = line[1:-1].split(",")
                    if len(coords) == 2:
                        try:
                            row = int(coords[0].strip())
                            col = int(coords[1].strip())
                            predictions.append((row, col))
                        except ValueError:
                            continue  # Skip if not valid integers

            return reasoning_part, predictions
        else:
            return None, None


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
