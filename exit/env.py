from pettingzoo import ParallelEnv  # type: ignore
from gymnasium.spaces import MultiBinary, Discrete, Tuple
import numpy as np
from typing import Any
from colorama import Fore
from pathfinding.core.grid import Grid, GridNode  # type: ignore
from pathfinding.finder.a_star import AStarFinder  # type: ignore
from .core import STAY, UP, DOWN, LEFT, RIGHT
from .core import ExitCore, Event, AgentType
from .core import HEIGHT, WIDTH
from .core import WALL, EXIT, DECOY, ATTACKER, DEFENDER

WALL_IDX = 0
EXIT_IDX = 1
DECOY_IDX = 2
ATTACKER_IDX = 3
DEFENDER_IDX = 4

WIN_REWARD = 10
LOSE_REWARD = -10


def find_path(
    grid: Grid,
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[list[GridNode], list[int]]:
    start_node = grid.node(x=start[1], y=start[0])  # type: ignore
    end_node = grid.node(x=end[1], y=end[0])  # type: ignore
    finder = AStarFinder()
    path: list[GridNode] = finder.find_path(start_node, end_node, grid)[0]  # type: ignore
    actions: list[int] = []
    curr_node = start_node
    for path_node in path[1:]:  # type: ignore
        if curr_node.x == path_node.x:  # type: ignore
            if curr_node.y - 1 == path_node.y:  # type: ignore
                action = UP
            elif curr_node.y + 1 == path_node.y:  # type: ignore
                action = DOWN
            else:
                raise Exception("Unreachable")
        elif curr_node.y == path_node.y:  # type: ignore
            if curr_node.x - 1 == path_node.x:  # type: ignore
                action = LEFT
            elif curr_node.x + 1 == path_node.x:  # type: ignore
                action = RIGHT
            else:
                raise Exception("Unreachable")
        else:
            raise Exception("Unreachable")
        actions.append(action)
        curr_node = path_node  # type: ignore
    return (path, actions)  # type: ignore


class ExitEnv(
    ParallelEnv[str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]], int]
):
    metadata = {"name": "exit_env_v0", "render_modes": ["rgb_array"]}
    attacker_name = "attacker"
    defender_name = "defender"

    def __init__(
        self,
        render_mode: str = "rgb_array",
        random_map: bool = True,
        att_def_distance_reward_coeff: float = 0.1,
        att_exit_distance_reward_coeff: float = 0.15,
        max_steps: int = 256,
    ):
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                '`render_mode` should be one of `ExitEnv.metadata["render_modes"].'
            )
        self.render_mode = render_mode
        self._att_def_distance_reward_coeff = att_def_distance_reward_coeff
        self._att_exit_distance_reward_coeff = att_exit_distance_reward_coeff
        self._max_steps = max_steps
        self._core = ExitCore(random_map=random_map)

    def _name_to_agent_type(self, agent_name: str) -> AgentType | None:
        if agent_name == self.attacker_name:
            return AgentType.ATTACKER
        if agent_name == self.defender_name:
            return AgentType.DEFENDER
        return None

    def observation_space(self, agent: str):
        return Tuple(
            (
                MultiBinary((5, HEIGHT, WIDTH)),
                Tuple((Discrete(HEIGHT), Discrete(WIDTH))),
            )
        )

    def action_space(self, agent: str):
        return Discrete(5)

    def _get_observation(
        self,
    ) -> dict[str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]]]:
        map = self._core.map
        stack: list[np.ndarray[Any, np.dtype[np.int8]] | None] = [
            None,
            None,
            None,
            None,
            None,
        ]
        stack[WALL_IDX] = (map & WALL) != 0
        stack[EXIT_IDX] = (map & EXIT) != 0
        stack[DECOY_IDX] = (map & DECOY) != 0
        stack[ATTACKER_IDX] = (map & ATTACKER) != 0
        stack[DEFENDER_IDX] = (map & DEFENDER) != 0
        stack_: list[np.ndarray[Any, np.dtype[np.int8]]] = stack  # type: ignore
        full_observation = np.stack(
            stack_,
            axis=0,
        )

        observation: dict[
            str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]]
        ] = {}

        attacker_pos: tuple[int, int] | None = self._core.attacker_pos
        if attacker_pos is None:
            attacker_pos = (-1, -1)
        observation[self.attacker_name] = (
            full_observation,
            attacker_pos,
        )

        defender_pos: tuple[int, int] | None = self._core.defender_pos
        if defender_pos is None:
            defender_pos = (-1, -1)
        observation[self.defender_name] = (
            full_observation,
            defender_pos,
        )
        return observation

    def _get_empty_infos(self) -> dict[str, dict[Any, Any]]:
        empty_infos: dict[str, dict[Any, Any]] = {}
        empty_infos[self.attacker_name] = {}
        empty_infos[self.defender_name] = {}
        return empty_infos

    def _compute_score(self, events: list[Event]) -> float:
        score: float = 0
        for event in events:
            if event == Event.ATTACKER_CAPTURED:
                score += WIN_REWARD
            elif event == Event.ATTACKER_EXITED:
                score += LOSE_REWARD
            elif event == Event.ATTACKER_DECOY:
                pass
            else:
                raise Exception("Unreachable")
        return score

    def _get_att_def_distance(self) -> int | None:
        attacker_pos = self._core.attacker_pos
        defender_pos = self._core.defender_pos
        if (attacker_pos is None) or (defender_pos is None):
            return None
        return len(find_path(self._grid, defender_pos, attacker_pos)[1])

    def _get_att_exit_distance(self) -> int | None:
        attacker_pos = self._core.attacker_pos
        exit_pos = self._core.exit_pos
        if (attacker_pos is None) or (exit_pos is None):
            return None
        return len(find_path(self._grid, exit_pos, attacker_pos)[1])

    def reset(
        self, seed: int | None = None, options: dict[Any, Any] | None = None
    ) -> tuple[
        dict[str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]]],
        dict[str, dict[Any, Any]],
    ]:
        self._core.reset(seed)
        walls = (self._core.map & WALL) != 0
        self._grid: Grid = Grid(matrix=1 - walls)

        att_def_distance = self._get_att_def_distance()
        assert (
            att_def_distance is not None
        )  # Distance is never `None` right after reset.
        self._last_att_def_distance: int = att_def_distance

        att_exit_distance = self._get_att_exit_distance()
        assert (
            att_exit_distance is not None
        )  # Distance is never `None` right after reset.
        self._last_att_exit_distance: int = att_exit_distance
        self._step: int = 0
        return self._get_observation(), self._get_empty_infos()

    def step(self, actions: dict[str, int]) -> tuple[
        dict[str, tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[Any, Any]],
    ]:
        self._step += 1
        if self._step > self._max_steps:
            raise Exception("`step()` called after environment termination.")
        for agent_name, action in actions.items():
            agent = self._name_to_agent_type(agent_name)
            if agent is None:
                continue
            self._core.perform_action(agent, action)

        score = self._compute_score(self._core.events)
        # Consume the event queue to compute score.
        self._core.events.clear()

        reward: float = score

        att_def_distance = self._get_att_def_distance()
        if att_def_distance is not None:
            delta_att_def_distance = att_def_distance - self._last_att_def_distance
            self._last_att_def_distance = att_def_distance
            # Defender wants to get closer to the attacker.
            reward -= self._att_def_distance_reward_coeff * delta_att_def_distance

        att_exit_distance = self._get_att_exit_distance()
        if att_exit_distance is not None:
            delta_att_exit_distance = att_exit_distance - self._last_att_exit_distance
            self._last_att_exit_distance = att_exit_distance
            # Defender doesn't want attacker to get close to the exit.
            reward += self._att_exit_distance_reward_coeff * delta_att_exit_distance

        if (not self._core.terminated) and (self._step >= self._max_steps):
            reward += WIN_REWARD

        rewards: dict[str, float] = {
            self.attacker_name: -reward,
            self.defender_name: reward,
        }

        dones: dict[str, bool] = {
            self.attacker_name: self._core.terminated
            or (self._step >= self._max_steps),
            self.defender_name: self._core.terminated
            or (self._step >= self._max_steps),
        }

        return self._get_observation(), rewards, dones, dones, self._get_empty_infos()

    def render(self) -> np.ndarray[Any, np.dtype[np.uint8]]:
        if self.render_mode == "rgb_array":
            return ExitEnv.render_observation_rgb(
                self._get_observation()[self.defender_name][0]
            )
        else:
            raise Exception("Unreachable")

    @classmethod
    def render_observation_rgb(
        cls,
        observation: np.ndarray[Any, np.dtype[np.int8]],
    ) -> np.ndarray[Any, np.dtype[np.uint8]]:
        tiles = observation
        reconstructed_map = (
            tiles[WALL_IDX] * WALL
            + tiles[EXIT_IDX] * EXIT
            + tiles[DECOY_IDX] * DECOY
            + tiles[ATTACKER_IDX] * ATTACKER
            + tiles[DEFENDER_IDX] * DEFENDER
        )
        image = np.zeros(shape=(HEIGHT, WIDTH, 3, 3, 3), dtype=np.uint8)
        for h in range(HEIGHT):
            for w in range(WIDTH):
                tile: np.int8 = reconstructed_map[h, w]
                image_tile = np.zeros(shape=(3, 3, 3), dtype=np.uint8)
                if tile & WALL:
                    image_tile = image_tile + [255, 255, 255]
                elif tile & DEFENDER:
                    color = [173, 216, 230]
                    image_tile = image_tile + color
                    image_tile[2, 0] = [0, 0, 0]
                    image_tile[2, 2] = [0, 0, 0]
                elif tile & ATTACKER:
                    color = [200, 0, 0]
                    image_tile[0, 1] = color
                    image_tile[1, 1] = color
                    image_tile[2, 0] = color
                    image_tile[2, 2] = color
                elif tile & EXIT:
                    color = [0, 200, 0]
                    image_tile = image_tile + color
                    image_tile[1, 1] = [0, 0, 0]
                    image_tile[2, 1] = [0, 0, 0]
                elif tile & DECOY:
                    color = [200, 200, 0]
                    image_tile = image_tile + color
                    image_tile[1, 1] = [0, 0, 0]
                    image_tile[2, 1] = [0, 0, 0]
                else:
                    pass
                image[h, w] = image_tile
        image = image.transpose((0, 2, 1, 3, 4))
        image = image.reshape((HEIGHT * 3, WIDTH * 3, 3))
        return image
