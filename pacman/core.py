import numpy as np
from typing import Any

NONE = np.int8(0)
WALL = np.int8(1)
PLAYER = np.int8(2)
GHOST = np.int8(4)
DOT = np.int8(8)
POWER = np.int8(16)

# TEMPLATE: np.ndarray[Any, np.dtype[np.int8]] = np.array(
#     [
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 1],
#         [1, 8, 1, 8, 1, 8, 1, 1, 1, 1, 1, 1, 1, 8, 1, 8, 1, 8, 1],
#         [1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1],
#         [1, 8, 1, 8, 1, 8, 1, 1, 1, 1, 1, 1, 1, 8, 1, 8, 1, 8, 1],
#         [1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 1],
#         [1, 8, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 8, 1],
#         [1, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 1],
#         [1, 8, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 8, 1],
#         [1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 1],
#         [1, 8, 1, 8, 1, 8, 1, 1, 1, 1, 1, 1, 1, 8, 1, 8, 1, 8, 1],
#         [1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1],
#         [1, 8, 1, 8, 1, 8, 1, 1, 1, 1, 1, 1, 1, 8, 1, 8, 1, 8, 1],
#         [1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 8, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     ]
# )

TEMPLATE: np.ndarray[Any, np.dtype[np.int8]] = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 8, 8, 8, 1, 8, 8, 8, 8, 8, 8],
        [1, 8, 1, 8, 1, 8, 1, 1, 1, 1, 1],
        [1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [1, 8, 1, 8, 1, 8, 1, 1, 1, 1, 1],
        [1, 8, 1, 8, 8, 8, 8, 8, 8, 8, 8],
        [1, 8, 1, 1, 1, 1, 8, 1, 1, 1, 1],
        [1, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1],
    ]
)

DOTS_IN_TEMPLATE = np.count_nonzero(TEMPLATE & 8)

HEIGHT = len(TEMPLATE)
WIDTH = len(TEMPLATE[0])

STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

DOT_SCORE = 1
POWER_SCORE = 10
GHOST_KILL_SCORE = 10
WIN_SCORE = 1000
LOSE_SCORE = -1000

POWER_DURATION = 10


class PacmanCore:
    def __init__(
        self, player: str, ghosts: list[str] = [], num_power: int = 4, seed: int = 1220
    ):
        self._rng = np.random.default_rng(seed)
        self._player: dict[str, tuple[int, int] | None] = {player: None}
        self._ghosts: dict[str, tuple[int, int] | None] = {
            ghost: None for ghost in ghosts
        }
        self._num_power = num_power
        self._map: np.ndarray[Any, np.dtype[np.int8]] = TEMPLATE.copy()
        self._score: int = 0
        self._player_power_remaining = 0
        self._terminated: bool = True
        self._remaining_dots = DOTS_IN_TEMPLATE
        self.reset()

    def _get_random(self, filter: np.int8 = WALL) -> tuple[int, int]:
        while True:
            h = self._rng.integers(0, HEIGHT, dtype=int)
            w = self._rng.integers(0, WIDTH, dtype=int)
            if not self._map[h, w] & filter:
                return (h, w)

    def _get_adjacent(
        self, position: tuple[int, int], action: int
    ) -> tuple[int, int] | None:
        (pos_h, pos_w) = position
        if action == STAY:
            return (pos_h, pos_w)
        if action == UP:
            if pos_h >= 1:
                return (pos_h - 1, pos_w)
            else:
                return None
        if action == UP:
            if pos_h >= 1:
                return (pos_h - 1, pos_w)
            else:
                return None
        if action == DOWN:
            if pos_h < HEIGHT - 1:
                return (pos_h + 1, pos_w)
            else:
                return None
        if action == LEFT:
            if pos_w >= 1:
                return (pos_h, pos_w - 1)
            else:
                return None
        if action == RIGHT:
            if pos_w < WIDTH - 1:
                return (pos_h, pos_w + 1)
            else:
                return None

    def _valid_agent(self, agent: str) -> bool:
        if agent in self._player or agent in self._ghosts:
            return True
        return False

    def reset(self):
        self._map: np.ndarray[Any, np.dtype[np.int8]] = TEMPLATE.copy()
        self._score: int = 0
        self._player_power_remaining = 0
        self._terminated: bool = False
        self._remaining_dots = DOTS_IN_TEMPLATE

        for player in self._player:
            player_pos = self._get_random(WALL | PLAYER)
            self._map[player_pos] ^= PLAYER
            self._player[player] = player_pos

        for ghost in self._ghosts:
            ghost_pos = self._get_random(WALL | PLAYER | GHOST)
            self._map[ghost_pos] ^= GHOST
            self._ghosts[ghost] = ghost_pos

        for i in range(self._num_power):
            power_pos = self._get_random(WALL | POWER)
            self._map[power_pos] ^= POWER

    def perform_action(self, agent: str, action: int):
        if not self._valid_agent(agent):
            return
        if self._terminated:
            return
        if agent in self._player:
            self.perform_player_action(agent, action)
        if agent in self._ghosts:
            self.perform_ghost_action(agent, action)

    def perform_ghost_action(self, ghost: str, action: int):
        ghost_pos = self._ghosts[ghost]
        if ghost_pos is None:
            return

        if self._player_power_remaining > 0:
            # Ghosts cannot move during power.
            return

        new_pos = self._get_adjacent(ghost_pos, action)
        if new_pos is None:
            # `action` moves ghost out of map boundaries.
            return
        if self._map[new_pos] & WALL:
            # `action` moves ghost into a  wall.
            return

        if self._map[new_pos] & GHOST:
            # `action` moves ghost into another ghost.
            return

        if self._map[new_pos] & PLAYER:
            # If the new tile has a player, remove player from map, update `self._player`, add lose score, and terminate game.
            self._map[new_pos] ^= PLAYER
            for player in self._player:
                if self._player[player] == new_pos:
                    self._player[player] = None
                    break
                raise Exception("Unreachable")
            self._score += LOSE_SCORE
            self._terminated = True

        # And move the ghost.
        self._map[ghost_pos] ^= GHOST
        self._map[new_pos] ^= GHOST
        self._ghosts[ghost] = new_pos
        return

    def perform_player_action(self, player: str, action: int):
        player_pos = self._player[player]
        if player_pos is None:
            return

        if self._player_power_remaining > 0:
            self._player_power_remaining -= 1

        new_pos = self._get_adjacent(player_pos, action)
        if new_pos is None:
            # `action` moves the player out of map boundaries.
            return
        if self._map[new_pos] & WALL:
            # `action` moves the player into a wall.
            return

        if self._map[new_pos] & DOT:
            # If the new tile has a dot, consume it, add to score, and decrement dot count.
            self._map[new_pos] ^= DOT
            self._score += DOT_SCORE
            self._remaining_dots -= 1
            if self._remaining_dots == 0:
                # If all dots were consumed, add win score, and terminate game.
                self._map[player_pos] ^= PLAYER
                self._map[new_pos] ^= PLAYER
                self._player[player] = new_pos
                self._score += WIN_SCORE
                self._terminated = True
                return

        if self._map[new_pos] & POWER:
            # If the new tile has a power, consume it, add to score, and refresh power.
            self._map[new_pos] ^= POWER
            self._score += POWER_SCORE
            self._player_power_remaining = POWER_DURATION

        if self._map[new_pos] & PLAYER:
            # The new tile cannot have a player, since there is always a single player.
            raise Exception("Unreachable.")

        if self._map[new_pos] & GHOST:
            # If the new tile has a ghost, check power.
            if self._player_power_remaining > 0:
                # If we have power, remove ghost from map, update `self._ghosts`, and add to score.
                self._map[new_pos] ^= GHOST
                for ghost in self._ghosts:
                    if self._ghosts[ghost] == new_pos:
                        self._ghosts[ghost] = None
                        break
                self._score += GHOST_KILL_SCORE
            else:
                # If we don't have power, remove player from map, update `self._player`, add lose score, and terminate game.
                self._map[player_pos] ^= PLAYER
                self._player[player] = None
                self._score += LOSE_SCORE
                self._terminated = True
                return
        # If player was not killed by a ghost, move player to new position.
        self._map[player_pos] ^= PLAYER
        self._map[new_pos] ^= PLAYER
        self._player[player] = new_pos
        return
