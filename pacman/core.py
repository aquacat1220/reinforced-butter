import numpy as np
from typing import Any

# Each tile is represented as a single 8bit integer.
# The upper four bits are bit flags that mark if a player/power/dot/wall is present at that tile.
# This is possible because there can be at max one player/power/dot/wall on a single tile.

NONE = np.int8(0b00000000)
WALL = np.int8(0b00010000)
DOT = np.int8(0b00100000)
POWER = np.int8(0b01000000)
PLAYER = np.int8(-0b10000000)
# PLAYER = np.int8(0b10000000)

# However (to stop ghosts from being stuck together), there can be more than one ghost on a single tile.
# Thus we use the remaining 4 bits to hold the number of ghosts on that tile.

GHOST = np.int8(0b00001111)
ONE_GHOST = np.int8(0b00000001)

# fmt: off
# Disable formatting, so that `DOT ,` doesn't get formatted to `DOT,`.
TEMPLATE: np.ndarray[Any, np.dtype[np.int8]] = np.array(
    [
        [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],
        [WALL, DOT , DOT , DOT , WALL, DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , WALL, DOT , DOT , DOT , WALL],
        [WALL, DOT , WALL, DOT , WALL, DOT , WALL, WALL, WALL, WALL, WALL, WALL, WALL, DOT , WALL, DOT , WALL, DOT , WALL],
        [WALL, DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , WALL],
        [WALL, DOT , WALL, DOT , WALL, DOT , WALL, WALL, WALL, WALL, WALL, WALL, WALL, DOT , WALL, DOT , WALL, DOT , WALL],
        [WALL, DOT , WALL, DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , WALL, DOT , WALL],
        [WALL, DOT , WALL, WALL, WALL, WALL, DOT , WALL, WALL, WALL, WALL, WALL, DOT , WALL, WALL, WALL, WALL, DOT , WALL],
        [WALL, DOT , DOT , DOT , DOT , DOT , DOT , WALL, WALL, WALL, WALL, WALL, DOT , DOT , DOT , DOT , DOT , DOT , WALL],
        [WALL, DOT , WALL, WALL, WALL, WALL, DOT , WALL, WALL, WALL, WALL, WALL, DOT , WALL, WALL, WALL, WALL, DOT , WALL],
        [WALL, DOT , WALL, DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , WALL, DOT , WALL],
        [WALL, DOT , WALL, DOT , WALL, DOT , WALL, WALL, WALL, WALL, WALL, WALL, WALL, DOT , WALL, DOT , WALL, DOT , WALL],
        [WALL, DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , WALL],
        [WALL, DOT , WALL, DOT , WALL, DOT , WALL, WALL, WALL, WALL, WALL, WALL, WALL, DOT , WALL, DOT , WALL, DOT , WALL],
        [WALL, DOT , DOT , DOT , WALL, DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , WALL, DOT , DOT , DOT , WALL],
        [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],
    ]
)

# TEMPLATE: np.ndarray[Any, np.dtype[np.int8]] = np.array(
#     [
#         [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],
#         [WALL, DOT , DOT , DOT , WALL, DOT , DOT , DOT , DOT , DOT , DOT ],
#         [WALL, DOT , WALL, DOT , WALL, DOT , WALL, WALL, WALL, WALL, WALL],
#         [WALL, DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT ],
#         [WALL, DOT , WALL, DOT , WALL, DOT , WALL, WALL, WALL, WALL, WALL],
#         [WALL, DOT , WALL, DOT , DOT , DOT , DOT , DOT , DOT , DOT , DOT ],
#         [WALL, DOT , WALL, WALL, WALL, WALL, DOT , WALL, WALL, WALL, WALL],
#         [WALL, DOT , DOT , DOT , DOT , DOT , DOT , WALL, WALL, WALL, WALL],
#     ]
# )
# fmt: on

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
WIN_SCORE = 100
LOSE_SCORE = -100

POWER_DURATION = 10


class PacmanCore:
    def __init__(self, player: str, ghosts: list[str] = [], num_power: int = 4):
        self.player: dict[str, tuple[int, int] | None] = {player: None}
        self.ghosts: dict[str, tuple[int, int] | None] = {
            ghost: None for ghost in ghosts
        }
        self._num_power = num_power
        # self.map: np.ndarray[Any, np.dtype[np.int8]] = TEMPLATE.copy()
        # self.score: int = 0
        # self.player_power_remaining = 0
        # self.terminated: bool = True
        # self._remaining_dots = np.count_nonzero(self.map & DOT)

    def _get_random(self, filter: np.int8 = WALL) -> tuple[int, int]:
        while True:
            h = self._rng.integers(0, HEIGHT, dtype=int)
            w = self._rng.integers(0, WIDTH, dtype=int)
            if not self.map[h, w] & filter:
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
        if agent in self.player or agent in self.ghosts:
            return True
        return False

    def reset(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self.map: np.ndarray[Any, np.dtype[np.int8]] = TEMPLATE.copy()
        self.score: int = 0
        self.player_power_remaining = 0
        self.terminated: bool = False

        for player in self.player:
            player_pos = self._get_random(WALL | PLAYER)
            self.map[player_pos] ^= (
                PLAYER | DOT
            )  # Players are always spawned on dots. Remove the dot we spawned on.
            self.player[player] = player_pos

        for ghost in self.ghosts:
            ghost_pos = self._get_random(WALL | PLAYER | GHOST)
            # Unlike other entities, that can be added by XOR-ing the flag, we need to ADD `ONE_GHOST`, since ghosts are stored as 4 bit uints.
            self.map[ghost_pos] += ONE_GHOST
            self.ghosts[ghost] = ghost_pos

        for _ in range(self._num_power):
            power_pos = self._get_random(WALL | POWER | PLAYER)
            self.map[power_pos] ^= (
                POWER | DOT
            )  # Powers are also always spawned on dots. Remove the dot.

        self._remaining_dots = np.count_nonzero(self.map & DOT)

    def perform_action(self, agent: str, action: int):
        if self.terminated:
            return
        if not self._valid_agent(agent):
            return
        if agent in self.player:
            self.perform_player_action(agent, action)
        if agent in self.ghosts:
            self.perform_ghost_action(agent, action)

    def perform_ghost_action(self, ghost: str, action: int):
        ghost_pos = self.ghosts[ghost]
        if ghost_pos is None:
            return

        if self.player_power_remaining > 0:
            # Ghosts cannot move during power.
            return

        if action == STAY:
            return

        new_pos = self._get_adjacent(ghost_pos, action)
        if new_pos is None:
            # `action` moves ghost out of map boundaries.
            return
        if self.map[new_pos] & WALL:
            # `action` moves ghost into a  wall.
            return

        if self.map[new_pos] & GHOST:
            # `action` moves ghost into another ghost.
            # We allow such action to make sure ghosts don't get stuck together.
            pass

        if self.map[new_pos] & PLAYER:
            # If the new tile has a player, remove player from map, update `self._player`, add lose score, and terminate game.
            self.map[new_pos] ^= PLAYER
            for player in self.player:
                if self.player[player] == new_pos:
                    self.player[player] = None
                    break
                raise Exception("Unreachable")
            self.score += LOSE_SCORE
            self.terminated = True

        # And move the ghost.
        self.map[ghost_pos] -= ONE_GHOST
        self.map[new_pos] += ONE_GHOST
        self.ghosts[ghost] = new_pos
        return

    def perform_player_action(self, player: str, action: int):
        player_pos = self.player[player]
        if player_pos is None:
            return

        if self.player_power_remaining > 0:
            self.player_power_remaining -= 1

        if action == STAY:
            return

        new_pos = self._get_adjacent(player_pos, action)
        if new_pos is None:
            # `action` moves the player out of map boundaries.
            return
        if self.map[new_pos] & WALL:
            # `action` moves the player into a wall.
            return

        if self.map[new_pos] & DOT:
            # If the new tile has a dot, consume it, add to score, and decrement dot count.
            self.map[new_pos] ^= DOT
            self.score += DOT_SCORE
            self._remaining_dots -= 1
            if self._remaining_dots == 0:
                # If all dots were consumed, add win score, and terminate game.
                self.map[player_pos] ^= PLAYER
                self.map[new_pos] ^= PLAYER
                self.player[player] = new_pos
                self.score += WIN_SCORE
                self.terminated = True
                return

        if self.map[new_pos] & POWER:
            # If the new tile has a power, consume it, add to score, and refresh power.
            self.map[new_pos] ^= POWER
            self.score += POWER_SCORE
            self.player_power_remaining = POWER_DURATION

        if self.map[new_pos] & PLAYER:
            # The new tile cannot have a player, since there is always a single player.
            raise Exception("Unreachable.")

        if self.map[new_pos] & GHOST:
            # If the new tile has a ghost, check power.
            if self.player_power_remaining > 0:
                # If we have power, remove all ghosts present in this tile, update `self._ghosts`, and add to score.
                # AND-ing the tile with `~GHOST` (0b11110000) removes all ghosts from the tile.
                self.map[new_pos] &= ~GHOST
                for ghost in self.ghosts:
                    if self.ghosts[ghost] == new_pos:
                        self.ghosts[ghost] = None
                        self.score += GHOST_KILL_SCORE
                # If that was the last ghost, add win score, and terminate game.
                was_last_ghost = True
                for ghost_pos in self.ghosts.values():
                    if ghost_pos is not None:
                        was_last_ghost = False
                        break
                if was_last_ghost:
                    self.score += WIN_SCORE
                    self.terminated = True
            else:
                # If we don't have power, remove player from map, update `self._player`, add lose score, and terminate game.
                self.map[player_pos] ^= PLAYER
                self.player[player] = None
                self.score += LOSE_SCORE
                self.terminated = True
                return
        # If player was not killed by a ghost, move player to new position.
        self.map[player_pos] ^= PLAYER
        self.map[new_pos] ^= PLAYER
        self.player[player] = new_pos
        return
