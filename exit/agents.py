import numpy as np
from typing import Any
from abc import ABC, abstractmethod
from pathfinding.core.grid import Grid  # type: ignore
from .core import STAY, UP, DOWN, LEFT, RIGHT
from .env import find_path

DEFAULT_STUPIDITY = 3


class AttackerAgentBase(ABC):
    @abstractmethod
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        return STAY


class IdleAttacker(AttackerAgentBase):
    def get_action(
        self,
        observation: tuple[np.ndarray[Any, np.dtype[np.int8]], tuple[int, int]],
    ) -> int:
        return STAY
