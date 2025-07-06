from pacman import (
    PacmanEnv,
    STAY,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    PursueGhost,
    PatrolPowerGhost,
    GymWrapper,
)
from rich import print
import numpy as np

env = PacmanEnv()
env = GymWrapper(env, lambda _: PatrolPowerGhost())

# observation, _ = env.reset()
observation, _ = env.reset(seed=1220)
is_done: bool = False

print(observation)
print(env.render())
while True:
    if is_done:
        print("Environment terminated.")
        break
    action = input("Select action: ")
    if action == "s":
        action = STAY
    elif action == "u":
        action = UP
    elif action == "d":
        action = DOWN
    elif action == "l":
        action = LEFT
    elif action == "r":
        action = RIGHT
    else:
        continue
    observation, _, terminated, truncated, _ = env.step(np.int64(action))
    print(observation)
    print(env.render())
    is_done = terminated or truncated
