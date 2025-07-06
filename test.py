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
    StripWrapper,
)
from rich import print
import numpy as np
from PIL import Image

env = PacmanEnv(render_mode="rgb_array")
env = GymWrapper(env, lambda _: PatrolPowerGhost())
env = StripWrapper(env)

# observation, _ = env.reset()
observation, _ = env.reset(seed=1220)
is_done: bool = False

print(observation)
image = Image.fromarray(env.render())  # type: ignore
image.save("observation.png")
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
    image = Image.fromarray(env.render())  # type: ignore
    image.save("observation.png")
    is_done = terminated or truncated
