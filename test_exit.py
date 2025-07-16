from exit import (
    ExitEnv,
    STAY,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    StupidAttacker,
    NaiveExitAttacker,
    GymWrapper,
    PartialObservabilityWrapper,
    StripWrapper,
)
from rich import print
import numpy as np
from PIL import Image

env = ExitEnv(render_mode="rgb_array")
env = GymWrapper(env, lambda: StupidAttacker(NaiveExitAttacker(), stupidity=2))
env = PartialObservabilityWrapper(env)
env = StripWrapper(env)
# observation, _ = env.reset()
observation, _ = env.reset(seed=1221)
is_done: bool = False

print(observation)
image = Image.fromarray(env.render())  # type: ignore
image.save("observation_exit.png")
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
    observation, reward, terminated, truncated, _ = env.step(np.int64(action))
    print(observation)
    print("Reward: ", reward)
    image = Image.fromarray(env.render())  # type: ignore
    image.save("observation_exit.png")
    is_done = terminated or truncated
