from exit import (
    ExitEnv,
    STAY,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    DEFENDER,
    AttackerAgentBase,
    RandomPursueAttacker,
    DeceptiveRandomAttacker,
    GymWrapper,
    PartialObservabilityWrapper,
    StripWrapper,
    FrameStackWrapper,
    DeterministicResetWrapper,
    OraclePreviewWrapper,
    StupidPreviewWrapper,
)
from rich import print
import numpy as np
from PIL import Image


def attacker_builder(seed: int | None) -> AttackerAgentBase:
    return DeceptiveRandomAttacker(
        seed=seed,
        min_safe_distance=3,
        max_commit_distance=1,
        stupidity=1,
        ignore_defender=False,
    )


env = ExitEnv(render_mode="rgb_array", random_map=False)
env = GymWrapper(
    env,
    attacker_builder,
)
# env = GymWrapper(env, lambda: StupidAttacker(NaiveExitAttacker(), stupidity=2))
env = StupidPreviewWrapper(env, preview_steps=4)
env = PartialObservabilityWrapper(env)
env = StripWrapper(env)
env = DeterministicResetWrapper(env)
# observation, _ = env.reset()
observation, _ = env.reset(seed=1220, options={"increment_seed_by": 2})
is_done: bool = False

while True:
    print(observation[4])
    image = Image.fromarray(env.render())  # type: ignore
    image.save("observation_exit.png")
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
    elif action == "reset":
        observation, _ = env.reset()
        continue
    else:
        continue
    observation, reward, terminated, truncated, _ = env.step(np.int64(action))
    is_done = terminated or truncated
