# %% ---------------- Imports and declarations. ----------------
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
    LLMPreviewWrapper,
)
from rich import print
import numpy as np
from PIL import Image


def attacker_builder(seed: int | None) -> AttackerAgentBase:
    # return DeceptiveRandomAttacker(
    #     seed=seed,
    #     min_safe_distance=3,
    #     max_commit_distance=1,
    #     stupidity=1,
    #     ignore_defender=False,
    # )
    return RandomPursueAttacker(seed=seed)


# %% ---------------- Create the environment. ----------------
env = ExitEnv(render_mode="rgb_array", random_map=False, max_steps=32)
RENDER_AS_DEFENDER = False
# %% ---------------- Wrap the environment with "wrappers". ----------------
env = GymWrapper(
    env,
    attacker_builder,
)
# env = StupidPreviewWrapper(env, preview_steps=4)
env = LLMPreviewWrapper(
    env, preview_steps=4, model_name="gpt-4.1-nano", log_file="./llm.log"
)
# env = OraclePreviewWrapper(env, preview_steps=4, attacker_builder=attacker_builder)
# Set `render_partial` to `True` to make the decoy undistinguishable to the real exit.
env = PartialObservabilityWrapper(env, render_partial=RENDER_AS_DEFENDER)
env = StripWrapper(env)
env = DeterministicResetWrapper(env)
# %% ---------------- Reset the environment. ----------------
observation, info = env.reset(seed=1220, options={"increment_seed_by": 2})
is_done: bool = False
step = 0

# %% ---------------- Main loop. ----------------
while True:
    # %% ---------------- Render the current observation. ----------------
    steps_remaining = info["steps_remaining"]
    print(f"{steps_remaining} turns remaining!")  # type: ignore
    step += 1
    # print(observation[4])
    image = Image.fromarray(env.render())  # type: ignore
    image.save("observation_exit.png")
    if is_done:
        print("Environment terminated.")
        if reward < 0:  # type: ignore
            print("Attacker wins!")
        else:
            print("Defender wins!")
        break

    # %% ---------------- Fetch action from user input. ----------------
    action = input("Select action: ")
    if action == "s" or action == "":
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

    # %% ---------------- Feed the action to the environment. ----------------
    observation, reward, terminated, truncated, info = env.step(np.int64(action))
    is_done = terminated or truncated

    # %% ---------------- ... and repeat until termination! ----------------
