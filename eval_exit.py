import torch
from torch.distributions.categorical import Categorical
import gymnasium as gym
import numpy as np
from rich import print
from PIL import Image
from exit import (
    ExitEnv,
    STAY,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    UserAttacker,
    GymWrapper,
    PartialObservabilityWrapper,
    StripWrapper,
)
from train_exit import Agent, Args

CHECKPOINT_PATH = (
    "checkpoints/exit_env_v0__exit_env_switching_attacker__1__1752649249/iter_48500.pt"
)


def load_agent_from_checkpoint(
    checkpoint_path: str,
    observation_space_shape: tuple[int, int, int],
    action_space_shape: int,
) -> tuple[Agent, Args]:
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args: Args = Args()
    args_dict = checkpoint["args"]
    vars(args).update(args_dict)

    # Create agent and load state dict
    agent = Agent(
        observation_space_shape=observation_space_shape,
        action_space_shape=action_space_shape,
    )
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    return (agent, args)


env = ExitEnv(render_mode="rgb_array")
env = GymWrapper(env, lambda: UserAttacker())
env = PartialObservabilityWrapper(env)
env = StripWrapper(env)
# observation, _ = env.reset()
observation, _ = env.reset(seed=1221)
is_done: bool = False

(agent, _) = load_agent_from_checkpoint(
    CHECKPOINT_PATH,
    observation_space_shape=env.observation_space.shape,  # type: ignore
    action_space_shape=env.action_space.n,  # type: ignore
)

while True:
    if is_done:
        print("Environment terminated.")
        break
    # input("Press any key for next step: ")

    # Add a new axis to act like a batch of size 1.
    observation = torch.Tensor(observation[np.newaxis, ...])  # type: ignore
    action = agent.get_action_and_value(observation)[0]  # type: ignore
    action: int = int(action)

    observation, reward, terminated, truncated, _ = env.step(np.int64(action))
    print("Reward: ", reward)
    is_done = terminated or truncated
