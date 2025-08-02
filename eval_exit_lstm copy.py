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
    DecisiveNaiveAttacker,
    GymWrapper,
    PartialObservabilityWrapper,
    FrameStackWrapper,
    DeterministicResetWrapper,
    OraclePreviewWrapper,
    StripWrapper,
)
from train_exit_lstm import Agent, Args

CHECKPOINT_PATH = "results/checkpoints/1753294050__exit_env_v0__exit_env_lstm_5050_long__1/iter_12500.pt"


def load_agent_from_checkpoint(
    checkpoint_path: str,
    observation_space_shape: tuple[int, int, int],
    action_space_shape: int,
) -> tuple[Agent, Args]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args: Args = Args()
    args_dict = checkpoint["args"]
    vars(args).update(args_dict)

    agent = Agent(
        observation_space_shape=observation_space_shape,
        action_space_shape=action_space_shape,
    )
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()

    return (agent, args)


env = ExitEnv(render_mode="rgb_array", random_map=False, max_steps=64)
env = GymWrapper(
    env,
    # lambda: DecisiveNaiveAttacker(
    #     min_safe_distance=3, max_commit_distance=1, stupidity=1
    # ),
    lambda _: UserAttacker(),
)
env = OraclePreviewWrapper(
    env, attacker_builder=lambda _: UserAttacker(), preview_steps=0
)
env = PartialObservabilityWrapper(env)
env = StripWrapper(env)

observation, _ = env.reset(seed=1227)
is_done: bool = False

(agent, _) = load_agent_from_checkpoint(
    CHECKPOINT_PATH,
    observation_space_shape=env.observation_space.shape,  # type: ignore
    action_space_shape=env.action_space.n,  # type: ignore
)

# === NEW: Init hidden LSTM state ===
lstm_state = (
    torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size),
    torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size),
)

done = torch.zeros(1)  # batch of size 1

while True:
    image = Image.fromarray(env.render())  # type: ignore
    image.save("observation_exit.png")
    if is_done:
        print("Environment terminated.")
        break

    # Format obs for batch of size 1
    obs_tensor = torch.tensor(observation[np.newaxis, ...], dtype=torch.float32)

    # Forward with LSTM state + done flag
    with torch.no_grad():
        action, _, _, _, lstm_state = agent.get_action_and_value(
            obs_tensor, lstm_state, done
        )

    action: int = int(action)

    observation, reward, terminated, truncated, _ = env.step(np.int64(action))
    print("Reward: ", reward)
    is_done = terminated or truncated
