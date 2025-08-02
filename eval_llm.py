import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from rich import print
import time
from exit import (
    ExitEnv,
    RandomSelectAttacker,
    DecisiveRandomNaiveAttacker,
    DeceptiveRandomAttacker,
    GymWrapper,
    PartialObservabilityWrapper,
    DeterministicResetWrapper,
    LLMPreviewWrapper,
    StripWrapper,
)
from train_exit_lstm import Agent, Args

CHECKPOINT_PATH = "results/checkpoints/1753538886__exit_env_v0__exit_env_lstm_5050_long_w_preview_4__1/iter_10000.pt"
NUM_ENVS = 8


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


def attacker_builder(seed: int | None):
    return RandomSelectAttacker(
        seed=seed,
        choices=[
            DeceptiveRandomAttacker(
                seed=seed,
                min_safe_distance=3,
                max_commit_distance=1,
                stupidity=1,
                ignore_defender=False,
                stop_deception_after=32,
            ),
            DecisiveRandomNaiveAttacker(
                seed=seed,
                min_safe_distance=3,
                max_commit_distance=1,
                stupidity=1,
                ignore_defender=False,
            ),
        ],
    )


def make_env(idx: int):
    def thunk():
        env = ExitEnv(
            render_mode="rgb_array",
            random_map=False,
            att_def_distance_reward_coeff=0.1,
            att_exit_distance_reward_coeff=0.15,
            max_steps=64,
        )
        env = GymWrapper(env, attacker_builder=attacker_builder)
        env = LLMPreviewWrapper(env, preview_steps=4)
        env = PartialObservabilityWrapper(env)
        env = StripWrapper(env)
        env = DeterministicResetWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


envs = gym.vector.AsyncVectorEnv([make_env(idx) for idx in range(NUM_ENVS)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(agent, _) = load_agent_from_checkpoint(
    CHECKPOINT_PATH,
    observation_space_shape=envs.single_observation_space.shape,  # type: ignore
    action_space_shape=envs.single_action_space.n,  # type: ignore
)
agent = agent.to(device=device)

lstm_state: tuple[torch.Tensor, torch.Tensor] = (
    torch.zeros(agent.lstm.num_layers, NUM_ENVS, agent.lstm.hidden_size).to(
        device=device
    ),
    torch.zeros(agent.lstm.num_layers, NUM_ENVS, agent.lstm.hidden_size).to(
        device=device
    ),
)
observation, info = envs.reset(seed=1, options={"increment_seed_by": NUM_ENVS})  # type: ignore
observation = torch.Tensor(observation).to(device=device)
done = torch.zeros(NUM_ENVS).to(device=device)  # batch of size 1

writer = SummaryWriter(f"results/eval/llm/")
win_count = 0
lose_count = 0
total_count = 0
global_step = 0

start_time = time.time()
while True:
    # Forward with LSTM state + done flag
    with torch.no_grad():
        action, _, _, _, lstm_state = agent.get_action_and_value(
            observation, lstm_state, done
        )

    observation, reward, terminated, truncated, info = envs.step(action.cpu().numpy())

    done = np.logical_or(terminated, truncated)

    if "_episode" in info:
        _episode = info["_episode"]
        for i in range(NUM_ENVS):
            if _episode[i]:
                episodic_return = info["episode"]["r"][i]
                episodic_length = info["episode"]["l"][i]
                writer.add_scalar(
                    "charts/episodic_return",
                    episodic_return,
                    global_step,
                )
                writer.add_scalar(
                    "charts/episodic_length",
                    episodic_length,
                    global_step,
                )

                if reward[i] < 0:
                    lose_count += 1
                    writer.add_scalar("charts/lose_count", lose_count, global_step)
                else:
                    win_count += 1
                    writer.add_scalar("charts/win_count", win_count, global_step)
                total_count += 1
                writer.add_scalar(
                    "charts/win_rate", 100 * win_count / total_count, global_step
                )
                print(
                    f"{global_step} - WIN: {win_count}, LOSE: {lose_count}, WIN RATE: {win_count / total_count}"
                )

    observation = torch.Tensor(observation).to(device=device)
    done = torch.Tensor(done).to(device=device)
    global_step += NUM_ENVS
    print(f"SPS: {global_step / (time.time() - start_time)}")
