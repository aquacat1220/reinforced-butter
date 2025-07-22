# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from typing import Any
from exit import (
    ExitEnv,
    GymWrapper,
    StupidAttacker,
    PursueAttacker,
    DeceptiveAttacker,
    EvadeAttacker,
    DecisiveNaiveAttacker,
    StripWrapper,
    PartialObservabilityWrapper,
    FrameStackWrapper,
    DeterministicResetWrapper,
)


@dataclass
class Args:
    exp_name: str = "exit_env_deceptive_attacker_noignore"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ppo_exit"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    capture_interval: int = 2000
    """capture video once every `capture_interval` episodes"""
    checkpoint: bool = True
    """whether to store checkpoints"""
    stupidity: int = 2
    """how often `StupidPursueGhost`s move"""
    # preview_steps: int = 2
    # """number of steps to preview into the future"""
    distance_reward_coeff: float = 0.1
    """whether to use distance based rewards"""
    max_steps: int = 256
    min_safe_distance: int = 3
    """the minimum distance the attacker considers to be safe to get closer to the defender"""
    max_commit_distance: int = 1
    """the maximum distance the attacker will commit to its target, ignoring the defender"""
    stop_deception_after: int = 32
    """attacker will stop deception after `stop_deception_after` steps"""
    history_length: int = 2
    """length of the stacked history"""
    random_map: bool = True
    """whether to randomize map layout"""
    ignore_defender: bool = False
    """whether to ignore defender while attacker pathfinding"""

    # Algorithm specific arguments
    # env_id: str = "CartPole-v1"
    env_id: str = "exit_env_v0"  # Ignore command line arguments and use `PacmanEnv`.
    """the id of the environment"""
    total_timesteps: int = 80000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.02
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(
    env_id: str,
    idx: int,
    capture_video: bool,
    capture_interval: int,
    run_name: str,
    stupidity: int,
    # preview_steps: int,
    distance_reward_coeff: float,
    max_steps: int,
    min_safe_distance: int,
    max_commit_distance: int,
    stop_deception_after: int,
    history_length: int,
    random_map: bool,
    ignore_defender: bool,
):
    def thunk():
        # ghost_builder = lambda _: StupidPursueGhost(stupidity)
        def naive_attacker_builder() -> DecisiveNaiveAttacker:
            return DecisiveNaiveAttacker(
                min_safe_distance=min_safe_distance,
                max_commit_distance=max_commit_distance,
                stupidity=stupidity,
                ignore_defender=ignore_defender,
            )

        def deceptive_attacker_builder() -> DeceptiveAttacker:
            return DeceptiveAttacker(
                min_safe_distance=min_safe_distance,
                max_commit_distance=max_commit_distance,
                stupidity=stupidity,
                ignore_defender=ignore_defender,
                stop_deception_after=stop_deception_after,
            )

        if capture_video and idx == 0:
            # env = gym.make(env_id, render_mode="rgb_array")
            env = ExitEnv(
                render_mode="rgb_array",
                random_map=random_map,
                att_def_distance_reward_coeff=distance_reward_coeff,
                max_steps=max_steps,
            )
            env = GymWrapper(env, attacker_builder=deceptive_attacker_builder)
            env = PartialObservabilityWrapper(env)
            env = FrameStackWrapper(env, history_length=history_length)
            env = StripWrapper(env)
            env = DeterministicResetWrapper(env)
            env = gym.wrappers.RecordVideo(
                env,
                f"results/videos/{run_name}",
                episode_trigger=lambda id: id % capture_interval == 0,
                fps=1,
            )
        else:
            # env = gym.make(env_id)
            env = ExitEnv(
                render_mode="rgb_array",
                random_map=random_map,
                att_def_distance_reward_coeff=distance_reward_coeff,
                max_steps=max_steps,
            )
            if idx % 2 == 0:
                env = GymWrapper(env, attacker_builder=deceptive_attacker_builder)
            else:
                env = GymWrapper(env, attacker_builder=naive_attacker_builder)
            env = PartialObservabilityWrapper(env)
            env = FrameStackWrapper(env, history_length=history_length)
            env = StripWrapper(env)
            env = DeterministicResetWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(
        self,
        observation_space_shape: tuple[int, int, int],
        action_space_shape: int,
    ):
        super().__init__()

        (C, H, W) = observation_space_shape
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(C, 16, 3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, 3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * (H - 6) * (W - 6), 256)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(256, action_space_shape), std=0.01)
        # self.actor = nn.Sequential(
        #     layer_init(nn.Linear(256, 64), std=0.01),
        #     layer_init(nn.Linear(64, action_space_shape), std=0.01),
        # )
        self.critic = layer_init(nn.Linear(256, 1), std=1)
        # self.critic = nn.Sequential(
        #     layer_init(nn.Linear(256, 64), std=1), layer_init(nn.Linear(64, 1), std=1)
        # )

    def get_value(self, x):
        x = self.network(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.network(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            mode="offline",
            dir="results/wandb",
        )
    writer = SummaryWriter(f"results/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    if args.checkpoint:
        os.makedirs(f"results/checkpoints/{run_name}", exist_ok=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=args.env_id,
                idx=i,
                capture_video=args.capture_video,
                capture_interval=args.capture_interval // args.num_envs,
                run_name=run_name,
                stupidity=args.stupidity,
                # preview_steps=args.preview_steps,
                distance_reward_coeff=args.distance_reward_coeff,
                max_steps=args.max_steps,
                min_safe_distance=args.min_safe_distance,
                max_commit_distance=args.max_commit_distance,
                stop_deception_after=args.stop_deception_after,
                history_length=args.history_length,
                random_map=args.random_map,
                ignore_defender=args.ignore_defender,
            )
            for i in range(args.num_envs)
        ],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(
        observation_space_shape=envs.single_observation_space.shape,
        action_space_shape=envs.single_action_space.n,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(
        seed=args.seed, options={"increment_seed_by": args.num_envs}
    )
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            if "_episode" in infos:
                _episode = infos["_episode"]
                for i in range(len(_episode)):
                    if _episode[i]:
                        print(
                            f"global_step={global_step}, episodic_return={infos['episode']['r'][i]}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return",
                            infos["episode"]["r"][i],
                            global_step,
                        )
                        writer.add_scalar(
                            "charts/episodic_length",
                            infos["episode"]["l"][i],
                            global_step,
                        )

            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if info and "episode" in info:
            #             print(
            #                 f"global_step={global_step}, episodic_return={info['episode']['r']}"
            #             )
            #             writer.add_scalar(
            #                 "charts/episodic_return", info["episode"]["r"], global_step
            #             )
            #             writer.add_scalar(
            #                 "charts/episodic_length", info["episode"]["l"], global_step
            #             )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        if args.checkpoint and iteration % 500 == 0:
            checkpoint: dict[str, Any] = {
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iteration": iteration,
                "global_step": global_step,
                "args": vars(args),
            }
            torch.save(
                checkpoint, f"results/checkpoints/{run_name}/iter_{iteration}.pt"
            )

    envs.close()
    writer.close()
