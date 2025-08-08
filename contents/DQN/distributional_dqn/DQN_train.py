import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def make_env(env_id, seed, idx, capture_video):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        return env
    return thunk


class C51Network(nn.Module):
    def __init__(self, env, atom_size, support):
        super().__init__()
        self.atom_size = atom_size
        self.support = support
        self.action_size = env.single_action_space.n

        self.feature_layer = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        self.advantage = nn.Linear(512, self.action_size * atom_size)

    def forward(self, x):
        x = self.feature_layer(x / 255.0)
        dist = self.advantage(x).view(-1, self.action_size, self.atom_size)
        dist = F.softmax(dist, dim=2)
        return dist

    def q_values(self, x):
        dist = self.forward(x)
        q = torch.sum(dist * self.support, dim=2)
        return q


class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, device="cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.observations = np.zeros((buffer_size, *state_dim), dtype=np.uint8)
        self.next_observations = np.zeros((buffer_size, *state_dim), dtype=np.uint8)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size):
        total = self.buffer_size if self.full else self.pos
        indices = np.random.choice(total, batch_size, replace=False)
        obs = torch.tensor(self.observations[indices], dtype=torch.uint8, device=self.device)
        next_obs = torch.tensor(self.next_observations[indices], dtype=torch.uint8, device=self.device)
        actions = torch.tensor(self.actions[indices], dtype=torch.int64, device=self.device)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device).unsqueeze(1)
        return obs, actions, next_obs, rewards, dones


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    model_path = "./model"
    os.makedirs(model_path, exist_ok=True)
    seed = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = "BreakoutNoFrameskip-v4"
    envs = gym.vector.SyncVectorEnv([make_env(env_name, seed, 0, False)])

    # Config
    V_min = -10
    V_max = 10
    atom_size = 51
    support = torch.linspace(V_min, V_max, atom_size).to(device)
    delta_z = (V_max - V_min) / (atom_size - 1)

    learning_rate = 1e-4
    buffer_size = 100_000
    total_timesteps = 1_000_0000
    start_e = 1.0
    end_e = 0.1
    exploration_fraction = 0.1
    learning_starts = 80_000
    train_frequency = 4
    batch_size = 32
    gamma = 0.99
    target_network_frequency = 1000
    tau = 1.0

    # wandb 설정
    use_wandb = True
    if use_wandb:
        import wandb
        wandb.init(
            project="dqn-breakout",
            config={
                "env_name": env_name,
                "total_timesteps": total_timesteps,
                "learning_rate": learning_rate,
                "buffer_size": buffer_size,
                "batch_size": batch_size,
                "gamma": gamma,
                "start_e": start_e,
                "end_e": end_e,
                "exploration_fraction": exploration_fraction,
                "train_frequency": train_frequency,
                "learning_starts": learning_starts,
                "target_network_frequency": target_network_frequency,
                "tau": tau,
                "seed": seed,
                "V_min": V_min,
                "V_max": V_max,
                "atom_size": atom_size,
            },
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    q_network = C51Network(envs, atom_size, support).to(device)
    target_network = C51Network(envs, atom_size, support).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    obs_shape = envs.single_observation_space.shape
    action_shape = (1,)
    rb = ReplayBuffer(buffer_size, obs_shape, action_shape[0], device)

    state, _ = envs.reset(seed=seed)
    episode = 0

    for global_step in range(total_timesteps):
        epsilon = linear_schedule(start_e, end_e, exploration_fraction * total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.uint8, device=device)
                q_vals = q_network.q_values(state_tensor)
                actions = torch.argmax(q_vals, dim=1).cpu().numpy()

        next_state, rewards, terminations, truncations, infos = envs.step(actions)

        real_next_state = next_state.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_state[idx] = infos["final_observation"][idx]

        rb.add(state, real_next_state, actions, rewards, terminations)
        state = next_state

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode += 1
                    print(f"step: {global_step}, episode: {episode}, reward: {info['episode']['r']}")
                    if use_wandb:
                        wandb.log({
                            "episode": episode,
                            "episodic_return": info["episode"]["r"],
                            "episodic_length": info["episode"]["l"],
                            "epsilon": epsilon,
                            "global_step": global_step,
                        })

        if global_step > learning_starts and global_step % train_frequency == 0:
            obs, actions, next_obs, rewards, dones = rb.sample(batch_size)

            with torch.no_grad():
                next_dist = target_network(next_obs)
                next_q = torch.sum(next_dist * support, dim=2)
                next_actions = torch.argmax(next_q, dim=1)
                next_dist = next_dist[range(batch_size), next_actions]

                t_z = rewards + (1 - dones) * gamma * support.unsqueeze(0)
                t_z = t_z.clamp(V_min, V_max)
                b = (t_z - V_min) / delta_z
                l = b.floor().long()
                u = b.ceil().long()

                proj_dist = torch.zeros_like(next_dist)
                for i in range(batch_size):
                    for j in range(atom_size):
                        l_idx, u_idx = l[i, j], u[i, j]
                        proj_dist[i, l_idx] += next_dist[i, j] * (u[i, j] - b[i, j])
                        proj_dist[i, u_idx] += next_dist[i, j] * (b[i, j] - l[i, j])

            dist = q_network(obs)
            log_p = torch.log(dist[range(batch_size), actions.squeeze()])
            loss = -(proj_dist * log_p).sum(1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "global_step": global_step
                })

        if global_step % target_network_frequency == 0:
            for t_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                t_param.data.copy_(tau * q_param.data + (1 - tau) * t_param.data)

        if episode % 1000 == 0 and episode > 0:
            model_file = os.path.join(model_path, f"Breakout_c51_{episode}.pth")
            torch.save(q_network.state_dict(), model_file)
            print(f"✅ Model saved at episode {episode} → {model_file}")

    envs.close()
