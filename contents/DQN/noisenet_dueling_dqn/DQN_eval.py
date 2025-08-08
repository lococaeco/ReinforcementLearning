import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import math

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, ResizeObservation, GrayScaleObservation, FrameStack


def make_env(env_id, seed, idx, capture_video):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if capture_video else None)
        if capture_video and idx == 0:
            env = RecordVideo(env, video_folder=f"videos_eval/{env_id}", episode_trigger=lambda e: True)
        env = RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = ResizeObservation(env, (84, 84))
        env = GrayScaleObservation(env)
        env = FrameStack(env, 4)
        env.action_space.seed(seed)
        return env
    return thunk


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()



class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.flattened_size = 3136

        self.value_stream = nn.Sequential(
            NoisyLinear(self.flattened_size, 512),
            nn.ReLU(),
            NoisyLinear(512, 1),
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.flattened_size, 512),
            nn.ReLU(),
            NoisyLinear(512, env.action_space.n),
        )

    def forward(self, x):
        x = x / 255.0
        features = self.feature_extractor(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

if __name__ == "__main__":
    model_file = "./model/Breakout_dqn_classic_6000.pth"
    seed = 42
    env_id = "BreakoutNoFrameskip-v4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capture_video = True
    episodes_to_eval = 5
    

    os.makedirs("videos_eval", exist_ok=True)

    env = make_env(env_id, seed, 0, capture_video=capture_video)()
    q_network = QNetwork(env).to(device)
    q_network.load_state_dict(torch.load(model_file, map_location=device))
    q_network.eval()

    episode_returns = []

    for episode in range(episodes_to_eval):
        state, _ = env.reset(seed=seed)
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
                q_values = q_network(state_tensor)
                action = torch.argmax(q_values, dim=1).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        episode_returns.append(total_reward)
        print(f"🎮 Episode {episode + 1} Return: {total_reward}")

    env.close()
    print(f"✅ Evaluation done. Mean return: {np.mean(episode_returns):.2f}")
