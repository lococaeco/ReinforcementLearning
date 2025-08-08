import os
import time
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
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    RecordVideo,
    ResizeObservation,
    GrayScaleObservation,
    FrameStack,
)


def make_env(env_id, seed, idx, capture_video):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if capture_video else None)
        if capture_video and idx == 0:
            env = RecordVideo(
                env,
                video_folder=f"videos_eval/{env_id}",
                episode_trigger=lambda e: True,
            )
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


class C51Network(nn.Module):
    def __init__(self, env, atom_size, support):
        super().__init__()
        self.atom_size = atom_size
        self.support = support
        self.action_size = env.action_space.n

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


if __name__ == "__main__":
    model_file = "./model/Breakout_c51_4000.pth"
    seed = 0
    env_id = "BreakoutNoFrameskip-v4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capture_video = True
    episodes_to_eval = 5
    V_min = -10
    V_max = 10
    atom_size = 51
    support = torch.linspace(V_min, V_max, atom_size).to(device)
    delta_z = (V_max - V_min) / (atom_size - 1)

    # 영상 저장 폴더 생성
    os.makedirs(f"videos_eval/{env_id}", exist_ok=True)

    # 환경 및 네트워크 초기화
    env = make_env(env_id, seed, 0, capture_video=capture_video)()
    q_network = C51Network(env, atom_size, support).to(device)
    q_network.load_state_dict(torch.load(model_file, map_location=device))
    q_network.eval()

    episode_returns = []

    for episode in range(episodes_to_eval):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                # FrameStack: (C, H, W) 형태로 들어옴
                state_np = np.array(state)  # shape: (4, 84, 84)
                state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 4, 84, 84)
                q_values = q_network.q_values(state_tensor)
                action = torch.argmax(q_values, dim=1).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        episode_returns.append(total_reward)
        print(f"🎮 Episode {episode + 1} Return: {total_reward}")

    env.close()
    time.sleep(1)  # 영상 저장 대기
    print(f"✅ Evaluation done. Mean return: {np.mean(episode_returns):.2f}")
