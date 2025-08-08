import os
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import ale_py
gym.register_envs(ale_py)

from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo

# 하이퍼파라미터
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = ["ALE/Breakout-v5", "ALE/Boxing-v5", "ALE/Enduro-v5", "ALE/Alien-v5", "ALE/Pong-v5"]
# ENV_NAME = ["ALE/Breakout-v5"]
SEED = 3 
MODEL_DIR = "./model"

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Q-Network 정의
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)

# 영상 기록 + 학습된 모델 추론
for env_name in ENV_NAME:
    print(f"Rendering {env_name}...")
    # env = gym.make(env_name, frameskip=1, render_mode="rgb_array", difficulty = 1, mode = 0, repeat_action_probability=0.0)
    env = gym.make(env_name, frameskip=1, render_mode="rgb_array", repeat_action_probability=0.0)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss = False, grayscale_obs=True)
    env = FrameStackObservation(env, stack_size=4, padding_type="zero")

    # 영상 저장 래퍼
    env = RecordVideo(
        env,
        episode_trigger=lambda _: True,
        video_folder=f"./video/video_{env_name.split('/')[1]}",
        name_prefix="video-"
    )

    # Q-Network 로드
    q_net = QNetwork(env).to(device)
    model_path = os.path.join(MODEL_DIR, f"{env_name.split('/')[-1]}_seed{SEED}", "dqn_latest.pth")
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()
    for episode in range(1):
        obs, _ = env.reset(seed=SEED + 5)
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4, 84, 84)
                q_values = q_net(obs_tensor)
                action = q_values.argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(total_reward, episode)

    env.close()
