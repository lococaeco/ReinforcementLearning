import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
import numpy as np
import os

import ContinuousGaussianPolicy

video_save_dir = "./video"
os.makedirs(video_save_dir, exist_ok=True)

os.environ['MUJOCO_GL'] = 'egl'
# os.environ['MUJOCO_GL'] = 'omesa'

# -------------------------------
# Config
# -------------------------------
HIDDEN_DIM = 64
SEEDS = [1]  # 필요한 시드 목록
BASE_MODEL_DIR = "./model/FIXED"  # 날짜 폴더
ENV_NAME = [
    "HalfCheetah-v5",
]

# ENV_NAME = [
#     "HalfCheetah-v5",
#     "Ant-v5",
#     "Hopper-v5",
#     "Humanoid-v5",
#     "HumanoidStandup-v5",
#     "InvertedDoublePendulum-v5",
#     "InvertedPendulum-v5",
#     "Pusher-v5",
#     "Reacher-v5",
#     "Swimmer-v5",
#     "Walker2d-v5"
# ]


# -------------------------------
# Policy Network
# -------------------------------

# class ContinuousGaussianPolicy(nn.Module):
#     def __init__(self, obs_dim, act_dim, action_low, action_high, hidden_dim=64):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#         )
#         self.mean_head = nn.Linear(hidden_dim, act_dim)
#         self.log_std_head = nn.Linear(hidden_dim, act_dim)

#         self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
#         self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))
    
#     def forward(self, obs):
#         x = self.net(obs)
#         mean = self.mean_head(x)
#         log_std = torch.tanh(self.log_std_head(x))
#         std = torch.exp(log_std)
#         return Normal(mean, std)

#     def _scale_action(self, norm_action):
#         return self.action_low + 0.5 * (norm_action + 1.0) * (self.action_high - self.action_low)

#     def _unscale_action(self, scaled_action):
#         return 2.0 * (scaled_action - self.action_low) / (self.action_high - self.action_low) - 1.0

#     def sample(self, obs):
#         with torch.no_grad():
#             dist = self.forward(obs)
#             raw_action = dist.sample()  # u ~ N(μ, σ)
#             norm_action = torch.tanh(raw_action)  # a = tanh(u)
#             action = self._scale_action(norm_action)
#         return action
        
#     def get_deterministic_action(self, obs):
#         with torch.no_grad():
#             dist = self.forward(obs)
#             norm_action = torch.tanh(dist.mean)
#             action = self._scale_action(norm_action)
#         return action

#     def log_prob(self, obs, action):
#         norm_action = self._unscale_action(action)
#         norm_action = torch.clamp(norm_action, -0.999999, 0.999999)
#         raw_action = torch.atanh(norm_action)
#         dist = self.forward(obs)
#         log_prob = dist.log_prob(raw_action).sum(dim=-1)
#         log_prob -= torch.log(1 - norm_action.pow(2) + 1e-6).sum(dim=-1)

#         return log_prob

# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate_and_record(env_name, model_path, device, seed):
    save_dir = os.path.join(video_save_dir, env_name.split('-')[0], f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Evaluating {env_name} | Seed: {seed} | Model: {model_path} ===")
    eval_env = gym.make(env_name, render_mode="rgb_array")

    record_env = gym.wrappers.RecordVideo(
        eval_env,
        video_folder=save_dir,
        name_prefix=f"trpo-{env_name.split('-')[0]}"
    )

    obs_dim = record_env.observation_space.shape[0]
    act_dim = record_env.action_space.shape[0]

    
    act_low = record_env.action_space.low
    act_high = record_env.action_space.high

    policy = ContinuousGaussianPolicy(obs_dim, act_dim, act_low, act_high).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    obs, _ = record_env.reset()
    done = False
    truncated = False
    total_reward = 0

    while not (done or truncated):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy.get_deterministic_action(obs_tensor)
            
        action = action.cpu().numpy()[0]

        next_obs, reward, done, truncated, _ = record_env.step(action)
        obs = next_obs
        total_reward += reward

    print(f"Total Reward: {total_reward:.2f}")
    record_env.close()


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for env_name in ENV_NAME:
        for seed in SEEDS:
            model_path = os.path.join(
                BASE_MODEL_DIR,
                env_name.split('-')[0],
                f"seed_{seed}",
                "policy.pth"
            )

            if not os.path.exists(model_path):
                print(f"⚠ Model not found: {model_path}")
                continue

            evaluate_and_record(env_name, model_path, device, seed)

    print("\n✅ All evaluations completed. Videos saved in:", video_save_dir)