import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import imageio
import os
from torch.distributions.normal import Normal
import torch.nn.functional as F

ENV_NAME = "HalfCheetah-v5"
MODEL_DIR = "./model_halfcheetah_SAC"
VIDEO_DIR = os.path.join(MODEL_DIR, "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)

LOG_STD_MIN = -5
LOG_STD_MAX = 2

class Actor(nn.Module):
    def __init__(self, env, device):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        hidden_dim = 256

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, act_dim)
        self.fc_logstd = nn.Linear(hidden_dim, act_dim)

        action_high = torch.tensor(env.action_space.high, dtype=torch.float32)
        action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
        self.action_scale = ((action_high - action_low) / 2.0).to(device)
        self.action_bias = ((action_high + action_low) / 2.0).to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

def evaluate(seed, device):
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    actor = Actor(env, device).to(device)
    actor.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"actor_seed{seed}.pt")))
    actor.eval()

    state, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0
    k = 0
    for i in range(1000):
        frame = env.render()
        frames.append(frame)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        # if i < 100:
        #     # 앞다리는 정지, 뒷다리는 반복적으로 강하게 움직이기
        #     back_leg_wave = np.sin(i / 5.0 * np.pi)  # 파형 기반
        #     actions = np.array([[
        #         back_leg_wave,         # bthigh
        #         back_leg_wave,        # bshin
        #         back_leg_wave,   # bfoot
        #         back_leg_wave,                   # fthigh
        #         back_leg_wave,                   # fshin
        #         back_leg_wave                    # ffoot
        #     ]])
        # else:
        with torch.no_grad():
            actions, _, _ = actor.get_action(state_tensor)
            actions = actions.detach().cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(actions[0])
        total_reward += reward
        state = next_state

        if terminated or truncated:
            break

    env.close()

    video_path = os.path.join(VIDEO_DIR, f"eval_seed{seed}.mp4")
    imageio.mimsave(video_path, frames, fps=30)
    print(f"[Seed {seed}] Evaluation return: {total_reward:.2f} | Video saved to {video_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # for seed in [1, 2, 3, 5, 8]:
    for seed in [1]:
        evaluate(seed, device)
