import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
import wandb
import os

# 저장 디렉토리
save_dir = "./model_halfcheetah"
os.makedirs(save_dir, exist_ok=True)

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.mean_head = nn.Linear(64, act_dim)
        self.std_head = nn.Linear(64, act_dim)

    def forward(self, x):
        x = self.shared(x)
        mean = self.mean_head(x)
        std = torch.log(1 + torch.exp(self.std_head(x)))  # Softplus
        return mean, std

class REINFORCE:
    def __init__(self, obs_dim, act_dim, lr=1e-4, gamma=0.99):
        self.policy = PolicyNetwork(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.loss = 0

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        mean, std = self.policy(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.detach().numpy(), log_prob

    def update(self, rewards, log_probs):
        discounted_returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted_returns.insert(0, G)
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float32)
        loss = 0
        # for log_prob, G in zip(log_probs, discounted_returns):
        #     loss -= log_prob * G
        baseline = discounted_returns.mean()
        for log_prob, G in zip(log_probs, discounted_returns):
            loss -= log_prob * (G - baseline)    

        self.loss = loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def train_reinforce(seed):
    # 시드 설정
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make("HalfCheetah-v5")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = REINFORCE(obs_dim, act_dim)

    wandb.init(
        project="REINFORCE-HalfCheetah",
        name=f"REINFORCE-seed{seed}-baseline",
        group="REINFORCE-HalfCheetah",
        config={
            "env": "HalfCheetah-v5",
            "lr": 1e-4,
            "gamma": 0.99,
            "episodes": 100000,
            "hidden_units": 64,
            "seed": seed
        },
        save_code=True
    )

    num_episodes = 100000
    score_list = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        total_reward = 0

        done = False
        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward
            state = next_state

        agent.update(rewards, log_probs)
        score_list.append(total_reward)

        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(score_list[-10:])
            print(f"Seed {seed} | Episode {ep+1} | Average Reward: {avg_reward:.2f}")
            wandb.log({"episode": ep+1, "avg_reward": avg_reward})
            wandb.log({"episode": ep+1, "loss": agent.loss})

    torch.save(agent.policy.state_dict(), os.path.join(save_dir, f"reinforce_policy_seed{seed}.pt"))

    wandb.finish()

if __name__ == "__main__":
    for seed in [1, 2, 3, 5, 8]:
        train_reinforce(seed)
