import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributions.normal import Normal
import gymnasium as gym
import numpy as np
import os
import torch.nn.functional as F
import wandb

# 저장 경로
save_dir = "./model_halfcheetah_a3c"
os.makedirs(save_dir, exist_ok=True)

# A3C 설정
GAMMA = 0.99
LAMBDA = 0.95  # GAE lambda
LR = 1e-4
NUM_WORKERS = 4
MAX_EPISODES = 30000
ENV_NAME = "HalfCheetah-v5"

wandb.init(project="A3C-HalfCheetah")

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh()
        )
        self.mean_head = nn.Linear(64, act_dim)
        self.std_head = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        mean = self.mean_head(x)
        std = torch.clamp(F.softplus(self.std_head(x)) + 1e-5, min=1e-3, max=1.0)
        value = self.value_head(x).squeeze(-1)
        return mean, std, value


def compute_gae(rewards, values, next_value, dones, gamma, lam):
    values = values + [next_value] #1000 + 1짜리 list
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])
    return returns

def worker(rank, global_model, optimizer):
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    local_model = ActorCritic(obs_dim, act_dim)
    local_model.load_state_dict(global_model.state_dict()) #Global Model의 parameter을 복사해온다.

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        log_probs = [] #log(pi(a,s))
        values = []
        rewards = []
        dones = []
        entropies = []

        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            mean, std, value = local_model(state_tensor)
            
            dist = Normal(mean, std) # Actor Critic으로부터 얻은 평균과 분산으로 distribution을 만든다. [6] mean과 std가 각 action space의 size만큼 있기 때문에
            
            action = dist.sample() # Action은 위의 분포에서 6가지 값을 가져온다.

            log_prob = dist.log_prob(action).sum()
            entropy = dist.entropy().sum()

            next_state, reward, terminated, truncated, infos = env.step(action.detach().numpy())
            done = terminated or truncated

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)
            entropies.append(entropy)

            total_reward += reward
            state = next_state

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        with torch.no_grad():
            _, _, next_value = local_model(next_state_tensor)

        returns = compute_gae(rewards, values, next_value, dones, GAMMA, LAMBDA) #이전 1000번간의 reward, value, next value, dones를 넣어준다.

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        advantages = returns - values.detach()

        policy_loss = -(log_probs * advantages).sum()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -0.01 * entropies.sum()
        loss = policy_loss + 0.5 * value_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            global_param._grad = local_param.grad
        optimizer.step()
        local_model.load_state_dict(global_model.state_dict())

        wandb.log({
            f"worker_{rank}/episode": episode,
            f"worker_{rank}/reward": total_reward,
            f"worker_{rank}/loss": loss.item(),
            f"worker_{rank}/policy_loss": policy_loss.item(),
            f"worker_{rank}/value_loss": value_loss.item(),
            f"worker_{rank}/entropy": entropies.mean().item()
        })

        if episode % 10 == 0:
            print(f"[Worker {rank}] Episode {episode} | Total Reward: {total_reward:.2f}")

def train():
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()

    global_model = ActorCritic(obs_dim, act_dim)
    global_model.share_memory()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=LR)

    processes = []
    for rank in range(NUM_WORKERS):
        p = mp.Process(target=worker, args=(rank, global_model, optimizer))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    torch.save(global_model.state_dict(), os.path.join(save_dir, "a3c_actor_critic.pt"))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    train()