import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributions.normal import Normal
import gymnasium as gym
import numpy as np
import os
import torch.nn.functional as F
import wandb

save_dir = "./model"
os.makedirs(save_dir, exist_ok=True)


ENV_NAME = ["HalfCheetah-v5"]

# ENV_NAME = ["HalfCheetah-v5",
#             "Ant-v5",
#             "Hopper-v5",
#             "Humanoid-v5",
#             "HumanoidStandup-v5",
#             "InvertedDoublePendulum-v5",
#             "InvertedPendulum-v5",
#             "Pusher-v5",
#             "Reacher-v5",
#             "Swimmer-v5",
#             "Walker2d-v5"]


class ContinuousPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # 학습 가능한 std

    def forward(self, obs):
        x = self.net(obs)
        action_mean = self.mean_head(x)
        action_std = torch.exp(self.log_std)
        return Normal(action_mean, action_std)  # Normal 분포 반환

    def act(self, obs):
        with torch.no_grad():
            dist = self.forward(obs)
            action = dist.sample()
            return action.clamp(-1.0, 1.0), dist.log_prob(action)

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)



def calc_gae(rewards, values, dones, gamma, lamda):
    advantages = []
    gae = 0
    next_value = 0
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        gae = delta + gamma * lamda * (1 - dones[step]) * gae
        advantages.insert(0, gae) #왜 insert를 사용해야 되지.
        next_value = values[step]
        
    return torch.tensor(advantages, dtype = torch.float32)



def process_trajectory(trajectory, value_net, gamma=0.99, lamda=0.97):
    states, actions, rewards, dones = zip(*trajectory)

    states = torch.tensor(np.stack(states), dtype=torch.float32)
    actions = torch.tensor(np.stack(actions), dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    with torch.no_grad():
        values = value_net(states)

    advantages = calc_gae(rewards, values, dones, gamma, lamda)
    returns = advantages + values

    return {
        "obs": states,
        "acts": actions,
        "advs": advantages,
        "returns": returns
    }
    
    
def surrogate_loss(policy, obs, acts, advs, old_log_probs):
    dist = policy(obs)
    new_log_probs = dist.log_prob(acts).sum(axis=-1)
    ratio = torch.exp(new_log_probs - old_log_probs)
    return - (ratio * advs).mean()

def conjugate_gradient():
    return 0


def line_search():
    return 0

def mean_kl_divergence():
    return 0


def train():
    env = gym.make(ENV_NAME[0])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = ContinuousPolicy(obs_dim, act_dim)
    value_net = ValueNetwork(obs_dim)

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)

    trajectory = []
    total_reward = 0
    for step in range(1000):
        action, log_prob = policy.act(obs)
        action_np = action.numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        trajectory.append((obs.numpy(), action_np, reward, done))
        total_reward += reward

        if done:
            break
        obs = torch.tensor(next_obs, dtype=torch.float32)

    print(f"Trajectory length: {len(trajectory)}, Total reward: {total_reward:.2f}")

    #advantage, return 계산
    batch = process_trajectory(trajectory, value_net)
    print(f"Advantage mean: {batch['advs'].mean():.4f}, Return mean: {batch['returns'].mean():.4f}")

    # Advantage, Return 계산 완료 후:
    obs_batch = batch["obs"]
    acts_batch = batch["acts"]
    advs_batch = batch["advs"]
    returns_batch = batch["returns"]

    with torch.no_grad():
        dist = policy(obs_batch) 
        old_log_probs = dist.log_prob(acts_batch).sum(-1)

    # Surrogate loss 계산
    loss = surrogate_loss(policy, obs_batch, acts_batch, advs_batch, old_log_probs)
    print(f"Surrogate loss: {loss.item():.4f}")
        


if __name__ == "__main__":
    train()
    
    
    
    

    
    
    
    
    
    
    
    
    
# 우리는 Surrogate Loss를 최대화시키면서, state visitation에 의해 importance sampling 된 분포 아래에서 
# policy 간의 KL-Divergence가 작은 new policy를 찾아야됨.

# Monte Carlo 방법에서는 1, 2, 3 방법을 통해 기존 이론을 실험적으로 변화를 주었다.


# for iteration in range(max_iters):
#     # 1. Trajectory rollout 
#     obs, acts, rewards, values, dones = collect_trajectory()

#     # 2. Compute advantages
#     advs = compute_gae(rewards, values, dones)

#     # 3. Backup old policy
#     old_policy.load_state_dict(policy.state_dict())

#     # 4. Compute surrogate loss gradient
#     loss = surrogate_loss(policy, old_policy, obs, acts, advs)
#     grads = torch.autograd.grad(loss, policy.parameters())
#     loss_grad = torch.cat([g.view(-1) for g in grads]).data

#     # 5. Compute natural gradient step
#     step_dir = conjugate_gradient(
#         lambda v: fisher_vector_product(policy, obs, v), -loss_grad
#     )
#     max_step = np.sqrt(2 * max_kl / (step_dir @ fisher_vector_product(policy, obs, step_dir)))
#     full_step = max_step * step_dir

#     # 6. Line search to satisfy KL constraint
#     success = linesearch(policy, old_policy, full_step, loss_grad @ step_dir, obs, acts, advs)