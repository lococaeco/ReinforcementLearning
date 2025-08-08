import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import gymnasium as gym
import numpy as np
import os
import wandb
import random
import time

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_len=100000, device='cpu'):
        self.states   = np.zeros((max_len, *state_dim), dtype=np.float32)
        self.actions  = np.zeros((max_len, *action_dim), dtype=np.float32)
        self.rewards  = np.zeros((max_len, 1), dtype=np.float32)
        self.nstates  = np.zeros((max_len, *state_dim), dtype=np.float32)
        self.dones = np.zeros((max_len, 1), dtype=np.float32)

        self.max_len = max_len
        self.device = device

        self.current_idx = 0
        self.current_size = 0

    def add(self, state, action, reward, next_state, done):
        idx = self.current_idx % self.max_len

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.nstates[idx] = next_state
        self.dones[idx] = done

        self.current_idx += 1
        self.current_size = min(self.current_size + 1, self.max_len)

    def sample(self, batch_size, as_tensor=True):
        indices = np.random.choice(self.current_size, size=batch_size, replace=False)

        batch = (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.nstates[indices],
            self.dones[indices],
        )

        if as_tensor:
            return tuple(torch.tensor(b, device=self.device) for b in batch)
        return batch

    def __len__(self):
        return self.current_size

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        input_dim = obs_dim + act_dim
        hidden_dim = 256

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

LOG_STD_MIN = -5
LOG_STD_MAX = 2

class Actor(nn.Module):
    def __init__(self, env):
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

save_dir = "./model_halfcheetah_SAC"
os.makedirs(save_dir, exist_ok=True)

GAMMA = 0.99
LR = 1e-4
MAX_EPISODES = 30000
ENV_NAME = "HalfCheetah-v5"

if __name__ == "__main__":
    seeds = [1, 2, 3, 5, 8]
    for seed in seeds:
        run = wandb.init(project="SAC-HalfCheetah", config={
            "env": ENV_NAME,
            "gamma": GAMMA,
            "lr": LR,
            "batch_size": 256,
            "alpha": 0.2,
            "tau": 0.005,
            "seed": seed
        })

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        env = gym.make(ENV_NAME)
        SoftActor = Actor(env).to(device)

        Qfunction01 = SoftQNetwork(env).to(device)
        Qfunction02 = SoftQNetwork(env).to(device)
        
        TargetQfunction01 = SoftQNetwork(env).to(device)
        TargetQfunction02 = SoftQNetwork(env).to(device)

        TargetQfunction01.load_state_dict(Qfunction01.state_dict())
        TargetQfunction02.load_state_dict(Qfunction02.state_dict())

        QfuncOptim = optim.Adam(list(Qfunction01.parameters()) + list(Qfunction02.parameters()), lr=1e-3)
        SoftActorOptim = optim.Adam(SoftActor.parameters(), lr=3e-4)

        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=1e-3)

        memory = ReplayBuffer(env.observation_space.shape, env.action_space.shape, device=device)

        state, _ = env.reset(seed=seed)

        episode_return = 0
        # 넘어졌을 때 유연하게 다시 일어서는 것은 보상함수를 정의해주지 않음
        # 애초에 넘어지는 환경은 너무 OOD 환경인데
        # 
        for current_step in range(1000000):
            if current_step < 10000:
                actions = np.array([env.action_space.sample()])[0]
            else:
                actions, _, _ = SoftActor.get_action(torch.Tensor(state).to(device))
                actions = actions.detach().cpu().numpy()

            nstates, rewards, terminated, truncated, infos = env.step(actions)

            done = terminated or truncated

            memory.add(state, actions, rewards, nstates, done)
            episode_return += rewards
            
            state = nstates

            if done:
                wandb.log({"episode_return": episode_return}, step=current_step)
                print(f"episode_return: {episode_return}")
                state, _ = env.reset(seed=seed)
                episode_return = 0

            if current_step >= 10000:
                data = memory.sample(256)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = SoftActor.get_action(data[3])
                    qf1_next_target = TargetQfunction01(data[3], next_state_actions)
                    qf2_next_target = TargetQfunction02(data[3], next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data[2].flatten() + GAMMA * (1 - data[4].flatten()) * min_qf_next_target.view(-1)

                qf1_a_value = Qfunction01(data[0], data[1]).view(-1)
                qf2_a_value = Qfunction02(data[0], data[1]).view(-1)
                qf1_loss = F.mse_loss(qf1_a_value, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_value, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                QfuncOptim.zero_grad()
                qf_loss.backward()
                QfuncOptim.step()

                wandb.log({
                    "qf1_loss": qf1_loss.item(),
                    "qf2_loss": qf2_loss.item(),
                    "qf_loss": qf_loss.item()
                }, step=current_step)

                if current_step % 2 == 0:
                    for _ in range(2):
                        pi, log_pi, _ = SoftActor.get_action(data[0])
                        qf1_pi = Qfunction01(data[0], pi)
                        qf2_pi = Qfunction02(data[0], pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        SoftActorOptim.zero_grad()
                        actor_loss.backward()
                        SoftActorOptim.step()

                        wandb.log({
                            "actor_loss": actor_loss.item(),
                            "entropy": -log_pi.mean().item(),
                            "alpha": alpha,
                        }, step=current_step)
                    
                    _, log_pi, _ = SoftActor.get_action(data[0])
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

                if current_step % 1 == 0:
                    for param, target_param in zip(Qfunction01.parameters(), TargetQfunction01.parameters()):
                        target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)
                    for param, target_param in zip(Qfunction02.parameters(), TargetQfunction02.parameters()):
                        target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

        torch.save(SoftActor.state_dict(), os.path.join(save_dir, f"actor_seed{seed}.pt"))
        torch.save(Qfunction01.state_dict(), os.path.join(save_dir, f"q1_seed{seed}.pt"))
        torch.save(Qfunction02.state_dict(), os.path.join(save_dir, f"q2_seed{seed}.pt"))

        run.finish()
