import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import gymnasium as gym
import numpy as np
import os
import random
import wandb

class ReplayBuffer:
    def __init__(self, state_dim, action_dim , max_len=int(1e6), device='cpu'):
        self.states   = np.zeros((max_len, *state_dim), dtype=np.float32)
        self.actions  = np.zeros((max_len, *action_dim), dtype=np.float32)
        self.rewards  = np.zeros((max_len, 1), dtype=np.float32)
        self.nstates  = np.zeros((max_len, *state_dim), dtype=np.float32)
        self.dones    = np.zeros((max_len, 1), dtype=np.float32)
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

class ValueNetwork(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.model(state)

class QNetwork(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        self.model = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)




LOG_STD_MIN = -5
LOG_STD_MAX = 2

class Actor(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, act_dim)
        self.fc_logstd = nn.Linear(hidden_dim, act_dim)

        action_high = torch.tensor(env.action_space.high, dtype=torch.float32)
        action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        y = torch.tanh(z)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(z) - torch.log(self.action_scale * (1 - y.pow(2)) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def get_deterministic_action(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.action_scale + self.action_bias

# === 학습 설정 ===
save_dir = "./model_halfcheetah_SAC"
os.makedirs(save_dir, exist_ok=True)

ENV_NAME = "HalfCheetah-v5"
GAMMA = 0.99
LR = 3e-4
MAX_EPISODES = 1000
MAX_EPISODE_LENGTH = 1000
TAU = 0.005
ALPHA = 0.2
BATCH_SIZE = 256

seeds = [1, 2, 3, 5, 8]

# === 학습 ===
for seed in seeds:
    wandb.init(project="SAC-HalfCheetah-ep1000", name=f"seed-{seed}", config={
        "env": ENV_NAME,
        "gamma": GAMMA,
        "lr": LR,
        "tau": TAU,
        "alpha": ALPHA,
        "batch_size": BATCH_SIZE,
        "seed": seed,
        "episodes": MAX_EPISODES
    })

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    env = gym.make(ENV_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ValFunc = ValueNetwork(env).to(device)
    TargValFunc = ValueNetwork(env).to(device)
    TargValFunc.load_state_dict(ValFunc.state_dict())

    QFunc_1 = QNetwork(env).to(device)
    QFunc_2 = QNetwork(env).to(device)
    actor = Actor(env).to(device)

    ValOptim = optim.Adam(ValFunc.parameters(), lr=LR)
    QFuncOptim = optim.Adam(list(QFunc_1.parameters()) + list(QFunc_2.parameters()), lr=LR)
    ActorOptim = optim.Adam(actor.parameters(), lr=LR)

    memory = ReplayBuffer(env.observation_space.shape, env.action_space.shape, device=device)

    def soft_update(source, target, tau):
        for src, tgt in zip(source.parameters(), target.parameters()):
            tgt.data.copy_(tau * src.data + (1.0 - tau) * tgt.data)

    state, _ = env.reset()
    episode_reward = 0

    for episode in range(MAX_EPISODES):
        Val_loss = torch.tensor(0.0)
        q_loss = torch.tensor(0.0)
        actor_loss = torch.tensor(0.0)
        for t in range(MAX_EPISODE_LENGTH):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = actor.get_action(state_tensor)
            action_np = action.cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            memory.add(state, action_np, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(memory) > 1000:
                states, actions, rewards, nstates, dones = memory.sample(BATCH_SIZE)

                with torch.no_grad():
                    next_action, next_logprob, _ = actor.get_action(nstates)
                    min_q_next = torch.min(QFunc_1(nstates, next_action), QFunc_2(nstates, next_action))
                    TargVal = min_q_next - ALPHA * next_logprob

                Val_pred = ValFunc(states)
                Val_loss = F.mse_loss(Val_pred, TargVal)
                ValOptim.zero_grad()
                Val_loss.backward()
                ValOptim.step()

                with torch.no_grad():
                    v_target = TargValFunc(nstates)
                    y = rewards + GAMMA * (1 - dones) * v_target
                q1 = QFunc_1(states, actions)
                q2 = QFunc_2(states, actions)
                q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                QFuncOptim.zero_grad()
                q_loss.backward()
                QFuncOptim.step()

                new_action, logprob, _ = actor.get_action(states)
                min_q = torch.min(QFunc_1(states, new_action), QFunc_2(states, new_action))
                actor_loss = (ALPHA * logprob - min_q).mean()
                ActorOptim.zero_grad()
                actor_loss.backward()
                ActorOptim.step()

                soft_update(ValFunc, TargValFunc, TAU)

                wandb.log({
                    "td_error": ((q1 - y).pow(2).mean() + (q2 - y).pow(2).mean()).item(),
                    "target_q_value": y.mean().item(),
                    "entropy": -logprob.mean().item(),
                    "episode_reward": episode_reward,
                    "value_loss": Val_loss.item(),
                    "q_loss": q_loss.item(),
                    "actor_loss": actor_loss.item(),
                }, step=episode)

        print(f"[Seed {seed} | Episode {episode}] Reward: {episode_reward:.2f}")
        state, _ = env.reset()
        episode_reward = 0

    torch.save(actor.state_dict(), os.path.join(save_dir, f"actor_seed{seed}_Final.pt"))
    torch.save(QFunc_1.state_dict(), os.path.join(save_dir, f"q1_seed{seed}_Final.pt"))
    torch.save(QFunc_2.state_dict(), os.path.join(save_dir, f"q2_seed{seed}_Final.pt"))
    torch.save(ValFunc.state_dict(), os.path.join(save_dir, f"val_seed{seed}_Final.pt"))

    env.close()
    wandb.finish()
