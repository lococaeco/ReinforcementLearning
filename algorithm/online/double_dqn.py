import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import ale_py
import wandb
gym.register_envs(ale_py)
import matplotlib.pyplot as plt

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
)

############# HyperParameter ###############
ENV_NAME = ["ALE/Breakout-v5", "ALE/Boxing-v5", "ALE/Enduro-v5", "ALE/Alien-v5", "ALE/Pong-v5"]
LR = 1e-4
BUFFER_SIZE = 100000 
TOTAL_STEP = 10000000
START_E = 1.0
END_E = 0.1
FRACTION = 0.1
LEARNING_START = 8000
TRAIN_FREQ = 4
BATCH_SIZE = 32
GAMMA = 0.99
TARGET_UPDATE_FREQ = 1000
TAU = 1.0
# SEED = [1, 2, 3, 5, 8]
SEED = [3]
use_wandb = True  

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
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)

def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, device="cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.observations = np.zeros((self.buffer_size, *state_dim), dtype=np.uint8)
        self.next_observations = np.zeros((self.buffer_size, *state_dim), dtype=np.uint8)
        self.actions = np.zeros((self.buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size):
        total = self.buffer_size if self.full else self.pos
        indices = np.random.choice(total, batch_size, replace=False)
        return (
            torch.tensor(self.observations[indices], dtype=torch.uint8, device=self.device),
            torch.tensor(self.actions[indices], dtype=torch.int64, device=self.device).unsqueeze(1),
            torch.tensor(self.next_observations[indices], dtype=torch.uint8, device=self.device),
            torch.tensor(self.rewards[indices], dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(self.dones[indices], dtype=torch.float32, device=self.device).unsqueeze(1),
        )

def init_wandb(env_name, seed):
    wandb.init(
        project="DQN-Atari-10000000",
        group=env_name.replace("/", "_"),
        name=f"Double_DQN_{env_name.replace('/', '_')}_seed{seed}",
        config={
            "env_name": env_name,
            "seed": seed,
            "lr": LR,
            "gamma": GAMMA,
            "buffer_size": BUFFER_SIZE,
            "batch_size": BATCH_SIZE,
            "total_steps": TOTAL_STEP,
            "train_freq": TRAIN_FREQ,
            "target_update_freq": TARGET_UPDATE_FREQ,
            "tau": TAU,
            "start_e": START_E,
            "end_e": END_E,
            "exploration_fraction": FRACTION,
            "learning_start": LEARNING_START,
            "architecture": "DQN_CNN_3Conv_2FC",
        },
        save_code=True,
        monitor_gym=True,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for env_name in ENV_NAME:
        for seed in SEED:
            episode = 0
            eppisodic_return = 0

            print(f"Training {env_name} with seed {seed}")
            model_path = f"./model/{env_name.split('/')[-1]}_seed{seed}"
            os.makedirs(model_path, exist_ok=True)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

            if use_wandb:
                init_wandb(env_name, seed)

            envs = gym.vector.SyncVectorEnv([
                lambda: EpisodicLifeEnv( 
                    gym.wrappers.FrameStackObservation(
                    gym.wrappers.AtariPreprocessing(
                        gym.make(env_name, frameskip=1),
                        noop_max=30,
                        frame_skip=4,
                        screen_size=84,
                        terminal_on_life_loss=False,
                        grayscale_obs=True
                    ),
                    stack_size=4,
                    padding_type="zero",
                ),
                ),
            ])


            q_network = QNetwork(envs).to(device)
            target_network = QNetwork(envs).to(device)
            target_network.load_state_dict(q_network.state_dict())
            optimizer = optim.Adam(q_network.parameters(), lr=LR)

            obs_shape = envs.single_observation_space.shape
            rb = ReplayBuffer(BUFFER_SIZE, obs_shape, device)

            state, _ = envs.reset(seed=seed)

            for global_step in range(TOTAL_STEP):
                epsilon = linear_schedule(START_E, END_E, FRACTION * TOTAL_STEP, global_step)

                if random.random() < epsilon:
                    actions = np.array([envs.single_action_space.sample()])
                else:
                    with torch.no_grad():
                        q_values = q_network(torch.tensor(state, dtype=torch.float32, device=device))
                        actions = torch.argmax(q_values, dim=1).cpu().numpy()

                next_state, rewards, terminations, truncations, infos = envs.step(actions)
                dones = np.logical_or(terminations, truncations)
                rb.add(state[0], next_state[0], actions[0], rewards[0], dones[0])
                state = next_state
                eppisodic_return += rewards[0] 

                if dones[0]:
                    episode += 1
                    print(f'episode:{episode}, step:{global_step}, return:{eppisodic_return}')
                    state, _ = envs.reset()
                    
                    if use_wandb:
                        wandb.log({
                            "episode": episode,
                            "episodic_return": eppisodic_return,
                            "episodic_length": infos['episode_frame_number'][0],
                            "epsilon": epsilon,
                            "global_step": global_step
                        })
                    
                    eppisodic_return = 0

                if global_step > LEARNING_START and global_step % TRAIN_FREQ == 0:
                    obs, actions_batch, next_obs, rewards, dones_batch = rb.sample(BATCH_SIZE)

                    with torch.no_grad():
                        next_q_online = q_network(next_obs.float())
                        next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)

                        next_q_target = target_network(next_obs.float())
                        target_q = rewards + GAMMA * (1 - dones_batch) * next_q_target.gather(1, next_actions)

                    current_q = q_network(obs.float()).gather(1, actions_batch)
                    loss = F.mse_loss(current_q, target_q)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if use_wandb:
                        wandb.log({
                            "loss": loss.item(),
                            "global_step": global_step
                        })

                if global_step % TARGET_UPDATE_FREQ == 0:
                    for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                        target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

                if global_step % 100000 == 0:
                    model_file = os.path.join(model_path, f"dqn_latest.pth")
                    torch.save(q_network.state_dict(), model_file)
                    print(f"Saved model at step {global_step} to {model_file}")

            envs.close()
            if use_wandb:
                wandb.finish()