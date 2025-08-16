import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import gymnasium as gym
import numpy as np
import random
import os

import wandb
save_dir = "./model"
os.makedirs(save_dir, exist_ok=True)

ENV_NAME = ["HalfCheetah-v5"]

#
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

TODAY = "FIXED"
NUM_EPISODE = 3000
HIDDEN_DIM = 128
BUFFER_MAX = 2000
VALUE_LR = 0.0001
GAMMA = 0.99        
LAMDA = 0.95        
MAX_KL = 0.01     
CLIP_E = 0.2      
SEED = [1, 3, 5, 7, 9]
PPO_EPOCHS = 20

class ContinuousGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, action_low, action_high, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))
    
    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_head(x)
        log_std = torch.tanh(self.log_std_head(x))
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def _scale_action(self, norm_action):
        return self.action_low + 0.5 * (norm_action + 1.0) * (self.action_high - self.action_low)

    def _unscale_action(self, scaled_action):
        return 2.0 * (scaled_action - self.action_low) / (self.action_high - self.action_low) - 1.0

    def sample(self, obs):
        with torch.no_grad():
            dist = self.forward(obs)
            raw_action = dist.sample()  # u ~ N(μ, σ)
            norm_action = torch.tanh(raw_action)  # a = tanh(u)
            action = self._scale_action(norm_action)
        return action
        
    def get_deterministic_action(self, obs):
        with torch.no_grad():
            dist = self.forward(obs)
            norm_action = torch.tanh(dist.mean)
            action = self._scale_action(norm_action)
        return action

    def log_prob(self, obs, action):
        norm_action = self._unscale_action(action)
        norm_action = torch.clamp(norm_action, -0.999999, 0.999999)
        # a = tanh(u)  =>  u = atanh(a)
        raw_action = torch.atanh(norm_action)
        dist = self.forward(obs)
        # log π(a|s) = log μ(u|s) - Σ log(1 - a²)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        log_prob -= torch.log(1 - norm_action.pow(2) + 1e-6).sum(dim=-1)
        return log_prob


# -------------------------------
# Value Network
# -------------------------------
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# -------------------------------
# Rollout Buffer
# -------------------------------
class RolloutBuffer:
    def __init__(self, max_size, state_dim, action_dim, device):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, transition):
        s, a, r, s_next, d = transition
        self.states[self.ptr] = np.array(s, dtype=np.float32)
        self.actions[self.ptr] = np.array(a, dtype=np.float32).reshape(-1)
        self.rewards[self.ptr] = np.array(r, dtype=np.float32)
        self.next_states[self.ptr] = np.array(s_next, dtype=np.float32)
        self.dones[self.ptr] = np.array(d, dtype=np.float32)

        self.ptr += 1
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        idx = slice(0, self.size)
        return (
            torch.from_numpy(self.states[idx]).float().to(self.device),
            torch.from_numpy(self.actions[idx]).float().to(self.device),
            torch.from_numpy(self.rewards[idx]).float().to(self.device),
            torch.from_numpy(self.next_states[idx]).float().to(self.device),
            torch.from_numpy(self.dones[idx]).float().to(self.device)
        )

    def reset(self):
        self.ptr = 0
        self.size = 0


class PPO:
    def __init__(self, obs_dim, act_dim, act_low, act_high):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_low = act_low
        self.act_high = act_high
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Networks ---
        # The policy network that will be trained
        self.policy = ContinuousGaussianPolicy(obs_dim, act_dim, self.act_low, self.act_high).to(self.device)
        # The old policy, used to calculate the probability ratio
        self.old_policy = ContinuousGaussianPolicy(obs_dim, act_dim, self.act_low, self.act_high).to(self.device)
        self.value_network = ValueNetwork(obs_dim).to(self.device)

        # --- Optimizers ---
        # PPO uses a standard first-order optimizer like Adam
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=1e-4)
        
        # --- Buffer ---
        self.trajectories = RolloutBuffer(max_size=BUFFER_MAX,
                                          state_dim=obs_dim,
                                          action_dim=act_dim,
                                          device=self.device)
        
        # --- Hyperparameters ---
        self.ppo_epochs = PPO_EPOCHS  # Number of epochs to train on the collected data
        self.gamma = GAMMA
        self.lamda = LAMDA
        self.clip_epsilon = CLIP_E  # Epsilon for the clipped objective
        
        # Coefficients for the combined loss function
        self.c1 = 0.5  # Value function coefficient
        self.c2 = 0.001 # Entropy bonus coefficient

    def compute_gae(self, states, rewards, next_states, dones):
        with torch.no_grad():
            rewards = rewards.squeeze(-1)         # (B,)
            dones = dones.squeeze(-1)             # (B,)
            values = self.value_network(states)   # (B,)
            next_values = self.value_network(next_states)  # (B,)

            deltas = rewards + self.gamma * (1 - dones) * next_values - values
            advantages = torch.zeros_like(rewards).to(self.device)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.lamda * (1 - dones[t]) * gae
                advantages[t] = gae

            returns = advantages + values
            # Normalize advantages for stable updates
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def train(self):
        # 1. Sync the old policy with the current policy before the update
        self.old_policy.load_state_dict(self.policy.state_dict())

        # 2. Sample a batch of trajectories
        states, actions, rewards, next_states, dones = self.trajectories.sample()

        # 3. Compute advantages and returns (TD(lambda) returns)
        advantages, returns = self.compute_gae(states, rewards, next_states, dones)
        
        # Pre-calculate log probabilities with the old policy (no gradients needed)
        with torch.no_grad():
            old_log_probs = self.old_policy.log_prob(states, actions)
        
        # 4. Update the policy and value networks for several epochs
        for _ in range(self.ppo_epochs):
            # --- Policy Loss (Clipped Surrogate Objective) ---
            dist = self.policy(states)
            log_probs = self.policy.log_prob(states, actions)
            
            # Ratio of new policy to old policy: pi_theta(a|s) / pi_theta_old(a|s)
            ratio = torch.exp(log_probs - old_log_probs)

            # The two terms of the PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            
            # The policy loss is the negative minimum of the two terms
            policy_loss = -torch.min(surr1, surr2).mean()

            # --- Value Loss ---
            # Mean Squared Error between predicted values and actual returns
            v_pred = self.value_network(states)
            value_loss = F.mse_loss(v_pred, returns)

            # --- Entropy Bonus ---
            # An entropy term is added to encourage exploration
            entropy_bonus = dist.entropy().mean()

            # --- Total Loss and Optimization Step ---
            # The total loss is a combination of policy loss, value loss, and entropy bonus
            loss = policy_loss + (self.c1 * value_loss) - (self.c2 * entropy_bonus)
            
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()

        # Logging for monitoring
        with torch.no_grad():
            # Approximate KL divergence for debugging and monitoring
            approx_kl = (old_log_probs - self.policy.log_prob(states, actions)).mean().item()
            
            wandb.log({
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy_bonus.item(),
                "approx_kl_divergence": approx_kl,
                "advantage_mean": advantages.mean().item(),
                "return_mean": returns.mean().item(),
            })
            
        # 5. Clear the buffer for the next iteration
        self.trajectories.reset()
        
# -------------------------------
# Seeding
# -------------------------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# -------------------------------
# Logging
# -------------------------------
def init_wandb(env_name, seed):
    wandb.init(
        project="PPO_"+TODAY,
        group=env_name.split("-")[0],
        name=f"{env_name.split('-')[0]}_{seed}",
        config={
            "env_name": env_name,
            "seed": seed,
            "value_lr": VALUE_LR,
            "gamma": GAMMA,
            "buffer_max": BUFFER_MAX,
        },
        tags=[
            env_name.split("-")[0],    # 환경명 (버전 제외)
            f"seed-{seed}",            # 시드
            "PPO",                    # 알고리즘명
        ],
        save_code=True,
        monitor_gym=True,
    )

# -------------------------------
# Evaluationg
# -------------------------------
def evaluate(env_name, agent, seed, eval_iterations):
    agent.policy.eval()

    env = gym.make(env_name)
    scores = []
    
    for i in range(eval_iterations):
        (obs, _) = env.reset(seed=seed + i)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
        
        terminated = False
        truncated = False
        score = 0
        
        while not (terminated or truncated):
            with torch.no_grad():
                action = agent.policy.get_deterministic_action(obs_tensor)
            
            action = action.cpu().numpy()[0]
            s_prime, r, terminated, truncated, _ = env.step(action)
            
            score += r
            obs_tensor = torch.tensor(s_prime, dtype=torch.float32).unsqueeze(0).to(agent.device) # obs 갱신
            
        scores.append(score)
        
    env.close()
    agent.policy.train()

    return np.mean(scores)



# -------------------------------
# Main Loop
# -------------------------------
if __name__ == "__main__":
    for env_name in ENV_NAME:
        for seed in SEED:
            today = TODAY
            save_dir = f"./model/{today}/{env_name.split('-')[0]}/seed_{seed}"
            os.makedirs(save_dir, exist_ok=True)
            
            env = gym.make(env_name)
            seed_all(seed)
            init_wandb(env_name, seed)
            
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]

            act_low = env.action_space.low
            act_high = env.action_space.high
            
            agent = PPO(obs_dim, act_dim, act_low, act_high)
            
            for episode in range(NUM_EPISODE):
                obs, _ = env.reset()
                episode_reward = 0

                while True:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
                    action = agent.policy.sample(obs_tensor)
                    action = action.cpu().numpy()[0]

                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    agent.trajectories.add((obs, action, reward, next_obs, done))

                    obs = next_obs
                    episode_reward += reward

                    if agent.trajectories.size >= agent.trajectories.max_size:
                        agent.train()

                    if done or truncated:
                        break
                
                wandb.log({"episodic_reward": episode_reward,
                           "episode": episode})
                print(f"Episode {episode} | Reward: {episode_reward:.2f}")
                
                if episode % 100 == 0 and episode > 0:
                    eval_score = evaluate(env_name, agent, seed, eval_iterations=10)
                    wandb.log({"eval_avg_return": eval_score, "episode": episode})
                    print(f"Episode {episode} | Evaluation Average Return: {eval_score}")
                
            policy_path = os.path.join(save_dir, "policy.pth")
            value_path = os.path.join(save_dir, "value.pth")
            torch.save(agent.policy.state_dict(), policy_path)
            torch.save(agent.value_network.state_dict(), value_path)

            wandb.save(os.path.join(save_dir, "*.pth"))
            wandb.finish()
            env.close()