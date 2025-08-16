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

ENV_NAME = ["HalfCheetah-v5",
            "Ant-v5",
            "Hopper-v5",
            "Humanoid-v5",
            "HumanoidStandup-v5",
            "InvertedDoublePendulum-v5",
            "InvertedPendulum-v5",
            "Pusher-v5",
            "Reacher-v5",
            "Swimmer-v5",
            "Walker2d-v5"]

TODAY = "FIXED"
NUM_EPISODE = 2000
HIDDEN_DIM = 64
BUFFER_MAX = 2000
VALUE_ITER = 150       
VALUE_LR = 0.0001
GAMMA = 0.99        
LAMDA = 0.95        
BACKTRACK_ITER = 10        
BACKTRACK_ALPHA = 0.5        
MAX_KL = 0.01     
CLIP_E = 0.2      
SEED = [1, 3]

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


# -------------------------------
# TRPO Agent
# -------------------------------
class TRPO:
    def __init__(self, obs_dim, act_dim, act_low, act_high):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_low = act_low
        self.act_high = act_high
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy = ContinuousGaussianPolicy(obs_dim, act_dim, self.act_low, self.act_high).to(self.device)
        self.old_policy = ContinuousGaussianPolicy(obs_dim, act_dim, self.act_low, self.act_high).to(self.device)
        self.value_network = ValueNetwork(obs_dim).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=1e-4)
        self.trajectories = RolloutBuffer(max_size=BUFFER_MAX,
                                          state_dim=obs_dim,
                                          action_dim=act_dim,
                                          device=self.device)
        
        self.value_network_iter = VALUE_ITER
        self.gamma = GAMMA
        self.lamda = LAMDA
        self.backtrack_iter = BACKTRACK_ITER
        self.backtrack_alpha = BACKTRACK_ALPHA
        self.max_kl = MAX_KL
        self.clip_epsilon = CLIP_E

    def flat_grad(self, grads):
        return torch.cat([
            (grad if grad is not None else torch.zeros_like(param)).view(-1)
            for grad, param in zip(grads, self.policy.parameters())
        ])
        
    def flat_params(self, model):
        return torch.cat([param.data.view(-1) for param in model.parameters()])

    def kl_divergence(self, dist_old, dist_new):
        mean_old, std_old = dist_old.mean, dist_old.stddev
        mean_new, std_new = dist_new.mean, dist_new.stddev
        
        kl = (
            torch.log(std_new / std_old)
            + (std_old.pow(2) + (mean_old - mean_new).pow(2)) / (2.0 * std_new.pow(2))
            - 0.5
        )
        return kl.sum(-1)
    
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
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def conjugate_gradient(self, obs, b, nsteps=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)

        for _ in range(nsteps):
            Ap = self.fisher_vector_product(obs, p)
            alpha = rdotr / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def fisher_vector_product(self, obs, v):
        kl = self.kl_divergence(self.old_policy(obs), self.policy(obs))
        kl = kl.mean()

        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = self.flat_grad(grads)
        kl_v = (flat_grad_kl * v).sum()
        grads2 = torch.autograd.grad(kl_v, self.policy.parameters())
        flat_grad2 = self.flat_grad(grads2).detach()
        return flat_grad2 + 0.01 * v 


    def update_model(self, model, new_params):
        index = 0
        for param in model.parameters():
            length = len(param.view(-1))
            param.data.copy_(new_params[index:index+length].view(param.size()))
            index += length
    
    def train(self):
        # 현 시점의 policy을 old_policy로 복사
        # 이제 old_policy는 이번 학습 스텝에서 데이터를 수집한 정책
        self.old_policy.load_state_dict(self.policy.state_dict())

        states, actions, rewards, next_states, dones = self.trajectories.sample()

        # GAE 및 Value-network 업데이트
        advantages, returns = self.compute_gae(states, rewards, next_states, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.value_network_iter):
            v_pred = self.value_network(states)
            v_loss = F.mse_loss(v_pred, returns)
            self.value_optimizer.zero_grad()
            v_loss.backward()
            self.value_optimizer.step()
        
        # old_policy의 로그 확률 계산
        with torch.no_grad():
            old_dist = self.old_policy(states)
            old_log_probs = self.old_policy.log_prob(states, actions)

        # 업데이트 전 Surrogate Loss 계산
        dist = self.policy(states)
        log_probs = self.policy.log_prob(states, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        surrogate_loss_before_update = (ratio * advantages).mean()

        # Surrogate Loss의 그래디언트 계산
        grads = torch.autograd.grad(surrogate_loss_before_update, self.policy.parameters())
        flat_grads = self.flat_grad(grads).detach()

        # Conjugate Gradient로 탐색 방향(step_dir) 계산
        step_dir = self.conjugate_gradient(states, flat_grads)

        # Backtracking Line Search
        gHg = (step_dir * self.fisher_vector_product(states, step_dir)).sum().abs() + 1e-8
        step_size = torch.sqrt(2 * self.max_kl / gHg)
        
        # 현재 파라미터를 '백업', 탐색 실패 시 복구
        old_params = self.flat_params(self.policy)
        
        with torch.no_grad():
            success = False
            for i in range(self.backtrack_iter):
                step_frac = self.backtrack_alpha ** i 
                new_params = old_params + step_frac * step_size * step_dir
                self.update_model(self.policy, new_params)

                # 새 파라미터로 Surrogate Loss와 KL Divergence 재계산
                new_dist = self.policy(states)
                new_log_probs = self.policy.log_prob(states, actions)
                new_ratio = torch.exp(new_log_probs - old_log_probs) 
                new_surrogate_loss = (new_ratio * advantages).mean()

                kl = self.kl_divergence(old_dist, new_dist).mean()

                # 실제 정책 성능 향상(actual_improve)과 KL 제약조건 확인
                actual_improve = new_surrogate_loss - surrogate_loss_before_update
                
                if actual_improve > 0 and kl <= self.max_kl:
                    print(f"Line search success at step fraction {step_frac:.3f}, KL={kl:.6f}, Actual Improve={actual_improve:.4f}")
                    success = True
                    break

            if not success:
                print("Line search failed, reverting to old parameters.")
                self.update_model(self.policy, old_params)

            wandb.log({
                    "value_loss": v_loss.item(),
                    "advantage_mean": advantages.mean().item(),
                    "advantage_std": advantages.std().item(),
                    "return_mean": returns.mean().item(),
                    "return_std": returns.std().item(),
                    "new_surrogate_loss": new_surrogate_loss.item(),
                    "surrogate_loss_before_update": surrogate_loss_before_update.item(),
                    "kl_divergence": kl,
                    "actual_improve": actual_improve,
                })
        
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
        project="TRPO_"+TODAY,
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
            "TRPO",                    # 알고리즘명
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
            
            agent = TRPO(obs_dim, act_dim, act_low, act_high)
            
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