import math
import os
import gym
import d4rl
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.distributions import Normal
from tqdm import trange


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    # wandb.run.save()


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

@dataclass # 아래처럼 예쁘게 만들어주는 데코레이터
class TrainConfig:
    # wandb params
    project: str = "OFFLINE"
    group: str = "SAC"
    name: str = "SAC"
    # model params
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    # training params
    train_start: int = 10000
    buffer_size: int = 1_000_000
    env_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 5
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    log_every: int = 100
    device: str = "cpu"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}" # str(uuid.uuid4())[:8]:고유 id
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class ReplayBuffer:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 buffer_size: int
                 device: str = 'cpu'):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        
        self._states = torch.zeros((buffer_size, state_dim), dtype = torch.float32, device = device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype = torch.float32, device = device)
        self._rewards = torch.zeros((buffer_size, 1), dtype = torch.float32, device = device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype = torch.float32, device = device)
        self._dones = torch.zeros((buffer_size, 1), dtype = torch.float32, device = device)
        self._device = device
        
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)
        
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        



class Critic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int
        ):
        super().__init__()
        input_dim = state_dim + action_dim
        self.critic = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 1))
        

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        # [batch_size, state_dim] + [batch_size, action_dim] = [batch_size, input_dim]
        x = torch.cat([state, action], dim=-1)  
        # [batch_size, 1]
        q_values = self.critic(x) 
        return q_values
    

class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float = 1.0
        ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        self.action_dim = action_dim
        self.max_action = max_action
        
    def forward(
        self, state: torch.Tensor, deterministic: bool = False, need_log_prob: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        shared_param = self.shared(state)
        mean = self.mean_head(shared_param)
        log_std = self.log_std_head(shared_param)
        
        # Clipping Params
        log_std = torch.clip(log_std, -20, 2)
        policy_dist = Normal(mean, torch.exp(log_std))
        
        if deterministic:
            action = mean
        else:
            action = policy_dist.rsample()
        tanh_action, log_prob = torch.tanh(action), None
        
        #log_prob가 작으면 (즉, 확률이 낮으면)
        # → 정책이 “덜 자신있다”는 의미 → entropy term이 크면 탐색 증가
        # log_prob가 크면 (확률 높음)
        # → 정책이 “확신” → entropy term 작음 → exploitation
        if need_log_prob:
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)
                            
        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = not self.training # self.training은 nn.Module로 부터 자연스럽게 train인지 eval인지 나온다.
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy() # self(state, deterministic=deterministic) → self.__call__() → 사실상 forward를 호출
        return action


class SAC:
    def __init__(
        self,
        actor: Actor,
        actor_optimizer: torch.optim.Optimizer,
        critic: Critic,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_learning_rate: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = device

        self.actor = actor
        self.critic1 = critic
        self.critic2 = critic
        with torch.no_grad():
            self.target_critic1 = deepcopy(self.critic1)
            self.target_critic2 = deepcopy(self.critic2)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.tau = tau
        self.gamma = gamma

        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.alpha = self.log_alpha.exp().detach()

    def _alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, action_log_prob = self.actor(state, need_log_prob=True)

        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()

        return loss

    def _actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        action, action_log_prob = self.actor(state, need_log_prob=True)
        q1_values = self.critic1(state, action)
        q2_values = self.critic2(state, action)
        q_values = torch.min(q1_values, q2_values)
        batch_entropy = -action_log_prob.mean().item()
        loss = (self.alpha * action_log_prob - q_values).mean()

        return loss, batch_entropy

    def _critic_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(
                next_state, need_log_prob=True
            )
            q1_next = self.target_critic1(next_state, next_action) # [batch_size, 1]
            q2_next = self.target_critic2(next_state, next_action) # [batch_size, 1]
            
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_action_log_prob.unsqueeze(-1) # # [batch_size, 1] = # [batch_size, 1] - # [batch_size] -> [batch_size, batch_size]
            q_target = reward + self.gamma * (1 - done) * q_next

        q1_values = self.critic1(state, action)
        q2_values = self.critic2(state, action)
        # [batch_size, 1] - [batch_size, 1]
        # loss = ((q_values - q_target) ** 2).mean(dim=1).sum(dim=0)
        loss1 = torch.nn.functional.mse_loss(q1_values, q_target) 
        loss2 = torch.nn.functional.mse_loss(q2_values, q_target) 
        loss = loss1+loss2
        return loss

    def update(self, batch: List[torch.Tensor]) -> Dict[str, float]:
        state, action, reward, next_state, done = [arr.to(self.device) for arr in batch]
        # Usually updates are done in the following order: critic -> actor -> alpha
        
        # Alpha update
        alpha_loss = self._alpha_loss(state)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        # Actor update
        actor_loss, actor_batch_entropy = self._actor_loss(state)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic update
        critic_loss = self._critic_loss(state, action, reward, next_state, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #  Target networks soft update
        with torch.no_grad():
            soft_update(self.target_critic1, self.critic1, tau=self.tau)
            soft_update(self.target_critic2, self.critic2, tau=self.tau)


        update_info = {
            "train/alpha_loss": alpha_loss.item(),
            "train/critic_loss": critic_loss.item(),
            "train/actor_loss": actor_loss.item(),
            "train/batch_entropy": actor_batch_entropy,
            "train/alpha": self.alpha.item(),
        }
        return update_info

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic1.state_dict(),
            "critic": self.critic2.state_dict(),
            "target_critic": self.target_critic1.state_dict(),
            "target_critic": self.target_critic2.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "alpha_optim": self.alpha_optimizer.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic1.load_state_dict(state_dict["critic"])
        self.critic2.load_state_dict(state_dict["critic"])
        self.target_critic1.load_state_dict(state_dict["target_critic"])
        self.target_critic2.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optim"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optim"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()



@torch.no_grad()
def eval_actor(env: gym.Env, actor: Actor, device: str, n_episodes: int, seed: int, video_path: Optional[str] = None) -> np.ndarray:
    actor.eval()
    episode_rewards = []

    # 마지막 에피소드만 영상 저장
    if video_path is not None:
        video_env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_path,
            episode_trigger=lambda ep: ep == n_episodes-1,
            name_prefix=datetime.now().strftime("%H%M")
        )
    else:
        video_env = env

    for ep in range(n_episodes):
        state, _ = video_env.reset(seed=seed+ep)
        done = False
        ep_reward = 0.0
        while not done:
            action = actor.act(state, device)
            next_state, reward, terminated, truncated, _ = video_env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state
        episode_rewards.append(ep_reward)

    actor.train()
    if video_path is not None:
        video_env.close()
    return np.array(episode_rewards)

@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))

    # env setup
    env = gym.make(config.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Replay buffer
    buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, buffer_size=config.buffer_size, device=config.device)
    
    # Actor & Critic setup
    actor = Actor(state_dim, action_dim, config.hidden_dim, config.max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
    critic = Critic(state_dim, action_dim, config.hidden_dim).to(config.device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_learning_rate)

    trainer = SAC(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        alpha_learning_rate=config.alpha_learning_rate,
        device=config.device,
    )
    
    # saving config
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    total_updates = 0
    state, _ = env.reset(seed=config.train_seed)

    episode_rewards_buffer = []  
    episode_reward = 0.0          
    total_step = 0
    
    for epoch in trange(config.num_epochs, desc="Training"):
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            actor.train()
            if total_step < config.train_start:
                action = env.action_space.sample()
            else:
                action = actor.act(state, config.device)
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_step += 1
            done = terminated or truncated
            
            buffer.add_transition(state, action, reward, next_state, done)
            episode_reward += reward
            
            if total_step >= config.train_start:
                batch = buffer.sample(config.batch_size)
                update_info = trainer.update(batch)

                if total_updates % config.log_every == 0:
                    wandb.log({
                               "train/total_step":total_step,
                               **update_info})

                total_updates += 1

            state = next_state
            
            if done:
                episode_rewards_buffer.append(episode_reward)
                avg_last5_reward = (np.mean(episode_rewards_buffer[-5:]) if episode_rewards_buffer else 0.0)
                wandb.log(
                        {"train/episodic_reward":episode_reward,
                         "train/episodic_avg_reward": avg_last5_reward,
                         "train/epoch": epoch,})
                
                episode_reward = 0.0
                state, _ = env.reset(seed=config.train_seed)

        # 평가
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = eval_actor(
                env=gym.make(config.env_name,render_mode="rgb_array"),
                actor=actor,
                n_episodes=config.eval_episodes,
                seed=config.eval_seed,
                device=config.device,
                video_path=config.checkpoints_path
            )
            eval_log = {
                "eval/episodic_reward_mean": np.mean(eval_returns),
                "eval/episodic_reward_std": np.std(eval_returns),
                "eval/epoch": epoch,
            }
            wandb.log(eval_log)

            if config.checkpoints_path is not None:
                torch.save(trainer.state_dict(),os.path.join(config.checkpoints_path, f"{epoch}.pt"))

    wandb.finish()


if __name__ == "__main__":
    train()