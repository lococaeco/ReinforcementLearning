import torch
import gymnasium as gym
import numpy as np
import os
import imageio
from torch.distributions.normal import Normal
from REINFORCE_train import PolicyNetwork  # 학습에 사용한 네트워크 구조 그대로 import

# 설정
save_dir = "./model_halfcheetah"
video_dir = os.path.join(save_dir, "videos")
os.makedirs(video_dir, exist_ok=True)

# 환경 준비
env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# 평가할 시드 및 반복 수
seeds = [1, 2, 3, 5, 8]  # 필요 시 [1, 2, 3, 5, 8]
n_eval_episodes = 1

# 평가 함수
def evaluate_and_save_video(seed):
    model_path = os.path.join(save_dir, f"reinforce_policy_seed{seed}.pt")
    if not os.path.exists(model_path):
        print(f"⚠️ 모델 없음: {model_path}")
        return

    # 정책 로드
    policy = PolicyNetwork(obs_dim, act_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    for ep in range(n_eval_episodes):
        state, _ = env.reset(seed=seed)
        done = False
        frames = []
        total_reward = 0
        time = 0 

        while not done:
            # 프레임 저장
            frame = env.render()
            frames.append(frame)
            time += 1
            print(time)
            # 행동 선택
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                mean, std = policy(state_tensor)
                dist = Normal(mean, std)
                action = dist.mean  # 평가 시에는 deterministic 행동 사용
            action = action.numpy()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        print(f"🎞️ Seed {seed} | Episode {ep + 1} | Total Reward: {total_reward:.2f}")

        # 영상 저장
        video_path = os.path.join(video_dir, f"reinforce_seed{seed}_ep{ep+1}.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"✅ 영상 저장 완료: {video_path}")

if __name__ == "__main__":
    for seed in seeds:
        evaluate_and_save_video(seed)
