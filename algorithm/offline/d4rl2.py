import d4rl
import gym

# 환경 생성
env_name = "halfcheetah-medium-v2"  # 원하는 D4RL 환경
env = gym.make(env_name)

# D4RL dataset 불러오기
dataset = d4rl.qlearning_dataset(env)

# dataset 확인
print("Observations shape:", dataset["observations"].shape)
print("Actions shape:", dataset["actions"].shape)
print("Rewards shape:", dataset["rewards"].shape)
print("Terminals shape:", dataset["terminals"].shape)
