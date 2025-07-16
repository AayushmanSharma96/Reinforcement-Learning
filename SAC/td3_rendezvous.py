import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
import torch
import numpy as np

# Define reward function
def reward_function(observation, action):
    diag_q = 1e-8 * 2 * np.array([1, 1, 1, 1e5, 1e5, 1e5, 0, 0, 0, 0, 0, 0, 0]) 
    r = 1e-8 * 2000 * np.array([1, 1, 1])
    cost = sum(diag_q[i] * observation[i]**2 for i in range(6))
    ctrl_cost = sum(r[i] * action[i]**2 for i in range(3))
    raw_reward = -(cost + ctrl_cost)
    normalized_reward = raw_reward / 100.0  # Scale to [-1, 1]
    return normalized_reward

# Custom Environment Wrapper for Reward Shaping
class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomEnvWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        custom_reward = reward_function(obs, action)  # Apply custom reward function
        return obs, custom_reward, done, truncated, info

# Create environment
env_name = "Rendezvous_v0"  # Replace with your environment's ID
env = gym.make(env_name)
env = Monitor(env)  # Track episode lengths and rewards
env = CustomEnvWrapper(env)  # Add custom reward function

# Optional: Vectorize environment for parallel processing (if supported by your hardware)
vec_env = make_vec_env(lambda: env, n_envs=1)

# Define Ornstein-Uhlenbeck action noise
n_actions = env.action_space.shape[0]
ou_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(n_actions),
    sigma=1.0 * np.ones(n_actions),  # Noise scale
    theta=0.15,  # Speed of mean reversion
    dt=0.01,  # Time step for noise update
)

policy_kwargs = dict(
    net_arch=[400, 300],  # Two hidden layers of size 512 each
    activation_fn=torch.nn.ReLU,  # ReLU activation for non-linearity
)

# Define TD3 model with tuned hyperparameters
model = TD3(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    learning_rate=1e-4,  # Optimized learning rate
    buffer_size=1_000_000,
    batch_size=512,  # Larger batch size for stability
    action_noise=ou_noise,
    tensorboard_log="./td3_tensorboard/",
    gamma=0.995,  # Slightly lower discount for long-term rewards
    tau=0.002,  # Target network update rate
    train_freq=4,  # Train every 4 steps
    gradient_steps=1,  # One gradient step per train step
    policy_delay=2,  # Update policy network less frequently
    policy_kwargs=policy_kwargs,
)

# Set new logger
tmp_path = "./run_logs/"
new_logger = configure(tmp_path, ["csv"])
model.set_logger(new_logger)

# Callbacks for monitoring
checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path="./td3_checkpoints/")
eval_callback = EvalCallback(
    env,
    eval_freq=10000,  # Evaluate every 5000 steps
    best_model_save_path="./td3_eval/",
    log_path="./td3_eval_logs/",
    deterministic=True,
    render=False,
)

# Train the model
model.learn(total_timesteps=1_000_000, callback=[checkpoint_callback, eval_callback])

# Save the trained model
model.save("td3_rendezvous_ou")

# Load and evaluate the trained model
'''
loaded_model = TD3.load("td3_rendezvous_ou")
obs = env.reset()

for _ in range(1000):
    action, _ = loaded_model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()
'''
