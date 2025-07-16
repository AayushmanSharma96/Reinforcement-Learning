import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from gymnasium.wrappers import TimeLimit
import os

class MetricsLoggingCallback(BaseCallback):
    """
    Custom callback for logging episodic rewards to a CSV file.
    """
    def __init__(self, log_path, verbose=0):
        super(MetricsLoggingCallback, self).__init__(verbose)
        self.log_path = log_path
        self.metrics_data = []
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Collect rewards at each step
        reward = self.locals["rewards"][0]  # Assumes single environment
        self.episode_rewards.append(reward)

        # Check if episode is done
        terminated = self.locals["dones"][0]  # Assumes single environment
        truncated = self.locals.get("truncateds", [False])[0]  # Check for truncation (optional)
        
        if terminated or truncated:
            # Summarize episode reward
            episode_reward = np.sum(self.episode_rewards)
            self.episode_count += 1

            # Store the metrics (only episode reward in this case)
            self.metrics_data.append({
                "episode": self.episode_count,
                "reward": episode_reward,
                "timesteps": self.num_timesteps,
            })

            # Reset episode rewards for the next episode
            self.episode_rewards = []

        return True

    def _on_training_end(self) -> None:
        # Save logged metrics to CSV at the end of training
        metrics_df = pd.DataFrame(self.metrics_data)
        metrics_df.to_csv(self.log_path, index=False)

     
        
env = gym.make("InvertedPendulum-v4", render_mode="rgb_array")
env = gym.make('Swimmer-v5', render_mode="rgb_array")
max_episode_steps = 100
env = TimeLimit(env, max_episode_steps=max_episode_steps)


# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.35 * np.ones(n_actions))
#NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Define the path to save CSV logs
log_path = "training_metrics_1000000.csv"

# Instantiate the callback
metrics_callback = MetricsLoggingCallback(log_path=log_path)


model = DDPG("MlpPolicy", env, action_noise=action_noise, buffer_size=1000000, verbose=1)
model.learn(total_timesteps=1000000, log_interval=100, callback=metrics_callback)
model.save("ddpg_pendulum_1000000")
vec_env = model.get_env()

# del model # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_pendulum")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     env.render("human")