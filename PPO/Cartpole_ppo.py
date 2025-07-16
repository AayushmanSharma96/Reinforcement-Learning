import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit, RescaleAction
import csv
import os
from stable_baselines3.common.logger import configure


# Create the environment
env = gym.make("InvertedPendulum-v4")
max_episode_steps = 100
min_action = -30
max_action = 30
env = RescaleAction(env, min_action=min_action, max_action=max_action)
env = TimeLimit(env, max_episode_steps=max_episode_steps)

# Custom callback to log training metrics and save to CSV
class LossCallback(BaseCallback):
    def __init__(self, n_steps=2048*4, csv_filename="cartpole_training_metrics_naveed.csv", verbose=0):
        super(LossCallback, self).__init__(verbose)
        self.csv_filename = csv_filename
        self.n_steps = n_steps  # Track every n_steps
        self.steps_since_last_update = 0

    def _on_step(self) -> bool:
        # Increment the step counter
        self.steps_since_last_update += 1
        
        # Log metrics every n_steps
        if self.steps_since_last_update >= self.n_steps:
            # Extract metrics from the logger
            metrics = {
                "step": self.num_timesteps,
                "policy_gradient_loss": self.logger.name_to_value.get("train/policy_gradient_loss", None),
                "value_loss": self.logger.name_to_value.get("train/value_loss", None),
                "entropy_loss": self.logger.name_to_value.get("train/entropy_loss", None),
                "total_loss": self.logger.name_to_value.get("train/loss", None),
                "episodic_reward": self.logger.name_to_value.get("rollout/ep_rew_mean", None)
            }

            # Open the CSV file in append mode
            with open(self.csv_filename, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                
                # Write the header only if the file is empty
                if os.stat(self.csv_filename).st_size == 0:  # If the file is empty
                    writer.writerow([
                        "step", 
                        "policy_gradient_loss", 
                        "value_loss", 
                        "entropy_loss", 
                        "total_loss", 
                        "episodic_reward"
                    ])
                
                # Write the metrics to the CSV file
                # if metrics["episodic_reward"] is not None:
                writer.writerow([
                    metrics["step"],
                    metrics["policy_gradient_loss"],
                    metrics["value_loss"],
                    metrics["entropy_loss"],
                    metrics["total_loss"],
                    metrics["episodic_reward"]
                ])
            
            # Reset the step counter after writing
            self.steps_since_last_update = 0

        return True  # Continue training

# Set the device to CPU (force CPU usage)
device = "cpu"
tmp_path = "./run_logs/400_300/"
# set up logger
new_logger = configure(tmp_path, ["csv"])

# Define the network architecture (two hidden layers of sizes 300 and 400)
policy_kwargs = dict(
    net_arch=[400, 300],  # Two hidden layers of size 300 and 400
)

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048*4, batch_size=256, 
            n_epochs=10, gamma=0.99, gae_lambda=0.9, clip_range=0.2, normalize_advantage=True, 
            ent_coef=0.0, vf_coef=0.19816, max_grad_norm=0.5, device=device, policy_kwargs=policy_kwargs)


# Set new logger
model.set_logger(new_logger)

# loss_callback = LossCallback(csv_filename="cartpole_training_metrics_naveed.csv")
model.learn(total_timesteps=4000000)#, callback=loss_callback)

# Save the trained model
model.save("ppo_inverted_pendulum_test")

# Test the trained agent
# obs = env.reset()[0]
# for _ in range(100):
#     action, _states = model.predict(obs)
#     obs, reward, done, truncated, info = env.step(action)
#     if done or truncated:
#         obs = env.reset()[0]
# print('Final State = ', obs)
env.close()
