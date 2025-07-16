import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.logger import configure
import numpy as np

# Define reward function
# def reward_function(observation, action):
#     diag_q = 1e-8 * 2 * np.array([1, 1, 1, 1e5, 1e5, 1e5, 0, 0, 0, 0, 0, 0, 0]) 
#     r = 1e-8 * 2000 * np.array([1, 1, 1])
#     cost = diag_q[0] * observation[0] ** 2 + diag_q[1] * observation[1] ** 2 + diag_q[2] * observation[2] ** 2
#     cost += diag_q[3] * observation[3] ** 2 + diag_q[4] * observation[4] ** 2 + diag_q[5] * observation[5] ** 2
#     ctrl_cost = r[0] * action[0] ** 2 + r[1] * action[1] ** 2 + r[2] * action[2] ** 2
#     return -(cost + ctrl_cost)/1e5

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
        custom_reward = reward_function(obs, action)  # Example integration
        return obs, custom_reward, done, truncated, info

# Create environment
env_name = "Rendezvous_v0"  # Replace with your environment's ID
env = gym.make(env_name)
env = Monitor(env)  # Track episode lengths and rewards
env = CustomEnvWrapper(env)  # Add custom reward function

# Optional: Vectorize environment for parallel processing (if supported by your hardware)
vec_env = make_vec_env(lambda: env, n_envs=1)

# Define action noise
action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=0.01 * np.ones(env.action_space.shape))

policy_kwargs = dict(
    net_arch=[400, 300],  # Two hidden layers of size 300 and 400
)

# Define SAC model
model = SAC(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    learning_rate=3e-5,
    buffer_size=1000000,
    batch_size=256,
    action_noise=action_noise,
    tensorboard_log="./sac_tensorboard/",
    gamma=0.99,
    tau=0.005,
    train_freq=4,
    gradient_steps=4,
    policy_kwargs=policy_kwargs,
    ent_coef="auto_0.1",  # Enable adaptive entropy
    learning_starts=20000,  # Start learning after 20,000 steps
)

# Set new logger
tmp_path = "./run_logs/"
# set up logger
new_logger = configure(tmp_path, ["csv"])
model.set_logger(new_logger)

# Callbacks for monitoring
checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path="./sac_checkpoints/")
eval_callback = EvalCallback(
    env,
    eval_freq=5000,  # Evaluate every 5000 steps
    best_model_save_path="./sac_eval/",
    log_path="./sac_eval_logs/",
    deterministic=True,
    render=False,
)




# Train the model
model.learn(total_timesteps=1000000, callback=[checkpoint_callback, eval_callback])

# Save the trained model
model.save("sac_rendezvous")

# Load and evaluate the trained model
'''loaded_model = SAC.load("sac_rendezvous")
obs = env.reset()

for _ in range(1000):
    action, _ = loaded_model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()'''
