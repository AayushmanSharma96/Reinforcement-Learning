import gymnasium as gym
from sb3_contrib import TRPO
from gymnasium.wrappers import TimeLimit

# Create the environment
env = gym.make("InvertedPendulum-v4")
max_episode_steps = 100
env = TimeLimit(env, max_episode_steps=max_episode_steps)

# Create the TRPO model
model = TRPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("trpo_inverted_pendulum")

# Test the trained agent
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    # env.render()
    # if done or truncated:
    #     obs = env.reset()
print('Final State = ', obs)
env.close()