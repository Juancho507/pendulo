# RL environment
import gymnasium as gym  

# PPO algorithm
from stable_baselines3 import PPO  

# Custom reward environment
from rl.env import CustomCartPole     

# NLP module
from nlp.interpreter import interpret_instruction  

# Control execution speed (delays)
import time

# Natural language instruction
instruction = """
Maintain balance as long as possible,
minimize sudden movements and prioritize stability.
"""

# Convert instruction to config
config = interpret_instruction(instruction)

# Create environment with human rendering (visualization)
env = gym.make("CartPole-v1", render_mode="human")

# Apply custom reward wrapper
env = CustomCartPole(env, config)

# Load trained model
model = PPO.load("cartpole_model")

# Reset environment
obs, _ = env.reset()

# Run simulation
for step in range(1000):

    # Predict action using trained model
    action, _ = model.predict(obs, deterministic=True)

    # Apply action
    obs, _, terminated, truncated, _ = env.step(action)

    # Slow down visualization (so it can be seen clearly)
    time.sleep(0.02)

    # Reset environment if episode ends
    if terminated or truncated:
        obs, _ = env.reset()