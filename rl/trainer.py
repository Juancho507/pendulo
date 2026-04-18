# RL environment (CartPole simulation)
import gymnasium as gym   

# PPO algorithm for training the agent
from stable_baselines3 import PPO  

# Custom environment with modified rewards
from rl.env import CustomCartPole   

import os

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/cartpole_model")


def train_model(config, timesteps=150000):

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Create CartPole environment
    env = gym.make("CartPole-v1")

    # Wrap with custom reward logic
    env = CustomCartPole(env, config)

    # Initialize PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=timesteps)

    # Save trained model
    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    return model, env