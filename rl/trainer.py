# RL environment (CartPole simulation)
import gymnasium as gym   

# PPO algorithm for training the agent
from stable_baselines3 import PPO  

# Custom environment with modified rewards
from rl.env import CustomCartPole   


def train_model(config, timesteps=10000):

    # Create CartPole environment
    env = gym.make("CartPole-v1")

    # Wrap with custom reward logic
    env = CustomCartPole(env, config)

    # Initialize PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=timesteps)

    # Save trained model
    model.save("cartpole_model")

    return model, env