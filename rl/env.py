# RL environment (CartPole simulation)
import gymnasium as gym 

class CustomCartPole(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        self.prev_action = 0

    def step(self, action):
        # Take a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Unpack observation
        x, x_dot, theta, theta_dot = obs

        # Custom Reward Engineering

        # Base reward (how long the pole stays alive)
        reward_total = reward * self.config["duration"]

        # Penalize deviation from vertical (stability)
        reward_total -= self.config["stability"] * abs(theta)

        # Penalize distance from center — keeps the cart from drifting
        reward_total -= self.config["stability"] * abs(x) * 0.5

        # Penalize fast movements (smoothness)
        reward_total -= self.config["smoothness"] * abs(x_dot) * 0.1

        # Penalize unnecessary action changes (efficiency behavior)
        if action != self.prev_action:
            reward_total -= self.config["smoothness"] * 0.05

        self.prev_action = action

        # Strong penalty if episode ends (pole falls)
        if terminated:
            reward_total -= 5.0

        return obs, reward_total, terminated, truncated, info