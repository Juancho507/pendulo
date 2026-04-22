import gymnasium as gym
from stable_baselines3 import PPO
from rl.env import CustomCartPole
from nlp.interpreter import interpret_instruction
import time
import sys
import os

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/cartpole_model")


def main():
    if len(sys.argv) > 1:
        instruction = " ".join(sys.argv[1:])
    else:
        instruction = "Maintain your balance for as long as possible, minimize sudden movements, and prioritize stability"

    print("\nINSTRUCTION:")
    print(instruction)

    config = interpret_instruction(instruction)
    print("\nGENERATED CONFIG:", config)

    env = gym.make("CartPole-v1", render_mode="human")
    env = CustomCartPole(env, config)

    try:
        model = PPO.load(MODEL_PATH)
        print(f"\nLoaded model from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"\nNo trained model found at: {MODEL_PATH}")
        print("Please run the train service first:  docker compose run train")
        sys.exit(1)

    obs, _ = env.reset()
    print("\nRunning visualization... Press Ctrl+C to stop.\n")

    try:
        for step in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            time.sleep(0.02)
            if terminated or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nVisualization stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
