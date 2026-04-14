# Numerical operations (e.g., averages, arrays)
import numpy as np

def evaluate_model(model, env, episodes=5):
    """
    Evaluate a trained RL model over several episodes.

    Parameters:
    - model: trained RL agent (PPO)
    - env: environment (CartPole)
    - episodes: number of evaluation runs
    """

    # List to store total reward of each episode
    results = []

    # Loop through each evaluation episode
    for ep in range(episodes):

        # Reset environment to initial state
        obs, _ = env.reset()

        # Initialize total reward for this episode
        total_reward = 0

        # Run one episode (max 500 steps for CartPole)
        for step in range(500):

            # Predict action using trained model
            # deterministic=True ensures consistent evaluation
            action, _ = model.predict(obs, deterministic=True)

            # Apply action to environment
            obs, reward, terminated, truncated, _ = env.step(action)

            # Accumulate reward
            total_reward += reward

            # Check if episode has ended
            if terminated:
                # Episode ended because pole fell or failure condition
                break

            if truncated:
                # Episode ended due to time limit (max steps reached)
                break

        # Store total reward for this episode
        results.append(total_reward)

        # Print episode result
        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f}")

    # Compute average performance
    average_reward = np.mean(results)

    # Print final evaluation result
    print("\nEvaluation Summary")
    print(f"Average Reward over {episodes} episodes: {average_reward:.2f}")