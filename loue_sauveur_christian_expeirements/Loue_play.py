#!/usr/bin/env python3
"""
play.py

Play the best trained DQN agent on Atari using official Gymnasium Atari wrapper.
"""

import os
import json
import gymnasium as gym
from stable_baselines3 import DQN

# ---------------- Config ----------------
os.environ["ALE_ROM_DIR"] = os.path.expanduser("~/ale_roms")
ROM_NAME = "Alien.bin"
ROM_PATH = os.path.join(os.environ["ALE_ROM_DIR"], ROM_NAME)
print("Using ROM:", ROM_PATH)

RESULTS_SUMMARY = "./results/results_summary.json"
EPISODES = 15

# ---------------- Main Play Loop ----------------
def main():
    # Select best model from results_summary.json
    with open(RESULTS_SUMMARY, "r") as f:
        results = json.load(f)
    best_model = max(results, key=lambda x: x["mean_reward"])
    best_model_path = best_model["model_path"]
    print(f"Selected best model: {best_model['algo']} with mean reward {best_model['mean_reward']}")

    # Use official Gymnasium Atari environment
    env = gym.make(f"ALE/{ROM_NAME}-v5", render_mode="human")
    
    model = DQN.load(best_model_path, env=env)  # load best trained model

    for ep in range(EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)  # Greedy policy
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        print(f"Episode {ep+1} finished with reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
