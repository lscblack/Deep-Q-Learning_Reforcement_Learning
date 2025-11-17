#!/usr/bin/env python3

import gymnasium as gym
from stable_baselines3 import DQN
import os
import json

ENV_ID = "ALE/IceHockey-v5"
RESULTS_DIR = "cyiza_results"


def find_best_model():
    summary_path = os.path.join(RESULTS_DIR, "results_summary.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError("Training results_summary.json not found.")

    with open(summary_path, "r") as f:
        results = json.load(f)

    valid = [r for r in results if "mean_reward" in r]

    if len(valid) == 0:
        raise RuntimeError("No successful experiment with a saved model.")

    best = max(valid, key=lambda x: x["mean_reward"])
    return best["model_path"]


def main():
    model_path = find_best_model()
    print("Loading best model:", model_path)

    model = DQN.load(model_path)

    env = gym.make(ENV_ID, render_mode="human")
    obs, _ = env.reset()
    total_reward = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print("Episode reward:", total_reward)
            obs, _ = env.reset()
            total_reward = 0


if __name__ == "__main__":
    main()
