#!/usr/bin/env python3
"""
train.py

Train DQN on Atari using official Gymnasium Atari wrapper for SB3.
"""

import os
import numpy as np
import random
import json
import gc
from pathlib import Path
import warnings

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

# ---------------- Config ----------------
os.environ["ALE_ROM_DIR"] = os.path.expanduser("~/ale_roms")
ROM_NAME = "Alien.bin"
ROM_PATH = os.path.join(os.environ["ALE_ROM_DIR"], ROM_NAME)
print("Using ROM:", ROM_PATH)

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTEPS = 50_000
EVAL_EPISODES = 5
SEED = 12345
DEVICE = "cuda"  # or "cpu"

# Hyperparameter table
hp_table = [
    {"lr": 1e-4, "gamma": 0.99, "batch_size": 16, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 1e5},
    {"lr": 5e-5, "gamma": 0.98, "batch_size": 2, "eps_start": 1.0, "eps_end": 0.02, "eps_decay": 2e5},
    {"lr": 2e-4, "gamma": 0.97, "batch_size": 16, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 1.5e5},
    {"lr": 1e-4, "gamma": 0.96, "batch_size": 8, "eps_start": 0.9, "eps_end": 0.05, "eps_decay": 1e5},
    {"lr": 3e-4, "gamma": 0.99, "batch_size": 4, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 2e5},
    {"lr": 1.5e-4, "gamma": 0.995, "batch_size": 2, "eps_start": 1.0, "eps_end": 0.005, "eps_decay": 5e5},
    {"lr": 1e-5, "gamma": 0.99, "batch_size": 8, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 3e5},
    {"lr": 2e-4, "gamma": 0.96, "batch_size": 8, "eps_start": 1.0, "eps_end": 0.1, "eps_decay": 1e5},
    {"lr": 5e-5, "gamma": 0.99, "batch_size": 4, "eps_start": 1.0, "eps_end": 0.02, "eps_decay": 1e6},
    {"lr": 2e-4, "gamma": 0.995, "batch_size": 6, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 1e5},
]

# ---------------- Helpers ----------------
def make_env():
    """
    Create a Gymnasium Atari environment wrapped in Monitor + DummyVecEnv.
    Uses official ALE environment instead of custom wrapper.
    """
    env = gym.make(f"ALE/{ROM_NAME}-v5", render_mode=None)
    return DummyVecEnv([lambda: Monitor(env)])

def make_callbacks(eval_env, exp_dir, name_prefix, save_freq=50_000, eval_freq=25_000):
    ckpt_cb = CheckpointCallback(save_freq=save_freq, save_path=exp_dir, name_prefix=f"{name_prefix}_ckpt")
    eval_cb = EvalCallback(eval_env, best_model_save_path=exp_dir, log_path=exp_dir,
                           eval_freq=eval_freq, n_eval_episodes=EVAL_EPISODES, deterministic=True, render=False)
    return CallbackList([ckpt_cb, eval_cb])

def evaluate_save(model, eval_env, exp_dir, label, results):
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True)
    model_path = exp_dir / f"{label}_final.zip"
    model.save(str(model_path))
    results.append({"algo": label, "mean_reward": float(mean_reward), "std_reward": float(std_reward), "model_path": str(model_path)})
    with open(RESULTS_DIR / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[EVAL] {label}: mean={mean_reward:.2f} std={std_reward:.2f}")

def create_dqn_from_hp(hp, train_env, timesteps):
    try:
        model = DQN("CnnPolicy", train_env, learning_rate=hp["lr"], gamma=hp["gamma"], batch_size=hp["batch_size"],
                    exploration_initial_eps=hp["eps_start"], exploration_final_eps=hp["eps_end"],
                    exploration_fraction=min(1.0, hp["eps_decay"]/timesteps),
                    verbose=1, device=DEVICE)
        return model
    except Exception as e:
        warnings.warn(f"DQN creation failed ({e}), fallback to default CnnPolicy")
        return DQN("CnnPolicy", train_env, verbose=1, device=DEVICE)

# ---------------- Main Training ----------------
def train_all():
    results = []

    for idx, hp in enumerate(hp_table):
        print(f"\n=== Experiment {idx+1} ===")
        exp_dir = RESULTS_DIR / f"dqn_exp{idx+1}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        train_env = make_env()
        eval_env = make_env()

        model = create_dqn_from_hp(hp, train_env, TIMESTEPS)
        callbacks = make_callbacks(eval_env, exp_dir, f"dqn_{idx+1}")

        model.learn(total_timesteps=TIMESTEPS, callback=callbacks, tb_log_name=f"DQN_{idx+1}")
        evaluate_save(model, eval_env, exp_dir, f"dqn_{idx+1}", results)

        del model, train_env, eval_env
        gc.collect()

    print("All experiments finished. Results summary:", RESULTS_DIR / "results_summary.json")

if __name__ == "__main__":
    train_all()
