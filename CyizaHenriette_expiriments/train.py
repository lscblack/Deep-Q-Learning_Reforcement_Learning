#!/usr/bin/env python3
"""
Cyiza Henriette - Atari DQN Experiments (10 Hyperparameter Tests)
Trains DQN on ALE/IceHockey-v5 using Stable Baselines3
Simplified & corrected to avoid TensorBoard and memory issues.
"""

import os
import time
import json
import warnings
from pathlib import Path
import gymnasium as gym
import ale_py

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.atari_wrappers import AtariWrapper


# ---------------- CONFIG ----------------
ENV_ID = "ALE/IceHockey-v5"
MEMBER_NAME = "Cyiza Henriette"
RESULTS_DIR = Path("./cyiza_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTEPS = 60_000        # safe for your RAM & GPU-less machine
EVAL_EPISODES = 3
SEED = 42


# ---------------- 10 EXPERIMENTS ----------------
my_hp_table = [
    {"lr": 8e-4, "gamma": 0.90, "batch_size": 64, "eps_start": 1.0, "eps_end": 0.02, "eps_decay": 5e4, "policy": "CnnPolicy", "buffer_size": 20000, "description": "Aggressive learner"},
    {"lr": 5e-6, "gamma": 0.999, "batch_size": 32, "eps_start": 1.0, "eps_end": 0.05, "eps_decay": 8e5, "policy": "CnnPolicy", "buffer_size": 20000, "description": "Conservative slow learner"},
    {"lr": 3e-4, "gamma": 0.98, "batch_size": 64, "eps_start": 1.0, "eps_end": 0.2, "eps_decay": 1e5, "policy": "CnnPolicy", "buffer_size": 20000, "description": "High exploration"},
    {"lr": 1e-4, "gamma": 0.99, "batch_size": 32, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 5e4, "policy": "MlpPolicy", "buffer_size": 20000, "description": "Fast greedy collapse"},
    {"lr": 2e-4, "gamma": 0.995, "batch_size": 128, "eps_start": 1.0, "eps_end": 0.02, "eps_decay": 3e5, "policy": "CnnPolicy", "buffer_size": 20000, "description": "Stable learner"},
    {"lr": 1.5e-4, "gamma": 0.97, "batch_size": 8, "eps_start": 1.0, "eps_end": 0.05, "eps_decay": 1e6, "policy": "CnnPolicy", "buffer_size": 20000, "description": "Noisy small batch"},
    {"lr": 5e-4, "gamma": 0.99, "batch_size": 64, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 2e6, "policy": "CnnPolicy", "buffer_size": 20000, "description": "Long exploration slow decay"},
    {"lr": 1e-4, "gamma": 0.96, "batch_size": 64, "eps_start": 1.0, "eps_end": 0.02, "eps_decay": 8e4, "policy": "MlpPolicy", "buffer_size": 20000, "description": "Short memory"},
    {"lr": 3e-4, "gamma": 0.999, "batch_size": 64, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 4e5, "policy": "CnnPolicy", "buffer_size": 20000, "description": "Long horizon"},
    {"lr": 2e-4, "gamma": 0.92, "batch_size": 32, "eps_start": 0.8, "eps_end": 0.05, "eps_decay": 5e4, "policy": "MlpPolicy", "buffer_size": 20000, "description": "Semi-greedy short-term"},
]


# ---------------- HELPERS ----------------
def register_ale():
    gym.register_envs(ale_py)

def make_env(env_id):
    """Create single wrapped Atari environment."""
    def _make():
        env = gym.make(env_id)
        env = Monitor(env)
        try:
            env = AtariWrapper(env)
        except:
            pass
        return env

    venv = DummyVecEnv([_make])
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=3)
    return venv


def create_dqn(hp, env, timesteps):
    """Make DQN model (fallback-safe)."""

    exploration_fraction = min(1.0, hp["eps_decay"] / max(1, timesteps))

    try:
        model = DQN(
            hp["policy"],
            env,
            learning_rate=hp["lr"],
            buffer_size=hp["buffer_size"],
            batch_size=hp["batch_size"],
            gamma=hp["gamma"],
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_initial_eps=hp["eps_start"],
            exploration_final_eps=hp["eps_end"],
            exploration_fraction=exploration_fraction,
            verbose=1
        )
        return model
    except:
        print("Policy failed, using CnnPolicy fallback...")
        return DQN(
            "CnnPolicy",
            env,
            learning_rate=hp["lr"],
            buffer_size=hp["buffer_size"],
            batch_size=hp["batch_size"],
            gamma=hp["gamma"],
            verbose=1
        )


def create_callbacks(eval_env, exp_dir):
    return CallbackList([
        CheckpointCallback(save_freq=50_000, save_path=str(exp_dir), name_prefix="ckpt"),
        EvalCallback(eval_env, best_model_save_path=str(exp_dir), eval_freq=25_000, deterministic=True)
    ])


# ---------------- MAIN TRAINING ----------------
def train_all_experiments():
    register_ale()
    results = []

    print("\n==== TRAINING STARTED ====\n")

    for idx, hp in enumerate(my_hp_table):
        exp_id = idx + 1
        exp_dir = RESULTS_DIR / f"exp_{exp_id}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Experiment {exp_id}/10 ---")
        print(hp["description"])
        print(hp)

        train_env = make_env(ENV_ID)
        eval_env = make_env(ENV_ID)

        model = create_dqn(hp, train_env, TIMESTEPS)
        callbacks = create_callbacks(eval_env, exp_dir)

        start = time.time()

        try:
            model.learn(total_timesteps=TIMESTEPS, callback=callbacks)
            duration = time.time() - start

            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True
            )

            model_path = exp_dir / "final_model.zip"
            model.save(str(model_path))

            results.append({
                "experiment": exp_id,
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "model_path": str(model_path),
                "hyperparameters": hp
            })

            print(f"✓ Experiment {exp_id} complete → Reward {mean_reward:.2f}")

        except Exception as e:
            print(f"✗ Experiment {exp_id} failed:", e)
            results.append({"experiment": exp_id, "status": "failed", "error": str(e)})

        train_env.close()
        eval_env.close()

        with open(RESULTS_DIR / "results_summary.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n TRAINING FINISHED\n")


if __name__ == "__main__":
    train_all_experiments()
