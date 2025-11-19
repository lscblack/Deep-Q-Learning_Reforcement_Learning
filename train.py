#!/usr/bin/env python3
"""
Tamanda Kaunda - Atari DQN Experiments (10 Hyperparameter Tests)
Trains DQN on ALE/IceHockey-v5 using Stable Baselines3
Unique Hyperparameter Tuning
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
MEMBER_NAME = "Tamanda Kaunda" 
RESULTS_DIR = Path("./TamandaKaunda_results") 
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESTEPS = 60_000         # Total steps for training
EVAL_EPISODES = 3
SEED = 42


# The structure and hyperparameters 
my_hp_table = [
    {"lr": 4e-4, "gamma": 0.995, "batch_size": 64, "eps_start": 1.0, "eps_end": 0.02, "eps_decay": 3e5, "policy": "CnnPolicy", "buffer_size": 20000, "description": "1. Moderate Aggressive LR, Long Exploration"},
    {"lr": 2e-4, "gamma": 0.80, "batch_size": 32, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 1e5, "policy": "CnnPolicy", "buffer_size": 20000, "description": "2. Extreme Short Horizon (Gamma 0.80)"},
    {"lr": 1e-4, "gamma": 0.99, "batch_size": 128, "eps_start": 0.7, "eps_end": 0.05, "eps_decay": 5e4, "policy": "CnnPolicy", "buffer_size": 50000, "description": "3. Low Initial Epsilon (0.7), Fast Decay"},
    {"lr": 6e-5, "gamma": 0.96, "batch_size": 64, "eps_start": 1.0, "eps_end": 0.25, "eps_decay": 2e5, "policy": "CnnPolicy", "buffer_size": 50000, "description": "4. Sustained High Randomness (Eps End 0.25)"},
    {"lr": 1e-4, "gamma": 0.999, "batch_size": 32, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 2.5e5, "policy": "MlpPolicy", "buffer_size": 20000, "description": "5. MlpPolicy Test 3 (Expected Failure)"},
    {"lr": 3e-4, "gamma": 0.97, "batch_size": 64, "eps_start": 0.95, "eps_end": 0.005, "eps_decay": 4e5, "policy": "CnnPolicy", "buffer_size": 20000, "description": "6. Precision Exploiter (Low Eps End 0.005)"},
    {"lr": 5e-5, "gamma": 0.99999, "batch_size": 128, "eps_start": 1.0, "eps_end": 0.05, "eps_decay": 5e5, "policy": "CnnPolicy", "buffer_size": 20000, "description": "7. Near Undiscounted Future (Max Gamma)"},
    {"lr": 2e-4, "gamma": 0.99, "batch_size": 256, "eps_start": 1.0, "eps_end": 0.02, "eps_decay": 1e6, "policy": "CnnPolicy", "buffer_size": 20000, "description": "8. Large Batch, Smooth Ultra-Slow Decay"},
    {"lr": 8e-5, "gamma": 0.98, "batch_size": 32, "eps_start": 1.0, "eps_end": 0.01, "eps_decay": 2e5, "policy": "MlpPolicy", "buffer_size": 20000, "description": "9. MlpPolicy Test 4 (Expected Failure)"},
    {"lr": 4e-4, "gamma": 0.995, "batch_size": 64, "eps_start": 1.0, "eps_end": 0.02, "eps_decay": 300000.0, "policy": "CnnPolicy", "buffer_size": 10000, "description": "10. Final Optimized Run: Best LR/Gamma with Smaller Buffer"}
]


# ---------------- HELPERS (Unchanged) ----------------
# The rest of the helper functions (register_ale, make_env, create_dqn, create_callbacks) 
# and the main training loop (train_all_experiments) remain unchanged from the original. 
# They are included below for completeness.

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


# ---------------- MAIN TRAINING (Unchanged) ----------------
def train_all_experiments():
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

            # Save the final model in case it wasn't the 'best' model checkpoint
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

        # Update the overall summary file after each experiment
        with open(RESULTS_DIR / "results_summary.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n TRAINING FINISHED\n")


if __name__ == "__main__":
    train_all_experiments()