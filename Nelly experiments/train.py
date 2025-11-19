import os
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# --- Fix for "Namespace ALE not found" ---
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass

# Directories
MODEL_DIR = "models"
RESULT_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------- Helper to Create Fresh Env ----------------
def create_fresh_env(seed=42):
    env_id = "ALE/Breakout-v5"
    try:
        vec_env = make_atari_env(env_id, n_envs=1, seed=seed)
        env = VecFrameStack(vec_env, n_stack=4)
        return env
    except Exception as e:
        print(f"\nCRITICAL ERROR creating environment: {e}")
        raise e

# ---------------- Experiment Runner ----------------
def run_experiment(exp_id, config):
    print(f"\n==================================================")
    print(f"STARTING EXPERIMENT {exp_id}: {config['focus']}")
    print(f"  > Timesteps: {config['timesteps']}")
    print(f"  > LR: {config['lr']} | Batch: {config['batch_size']} | Buffer: {config['buffer_size']}")
    print(f"==================================================")

    try:
        env = create_fresh_env(seed=42)
    except:
        return None

    log_name = f"exp_{exp_id}_{config['policy']}"
    log_dir = os.path.join(RESULT_DIR, log_name)
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv"])

    try:
        model = DQN(
            policy=config['policy'],
            env=env,
            learning_rate=config['lr'],
            gamma=config['gamma'],
            batch_size=config['batch_size'],
            buffer_size=config['buffer_size'],
            learning_starts=config['learning_starts'],
            target_update_interval=config['target_update'],
            exploration_initial_eps=config['epsilon_start'],
            exploration_final_eps=config['epsilon_end'],
            exploration_fraction=config['epsilon_decay'],
            train_freq=4,
            gradient_steps=1,
            verbose=0,
            seed=42,
            optimize_memory_usage=False
        )
        model.set_logger(new_logger)

        # Train
        model.learn(total_timesteps=config['timesteps'])

        # Save Model
        save_path = os.path.join(MODEL_DIR, f"model_exp_{exp_id}.zip")
        model.save(save_path)
        print(f"-> Model saved to {save_path}")

        # Evaluate (5 episodes)
        print("-> Evaluating (5 Episodes)...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)

        # Auto-Comment
        comment = "Average."
        if mean_reward < 2.0: comment = "Failed / Needs more time."
        elif mean_reward > 15.0: comment = "Great Performance!"
        elif mean_reward > 10.0: comment = "Good (Learned basics)."
        if config['policy'] == 'MlpPolicy': comment = "MLP Limitation."

        env.close()
        del model
        del env

        return {
            "Exp ID": exp_id,
            "Focus": config['focus'],
            "Timesteps": config['timesteps'],
            "LR": config['lr'],
            "Batch": config['batch_size'],
            "Buffer": config['buffer_size'],
            "Mean Reward": round(mean_reward, 2),
            "Std Reward": round(std_reward, 2),
            "Auto-Comment": comment
        }

    except Exception as e:
        print(f"!!! EXPERIMENT {exp_id} FAILED: {e}")
        if 'env' in locals(): env.close()
        return None

# ---------------- Main ----------------
def main():
    experiments_config = [
        {'focus': 'Standard Baseline', 'policy': 'CnnPolicy', 'timesteps': 5000,
         'lr': 1e-4, 'gamma': 0.99, 'batch_size': 32, 'buffer_size': 100000, 
         'learning_starts': 10000, 'target_update': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},

        {'focus': 'MLP Check', 'policy': 'MlpPolicy', 'timesteps': 5000,
         'lr': 1e-4, 'gamma': 0.99, 'batch_size': 32, 'buffer_size': 50000, 
         'learning_starts': 5000, 'target_update': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},

        {'focus': 'Aggressive LR (2.5e-4)', 'policy': 'CnnPolicy', 'timesteps': 5000,
         'lr': 2.5e-4, 'gamma': 0.99, 'batch_size': 32, 'buffer_size': 100000, 
         'learning_starts': 10000, 'target_update': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},

        {'focus': 'Precise LR (5e-5)', 'policy': 'CnnPolicy', 'timesteps': 5000,
         'lr': 5e-5, 'gamma': 0.99, 'batch_size': 32, 'buffer_size': 100000, 
         'learning_starts': 10000, 'target_update': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},

        {'focus': 'Small Buffer (10k)', 'policy': 'CnnPolicy', 'timesteps': 5000,
         'lr': 1e-4, 'gamma': 0.99, 'batch_size': 32, 'buffer_size': 10000, 
         'learning_starts': 1000, 'target_update': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},

        {'focus': 'Large Batch (128)', 'policy': 'CnnPolicy', 'timesteps': 5000,
         'lr': 1e-4, 'gamma': 0.99, 'batch_size': 128, 'buffer_size': 100000, 
         'learning_starts': 10000, 'target_update': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},

        {'focus': 'Stable Target (2000)', 'policy': 'CnnPolicy', 'timesteps': 5000,
         'lr': 1e-4, 'gamma': 0.99, 'batch_size': 32, 'buffer_size': 100000, 
         'learning_starts': 10000, 'target_update': 2000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},

        {'focus': 'More Explore (20% decay)', 'policy': 'CnnPolicy', 'timesteps': 5000,
         'lr': 1e-4, 'gamma': 0.99, 'batch_size': 32, 'buffer_size': 100000, 
         'learning_starts': 10000, 'target_update': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.2},

        {'focus': 'DeepMind Style', 'policy': 'CnnPolicy', 'timesteps': 5000,
         'lr': 1e-4, 'gamma': 0.99, 'batch_size': 32, 'buffer_size': 200000, 
         'learning_starts': 20000, 'target_update': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},

        {'focus': '*** CHAMPION MODEL ***', 'policy': 'CnnPolicy', 'timesteps': 5000,
         'lr': 1e-4, 'gamma': 0.99, 'batch_size': 32, 'buffer_size': 100000, 
         'learning_starts': 10000, 'target_update': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
    ]

    results_table = []
    for i, config in enumerate(experiments_config):
        result = run_experiment(i+1, config)
        if result:
            results_table.append(result)

    df = pd.DataFrame(results_table)
    summary_path = os.path.join(RESULT_DIR, "optimized_comparison.csv")
    df.to_csv(summary_path, index=False)

    print("\n" + "="*60)
    print("OPTIMIZED RESULTS TABLE")
    print("="*60)
    cols = ['Exp ID', 'Focus', 'Timesteps', 'Mean Reward', 'Std Reward', 'Auto-Comment']
    print(df[cols].to_string(index=False))
    print("\nFull table saved to:", summary_path)

if __name__ == "__main__":
    main()
