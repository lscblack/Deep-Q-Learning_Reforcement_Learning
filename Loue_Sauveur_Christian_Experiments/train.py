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
        # 1. THE INSTANT REFLEX (Lowest steps, Highest LR)
        {'focus': 'Micro Sprint 1', 'policy': 'CnnPolicy', 'timesteps': 10000,
        'lr': 1.0e-3, 'gamma': 0.90, 'batch_size': 16, 'buffer_size': 5000, 
        'learning_starts': 500, 'target_update': 200,
        'epsilon_start': 1.0, 'epsilon_end': 0.1, 'epsilon_decay': 0.5},

        # 2. THE RAPID BATCH
        {'focus': 'Micro Sprint 2', 'policy': 'CnnPolicy', 'timesteps': 15000,
        'lr': 9.0e-4, 'gamma': 0.91, 'batch_size': 24, 'buffer_size': 7500, 
        'learning_starts': 800, 'target_update': 300,
        'epsilon_start': 0.95, 'epsilon_end': 0.09, 'epsilon_decay': 0.45},

        # 3. THE STANDARD SPRINT
        {'focus': 'Micro Sprint 3', 'policy': 'CnnPolicy', 'timesteps': 20000,
        'lr': 8.0e-4, 'gamma': 0.92, 'batch_size': 32, 'buffer_size': 10000, 
        'learning_starts': 1000, 'target_update': 400,
        'epsilon_start': 0.9, 'epsilon_end': 0.08, 'epsilon_decay': 0.4},

        # 4. THE MID-RANGE CHECK
        {'focus': 'Micro Sprint 4', 'policy': 'CnnPolicy', 'timesteps': 25000,
        'lr': 7.0e-4, 'gamma': 0.93, 'batch_size': 40, 'buffer_size': 12500, 
        'learning_starts': 1500, 'target_update': 500,
        'epsilon_start': 0.85, 'epsilon_end': 0.07, 'epsilon_decay': 0.35},

        # 5. THE BALANCED RUN
        {'focus': 'Micro Sprint 5', 'policy': 'CnnPolicy', 'timesteps': 30000,
        'lr': 6.0e-4, 'gamma': 0.94, 'batch_size': 48, 'buffer_size': 15000, 
        'learning_starts': 2000, 'target_update': 600,
        'epsilon_start': 0.8, 'epsilon_end': 0.06, 'epsilon_decay': 0.3},

        # 6. THE STABILITY TEST
        {'focus': 'Micro Sprint 6', 'policy': 'CnnPolicy', 'timesteps': 35000,
        'lr': 5.0e-4, 'gamma': 0.95, 'batch_size': 64, 'buffer_size': 17500, 
        'learning_starts': 2500, 'target_update': 800,
        'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay': 0.25},

        # 7. THE WIDE BATCH
        {'focus': 'Micro Sprint 7', 'policy': 'CnnPolicy', 'timesteps': 40000,
        'lr': 4.0e-4, 'gamma': 0.96, 'batch_size': 96, 'buffer_size': 50000, 
        'learning_starts': 3000, 'target_update': 1000,
        'epsilon_start': 1.0, 'epsilon_end': 0.04, 'epsilon_decay': 0.2},

        # 8. THE DEEP MEMORY
        {'focus': 'Micro Sprint 8', 'policy': 'CnnPolicy', 'timesteps': 55000,
        'lr': 3.0e-4, 'gamma': 0.97, 'batch_size': 128, 'buffer_size': 22500, 
        'learning_starts': 4000, 'target_update': 1500,
        'epsilon_start': 1.0, 'epsilon_end': 0.03, 'epsilon_decay': 0.15},

        # 9. THE FUTURE LOOKER (High Gamma)
        {'focus': 'Micro Sprint 9', 'policy': 'CnnPolicy', 'timesteps': 60000,
        'lr': 2.0e-4, 'gamma': 0.98, 'batch_size': 256, 'buffer_size': 25000, 
        'learning_starts': 5000, 'target_update': 2000,
        'epsilon_start': 1.0, 'epsilon_end': 0.02, 'epsilon_decay': 0.1},

        # 10. THE MINI-CHAMP (Best chance of learning in short time)
        {'focus': 'Micro Sprint 10', 'policy': 'CnnPolicy', 'timesteps': 70000,
        'lr': 1.0e-4, 'gamma': 0.99, 'batch_size': 512, 'buffer_size': 30000, 
        'learning_starts': 6000, 'target_update': 2500,
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
