import os
import gymnasium as gym
import numpy as np
import pandas as pd
import ale_py
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper 
env = gym.make("ALE/Breakout-v5")
env = AtariWrapper(env)


model = DQN("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Directories
# NOTE: These directories will be created relative to your current working directory
MODEL_DIR = "models"
RESULT_DIR = "results"
env = AtariWrapper(env)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------- Helper to Create Fresh Env (FIXED FOR STABILITY) ----------------
# This uses the stable structure (DummyVecEnv + explicit wrappers) which avoids the dtype error.
def create_fresh_env(seed=42):
    env_id = "ALE/Breakout-v5"
    
    def _make_env():
        # 1. Standard environment creation
        env = gym.make(env_id)
        # 2. Monitor wrapper must be placed immediately after gym.make
        env = Monitor(env)
        # 3. Atari wrapper handles max pooling, frame stacking (optional but common), etc.
        env = AtariWrapper(env)
        return env

    try:
        # 1. Use DummyVecEnv for single environment
        vec_env = DummyVecEnv([_make_env])
        # 2. Transpose images (NHWC -> NCHW required by PyTorch/SB3 CNNs)
        vec_env = VecTransposeImage(vec_env) 
        # 3. Stack frames (n_stack=4 is standard for Breakout)
        env = VecFrameStack(vec_env, n_stack=4)
        return env
    except Exception as e:
        print(f"\nCRITICAL ERROR creating environment: {e}")
        raise e

# ---------------- Experiment Runner (FIXED) ----------------
def run_experiment(exp_id, config):
    print(f"\n==================================================")
    print(f"STARTING EXPERIMENT {exp_id}: {config['focus']}")
    print(f"  > Timesteps: {config['timesteps']}")
    print(f"  > LR: {config['lr']} | Batch: {config['batch_size']} | Buffer: {config['buffer_size']}")
    print(f"==================================================")

    try:
        # Recreating the environment for each run ensures isolation
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
            # FIX: Corrected parameter name to 'target_update_interval'
            target_update_interval=config['target_update_interval'], 
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

        # FIX: Ensure all dictionary keys match the 'cols' list in main()
        return {
            "Exp ID": exp_id,
            "Focus": config['focus'],
            "Timesteps": config['timesteps'],
            "LR": config['lr'],
            "Gamma": config['gamma'],
            "Batch": config['batch_size'],
            "Buffer": config['buffer_size'],
            "Learning Starts": config['learning_starts'],
            "Target Update": config['target_update_interval'],
            "Eps Decay Fraction": config['epsilon_decay'],
            "Mean Reward": round(mean_reward, 2),
            "Std Reward": round(std_reward, 2),
            "Auto-Comment": comment,
            "Policy": config['policy'], 
            "Epsilon Start": config['epsilon_start'],
            "Epsilon End": config['epsilon_end'],
        }

    except Exception as e:
        print(f"!!! EXPERIMENT {exp_id} FAILED: {e}")
        if 'env' in locals(): env.close()
        # Ensure FAILED experiments still return core data to prevent KeyError
        return {
            "Exp ID": exp_id,
            "Focus": config['focus'],
            "Policy": config['policy'],
            "LR": config['lr'],
            "Buffer": config['buffer_size'],
            "Learning Starts": config['learning_starts'],
            "Target Update": config['target_update_interval'],
            "Mean Reward": np.nan,
            "Std Reward": np.nan,
            "Auto-Comment": "CRASHED",
            "Timesteps": config['timesteps'],
            "Gamma": config['gamma'],
            "Batch": config['batch_size'],
            "Eps Decay Fraction": config['epsilon_decay'],
            "Epsilon Start": config['epsilon_start'],
            "Epsilon End": config['epsilon_end'],
        }


# ---------------- Main (FINAL FIXED CONFIG) ----------------
def main():
    experiments_config = [
        # 1: Baseline short (quick-check)
        {'focus': 'Baseline Short (quick-check)', 'policy': 'CnnPolicy', 'timesteps': 5000,
         'lr': 1e-4, 'gamma': 0.99, 'batch_size': 64, 'buffer_size': 5000,
         'learning_starts': 1000, 'target_update_interval': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay': 0.2},

        # 2: Optimized short-to-mid (Exp6-tuned)
        {'focus': 'Optimized Fast (Exp6-tuned)', 'policy': 'CnnPolicy', 'timesteps': 5000,
         'lr': 3e-4, 'gamma': 0.95, 'batch_size': 32, 'buffer_size': 10000,
         'learning_starts': 1000, 'target_update_interval': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay': 0.1},

        # 3: Longer stable training
        {'focus': 'Longer Stable (50k)', 'policy': 'CnnPolicy', 'timesteps': 150000,
         'lr': 1e-4, 'gamma': 0.90, 'batch_size': 32, 'buffer_size': 50000,
         'learning_starts': 5000, 'target_update_interval': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.2},

        # 4: Aggressive Start (MLP test)
        {'focus': 'Aggressive Start (MLP test)', 'policy': 'MlpPolicy', 'timesteps': 5000,
         'lr': 2e-4, 'gamma': 0.99, 'batch_size': 64, 'buffer_size': 100000,
         'learning_starts': 500, 'target_update_interval': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},

        # 5: Mid-length stable
        {'focus': 'Mid Stable (20k)', 'policy': 'CnnPolicy', 'timesteps': 20000,
         'lr': 1.5e-4, 'gamma': 0.96, 'batch_size': 32, 'buffer_size': 20000,
         'learning_starts': 2000, 'target_update_interval': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.02, 'epsilon_decay': 0.15},

        # 6: High gamma, longer horizon
        {'focus': 'High Gamma Long Horizon', 'policy': 'CnnPolicy', 'timesteps': 40000,
         'lr': 1e-4, 'gamma': 0.995, 'batch_size': 32, 'buffer_size': 40000,
         'learning_starts': 2000, 'target_update_interval': 2000,
         'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay': 0.2},

        # 7: Short buffer, aggressive epsilon decay
        {'focus': 'Short Buffer Fast Decay', 'policy': 'CnnPolicy', 'timesteps': 10000,
         'lr': 2e-4, 'gamma': 0.97, 'batch_size': 64, 'buffer_size': 5000,
         'learning_starts': 1000, 'target_update_interval': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.5},

        # 8: Medium buffer, delayed target update
        {'focus': 'Delayed Target Update (5k)', 'policy': 'CnnPolicy', 'timesteps': 15000,
         'lr': 1e-4, 'gamma': 0.99, 'batch_size': 32, 'buffer_size': 10000,
         'learning_starts': 1000, 'target_update_interval': 5000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},

        # 9: Balanced optimizer, medium LR
        {'focus': 'Balanced Optimizer', 'policy': 'CnnPolicy', 'timesteps': 20000,
         'lr': 1.5e-4, 'gamma': 0.995, 'batch_size': 32, 'buffer_size': 20000,
         'learning_starts': 1000, 'target_update_interval': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.02, 'epsilon_decay': 0.15},

        # 10: Fastest decay, test exploration
        {'focus': 'Fastest Decay (0.5)', 'policy': 'CnnPolicy', 'timesteps': 5000,
         'lr': 1e-4, 'gamma': 0.99, 'batch_size': 64, 'buffer_size': 50000,
         'learning_starts': 1000, 'target_update_interval': 1000,
         'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.5},
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
    # The columns list must use the exact string keys returned by run_experiment
    cols = ['Exp ID', 'Focus', 'Policy', 'LR', 'Gamma', 'Buffer', 'Learning Starts', 'Target Update', 'Mean Reward', 'Std Reward', 'Auto-Comment']
    print(df[cols].to_string(index=False))
    print("\nFull table saved to:", summary_path)

if __name__ == "__main__":
    main()
