import os
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import matplotlib.pyplot as plt


# Create folders to save models and results
MODEL_DIR = "model"
RESULT_DIR = "result"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


class TrainingLogger(BaseCallback):
    """Custom callback for logging rewards and episode lengths."""

    def __init__(self, verbose=0):
        super(TrainingLogger, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
        return True

    def save_logs(self, file_prefix="train_log"):
        df = pd.DataFrame({
            'episode_reward': self.episode_rewards,
            'episode_length': self.episode_lengths
        })
        csv_path = os.path.join(RESULT_DIR, f"{file_prefix}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Training log saved to {csv_path}")

        # Plot reward trends
        plt.figure(figsize=(12, 6))
        plt.plot(df['episode_reward'], label='Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward Trend During Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(RESULT_DIR, f"{file_prefix}_reward_plot.png"))
        plt.close()
        print(f"Reward plot saved to {RESULT_DIR}")


def train_dqn(env_id, policy_type, hyperparams, total_timesteps=100_000):
    """Train DQN agent with specified policy and hyperparameters."""
    
    # Initialize environment with a Monitor wrapper to get info for logging
    env = Monitor(gym.make(env_id))

    # Set up logging directory for Stable Baselines3 logger
    log_dir = os.path.join(RESULT_DIR, f"{policy_type}_logs")
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    model = DQN(
        policy=policy_type,
        env=env,
        learning_rate=hyperparams['lr'],
        gamma=hyperparams['gamma'],
        batch_size=hyperparams['batch_size'],
        exploration_initial_eps=hyperparams['epsilon_start'],
        exploration_final_eps=hyperparams['epsilon_end'],
        exploration_fraction=hyperparams['epsilon_decay'],
        tensorboard_log=log_dir,
        verbose=1,
        seed=42,
        train_freq=1,
        target_update_interval=500,
        buffer_size=10000,
        optimize_memory_usage=False  # Fix for replay buffer error
    )
    model.set_logger(new_logger)

    logger = TrainingLogger()
    model.learn(total_timesteps=total_timesteps, callback=logger)

    # Save model
    model_path = os.path.join(MODEL_DIR, f"dqn_{policy_type.lower()}_model.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save logs from callback
    logger.save_logs(file_prefix=f"dqn_{policy_type.lower()}_train_log")

    env.close()

    return model


def main():
    env_id = "CartPole-v1"
    
    # Define hyperparameter sets for experimentation
    hyperparameter_sets = [
        {'lr': 1e-3, 'gamma': 0.99, 'batch_size': 32, 'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay': 0.1},
        {'lr': 5e-4, 'gamma': 0.98, 'batch_size': 64, 'epsilon_start': 1.0, 'epsilon_end': 0.1, 'epsilon_decay': 0.2},
        {'lr': 1e-4, 'gamma': 0.95, 'batch_size': 128, 'epsilon_start': 0.9, 'epsilon_end': 0.05, 'epsilon_decay': 0.05},
        {'lr': 5e-4, 'gamma': 0.99, 'batch_size': 32, 'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_decay': 0.1},
        {'lr': 1e-3, 'gamma': 0.97, 'batch_size': 64, 'epsilon_start': 0.95, 'epsilon_end': 0.05, 'epsilon_decay': 0.15},
        {'lr': 2e-4, 'gamma': 0.99, 'batch_size': 128, 'epsilon_start': 1.0, 'epsilon_end': 0.1, 'epsilon_decay': 0.05},
        {'lr': 1e-4, 'gamma': 0.96, 'batch_size': 32, 'epsilon_start': 0.9, 'epsilon_end': 0.01, 'epsilon_decay': 0.2},
        {'lr': 5e-3, 'gamma': 0.97, 'batch_size': 64, 'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay': 0.1},
        {'lr': 1e-3, 'gamma': 0.95, 'batch_size': 128, 'epsilon_start': 0.95, 'epsilon_end': 0.1, 'epsilon_decay': 0.2},
        {'lr': 2e-3, 'gamma': 0.99, 'batch_size': 32, 'epsilon_start': 1.0, 'epsilon_end': 0.05, 'epsilon_decay': 0.1}
    ]

    # Dataframe to record hyperparameter results
    experiment_log = []

    # Train DQN agents with MLPPolicy (best for CartPole usually) and log results
    for idx, params in enumerate(hyperparameter_sets):
        print(f"\n---- Experiment {idx + 1} with MLPPolicy ----")
        model = train_dqn(env_id, 'MlpPolicy', params, total_timesteps=50000)

        # Evaluate trained model
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print(f"Evaluation over 10 episodes: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

        experiment_log.append({
            "Experiment": idx + 1,
            **params,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "policy": "MlpPolicy"
        })

    # Save experiment log to CSV
    df_log = pd.DataFrame(experiment_log)
    log_path = os.path.join(RESULT_DIR, "hyperparameter_experiments_log.csv")
    df_log.to_csv(log_path, index=False)
    print(f"\nHyperparameter experiments recorded in {log_path}")


if __name__ == "__main__":
    main()
