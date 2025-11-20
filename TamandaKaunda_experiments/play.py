import gymnasium as gym
import ale_py
import numpy as np
import os
import traceback
import time
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper

# ---------------- CONFIGURATION ----------------
ENV_ID = "ALE/IceHockey-v5"

# --- REQUIRED CORRECTION ---
# Set to the path of your BEST model: Experiment 6 (Score: -1.00)
# NOTE: Adjusted directory structure to match your results: TamandaKaunda_results
MODEL_PATH = Path("TamandaKaunda_results") / "exp_6" / "final_model.zip"

NUM_EPISODES = 10  # Number of episodes to play
# -----------------------------------------------


# --- CORRECTION: register_ale is OBSOLETE and has been removed ---
# def register_ale():
#     gym.register_envs(ale_py)

def make_play_env(env_id):
    """Create the environment exactly as it was during training, with rendering."""
    def _make():
        # Set render_mode="human" for the GUI display
        env = gym.make(env_id, render_mode="human")
        #env = Monitor(env)
        try:
            env = AtariWrapper(env)
        except:
            pass
        return env
    
    venv = DummyVecEnv([_make])
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=3)  # MUST match train.py (you used 3)
    
    return venv

def main():
    print("=" * 70)
    print("DQN AGENT GAMEPLAY")
    print("=" * 70)
    print(f"Loading model: {MODEL_PATH}")
    
    # Load the trained model
    try:
        model = DQN.load(str(MODEL_PATH))
        print(" Model loaded successfully!\n")
    except FileNotFoundError:
        print(f" Error: Model not found at {MODEL_PATH}\n")
        print("Available models:")
        
        # Check available paths in the correct directory (TamandaKaunda_results)
        for i in range(1, 11):
            path_check = Path("TamandaKaunda_results") / f"exp_{i}" / "final_model.zip"
            if path_check.exists():
                print(f" Experiment {i}: {path_check}")
        return
    except Exception as e:
        print(f" Error loading model: {e}")
        return

    # NOTE: register_ale() call has been removed from here as well.
    
    # Create environment
    print(f"Environment: {ENV_ID}")
    print(f"Episodes to play: {NUM_EPISODES}")
    print("=" * 70)
    
    env = make_play_env(ENV_ID)
    
    # Storage for results
    episode_rewards = []
    episode_lengths = []
    
    print("\n🎮 Starting gameplay...\n")
    
    try:
        for episode in range(NUM_EPISODES):
            # Reset environment
            obs = env.reset() # VecEnv in newer SB3 returns obs, info at reset
            
            episode_reward = 0.0
            episode_steps = 0
            done = False
            
            print(f"Playing Episode {episode + 1}/{NUM_EPISODES}...", end=" ", flush=True)
            
            # Play until episode ends
            while not done:
                # Get action from model (deterministic = GreedyQPolicy)
                action, _ = model.predict(obs, deterministic=True)
                
                # Take step in environment
                obs, reward, done_array, info = env.step(action)
                
                # Extract scalar from array (VecEnv wraps everything in arrays)
                reward_value = float(reward[0])
                done_value = bool(done_array[0])
                
                episode_reward += reward_value
                episode_steps += 1
                
                # Check if episode is done
                if done_value:
                    done = True
            
            # Store results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            
            # Print episode results
            print(f"Reward: {episode_reward:>7.2f} | Steps: {episode_steps:>4d}")
    
    except KeyboardInterrupt:
        print("\n\n Gameplay interrupted by user (Ctrl+C)")
    
    except Exception as e:
        print(f"\n\n Error during gameplay: {e}")
        traceback.print_exc()
    
    finally:
        # Clean up
        env.close()
    
    # Print summary statistics
    if episode_rewards:
        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"Episodes Completed:      {len(episode_rewards)}")
        print(f"Average Reward:          {np.mean(episode_rewards):>7.2f}")
        print(f"Std Dev Reward:          {np.std(episode_rewards):>7.2f}")
        print(f"Best Reward:             {np.max(episode_rewards):>7.2f}")
        print(f"Worst Reward:            {np.min(episode_rewards):>7.2f}")
        print(f"Average Steps:           {np.mean(episode_lengths):>7.0f}")
        print("=" * 70)
        
        # Interpretation
        avg_reward = np.mean(episode_rewards)
        print("\n💡 INTERPRETATION:")
        if avg_reward > 0:
            print("  Your agent is winning! Great job!")
        elif avg_reward >= -1.0:
            # -1.0 is your current best score, so this is the top tier.
            print("  Your agent is highly competitive and nearly breaking even! Excellent result!")
        elif avg_reward > -5:
            print(" Your agent is competitive but needs more training.")
        else:
            print(" Your agent is losing most games. This is expected with")
            print("   only 60k timesteps on Ice Hockey (needs 200k+).")
        
        print("\n📹 For your video: Record the screen while this script runs!")
        print("=" * 70)
    else:
        print("\n No episodes completed.")

if __name__ == "__main__":
    main()