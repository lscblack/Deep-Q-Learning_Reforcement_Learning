import gymnasium as gym
import ale_py
import numpy as np
import pandas as pd
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper
import os
os.environ["SDL_VIDEODRIVER"] = "x11"
os.environ["SDL_RENDER_DRIVER"] = "software"

# --- CONFIGURATION ---
CSV_PATH = "results/optimized_comparison.csv"
MODEL_DIR_PATTERN = "models/model_exp_{}.zip" # We will fill in the {} with the ID
ENV_ID = "ALE/Breakout-v5"
NUM_EPISODES = 6

def get_best_experiment_id(csv_file):
    """Reads the CSV and returns the ID of the experiment with the highest Mean Reward."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Could not find results file at: {csv_file}")

    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Clean column names (remove spaces) just in case
    df.columns = df.columns.str.strip()
    
    df['Reward_Range'] = df['Mean Reward'] - df['Std Reward']
    best_row = df.loc[df['Reward_Range'].idxmax()]

    exp_id = int(best_row['Exp ID'])
    score = best_row['Mean Reward']
    focus = best_row['Focus']
    
    print(f"🏆 BEST MODEL FOUND:")
    print(f"   - Experiment ID: {exp_id}")
    print(f"   - Strategy:      {focus}")
    print(f"   - Mean Reward:   {score}")
    
    return exp_id

def register_ale():
    gym.register_envs(ale_py)

def make_play_env(env_id):
    def _make():
        # Try 'human' first, fallback to 'rgb_array' if no screen found
        try:
            env = gym.make(env_id, render_mode="human")
        except:
            print("⚠️  Screen not found, running in background mode.")
            env = gym.make(env_id, render_mode="rgb_array")
            
        env = Monitor(env)
        env = AtariWrapper(env)
        return env
    
    venv = DummyVecEnv([_make])
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=4)
    return venv

def main():
    print("=" * 60)
    print("🔍 ANALYZING RESULTS FOR CHAMPION MODEL")
    print("=" * 60)

    # 1. Find Best ID
    try:
        best_id = get_best_experiment_id(CSV_PATH)
        model_path = MODEL_DIR_PATTERN.format(9)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 2. Verify Model Exists
    if not os.path.exists(model_path):
        print(f"❌ ERROR: The CSV says Exp {best_id} is best, but the file is missing!")
        print(f"   Expected location: {model_path}")
        return

    # 3. Load Model
    print(f"📂 Loading: {model_path}...")
    model = DQN.load(model_path)
    print("✅ Model loaded!")

    # 4. Play
    register_ale()
    env = make_play_env(ENV_ID)
    
    print("\n🎮 STARTING PLAYBACK")
    print("(Press Ctrl+C to stop early)\n")

    try:
        for episode in range(NUM_EPISODES):
            obs = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                total_reward += float(reward[0])
                steps += 1
                
                if done[0]:
                    done = True
            
            print(f"Episode {episode+1}: Score = {total_reward:.2f} ({steps} steps)")

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        env.close()
        print("\n✨ Done.")

if __name__ == "__main__":
    main()