import os
import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper


MODEL_PATH = "models/model_exp_3.zip"
NUM_EPISODES = 5

def make_env():
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = AtariWrapper(env)
    return env

def create_env():
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    return env

def play():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = DQN.load(MODEL_PATH, optimize_memory_usage=False)

    env = create_env()

    for episode in range(NUM_EPISODES):
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, _ = reset_result
        else:
            obs = reset_result

        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            total_reward += reward[0]
            env.render()
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()
    print("Play finished.")

if __name__ == "__main__":
    play()
