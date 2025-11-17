import gymnasium as gym
import ale_py

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper

ENV_ID = "ALE/IceHockey-v5"
MODEL_PATH = "cyiza_results/exp_1/final_model.zip"

def register_ale():
    gym.register_envs(ale_py)

def make_play_env(env_id):
    env = gym.make(env_id, render_mode="human")

    env = Monitor(env)
    try:
        env = AtariWrapper(env)
    except:
        pass

    venv = DummyVecEnv([lambda: env])
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=4)

    return venv

def main():
    print(f"Loading best model: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH)

    register_ale()

    env = make_play_env(ENV_ID)
    obs, _ = env.reset()

    total_reward = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print("Episode ended. Reward:", total_reward)
            obs, _ = env.reset()
            total_reward = 0

if __name__ == "__main__":
    main
