import gymnasium as gym
import ale_py
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper

ENV_ID = "ALE/Breakout-v5"
MODEL_PATH = "cyiza_results/exp_1/final_model.zip"   # Change to your best experiment
NUM_EPISODES = 10

def register_ale():
    gym.register_envs(ale_py)

def make_play_env(env_id):
    def _make():
        env = gym.make(env_id, render_mode="human")
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

def main():
    print("=" * 70)
    print("DQN AGENT GAMEPLAY — BREAKOUT")
    print("=" * 70)
    print(f"Loading model: {MODEL_PATH}")

    try:
        model = DQN.load(MODEL_PATH)
        print(" Model loaded!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    register_ale()
    env = make_play_env(ENV_ID)

    episode_rewards = []
    episode_lengths = []

    print("\nStarting Breakout gameplay...\n")

    for episode in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_len = 0

        print(f"Playing Episode {episode + 1}/{NUM_EPISODES}...", end=" ")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            r = float(reward[0])
            d = bool(done[0])

            ep_reward += r
            ep_len += 1

            if d:
                done = True

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

        print(f"Reward: {ep_reward:.2f} | Steps: {ep_len}")

    env.close()

    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY — BREAKOUT")
    print("=" * 70)
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Best Episode:   {np.max(episode_rewards):.2f}")
    print(f"Worst Episode:  {np.min(episode_rewards):.2f}")
    print(f"Average Steps:  {np.mean(episode_lengths):.0f}")
    print("=" * 70)

    print("\nINTERPRETATION:")
    if np.mean(episode_rewards) > 10:
        print("agent is successfully hitting bricks regularly.")
    elif np.mean(episode_rewards) > 0:
        print("agent is learning basic ball control. More training needed.")
    else:
        print("agent struggles. Breakout typically needs 200k–500k timesteps.")
    print("=" * 70)


if __name__ == "__main__":
    main()
