import os
import gymnasium as gym
from stable_baselines3 import DQN

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_mlppolicy_model.zip")  # Adjust filename if needed

def main():
    # Load the trained model
    model = DQN.load(MODEL_PATH)
    
    # Create the environment with render_mode="human" for proper visualization
    env = gym.make("CartPole-v1", render_mode="human")
    
    obs, _ = env.reset()
    
    done = False
    while True:
        # Predict action using the trained model (deterministic = True for best policy)
        action, _states = model.predict(obs, deterministic=True)
        
        # Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment to see the agent playing
        env.render()
        
        # Check if episode is done
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    main()
