
---

## Project Overview  
In this assignment, we implemented a Deep Q-Network (DQN) agent using **Stable Baselines 3** and **Gymnasium** to train the agent to play an Atari game. We experimented with multiple hyperparameters and policy types (MLPPolicy and CNNPolicy) to observe their effects on learning performance. After training, we evaluated the agent using `play.py` and recorded its performance in real-time.

---

## Environment  
We used the following Atari environment from Gymnasium:

**Breakout-v5**  

Required Python packages:  
- gymnasium  
- stable-baselines3  
- ale-py  
- AutoROM  

---

## Project Structure  

project_root/  
    train.py        # Script to train the DQN agent  
    play.py         # Script to run the trained DQN agent  
    dqn_model.zip   # Saved DQN policy  
    README.md       # This file  
    results/        # Experiment logs and performance tables  
    models/         # Saved models from training  

---

## Training Instructions (train.py)  

1. Install dependencies:  
  
    pip install gymnasium stable-baselines3 ale-py AutoROM numpy pandas  

2. Accept the ROM license for Atari environments:  
  
    AutoROM --accept-license  

3. Run training:  
  
    python train.py  
  
* This trains multiple experiments with different hyperparameters.  
* Models are saved in the `models/` directory.  
* Training results, including mean rewards and standard deviations, are logged in the `results/` directory.  

---

## Playing Instructions (play.py)  

1. Load the best trained model:  
  
    from stable_baselines3 import DQN  
    model = DQN.load("dqn_model.zip")  

2. Run the agent in the Breakout-v5 environment:  
  
    python play.py  

* The game will be rendered in a window.  
* The agent will act using a greedy policy, selecting the action with the highest Q-value.  

---

## Hyperparameter Tuning Results  

### Nelly Iyabikoze - Experiments  

| Hyperparameter Set                                                                         | Noted Behavior                      |  
| ------------------------------------------------------------------------------------------| -----------------------------------|  
| lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1      | Good (Learned basics)               |  
| lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1      | MLP Limitation                     |  
| lr=0.00025, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1     | Good (Learned basics)               |  
| lr=5e-05, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1       | Good (Learned basics)               |  
| lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1      | Great Performance!                  |  
| lr=0.0001, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1     | Great Performance!                  |  
| lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1      | Good (Learned basics)               |  
| lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.2      | Average                           |  
| lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1      | Good (Learned basics)               |  
| lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1      | Great Performance! (Champion Model)|  




### Loue Sauveur Christian - Experiments  





### Tamanda Lynn Thumba Kaunda - Experiments  





### Henriette Cyiza - Experiments  

video link: https://www.loom.com/share/3c445b6aa90b42cfa046c002a4743699

   Table of Hyperparameters and Noted Behavior

| Experiment | Hyperparameters (Simplified)                                  | Noted Behavior                                                      |
| ---------- | ------------------------------------------------------------- | ------------------------------------------------------------------- |
| **1**      | lr=0.0008, γ=0.90, batch=64, eps_end=0.02, decay=50k, CNN     | **Very strong** (18.3 avg) — aggressive learner helped early reward |
| **2**      | lr=0.000005, γ=0.999, batch=32, eps_end=0.05, decay=800k, CNN | **Underperformed** — extremely slow learning                        |
| **3**      | lr=0.0003, γ=0.98, batch=64, eps_end=0.2, decay=100k, CNN     | **Very strong** (18.7 avg) — high exploration helped                |
| **4**      | lr=0.0001, γ=0.99, batch=32, eps_end=0.01, decay=50k, **MLP** | **Weak** — MLP cannot extract pixel features                        |
| **5**      | lr=0.0002, γ=0.995, batch=128, eps_end=0.02, decay=300k, CNN  | **Strong** — stable learner with large batch                        |
| **6**      | lr=0.00015, γ=0.97, batch=8, eps_end=0.05, decay=1M, CNN      | **Moderate** — noisy due to tiny batch                              |
| **7**      | lr=0.0005, γ=0.99, batch=64, eps_end=0.01, decay=2M, CNN      | **BEST Overall (21.3 avg)** — long exploration gave best mastery    |
| **8**      | lr=0.0001, γ=0.96, batch=64, eps_end=0.02, decay=80k, **MLP** | **Weak** — MLP limitation visible                                   |
| **9**      | lr=0.0003, γ=0.999, batch=64, eps_end=0.01, decay=400k, CNN   | **Strong** — long horizon γ=0.999 helped                            |
| **10**     | lr=0.0002, γ=0.92, batch=32, eps_end=0.05, decay=50k, **MLP** | **Weak** — low gamma + MLP = shallow learning                       |



---

## Observations  

* Increasing batch size improved stability in training.  
* Smaller buffers required fewer resources but slightly reduced performance.  
* MLPPolicy is limited for visual Atari environments; CNNPolicy significantly outperformed it.  
* Proper epsilon decay balances exploration and exploitation, improving rewards.  

---

## Video Demonstration  

* Run `play.py` to record a gameplay clip showing the agent’s performance.  
* Ensure the video shows the agent interacting with the environment in real-time.  
* Include this video in the final submission for evaluation.  

---

## References  

* Stable Baselines3 Documentation: https://stable-baselines3.readthedocs.io/  
* Gymnasium Atari Environments: https://www.gymlibrary.dev/environments/atari/  
* AutoROM: https://github.com/mgbellemare/AutoROM  

---



