
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

| Experiment | Focus / Policy        | Timesteps | LR     | Batch | Buffer | Gamma | Epsilon Start | Epsilon End | Epsilon Decay | Mean Reward | Std Reward | Noted Behavior                        |
| ---------- | --------------------- | --------- | ------ | ----- | ------ | ----- | ------------- | ----------- | ------------- | ----------- | ---------- | ------------------------------------- |
| 1          | Micro Sprint 1 / CNN  | 10000     | 0.001  | 16    | 5000   | 0.90  | 1.0           | 0.1         | 0.5           | 8.4         | 2.33       | Average                               |
| 2          | Micro Sprint 2 / CNN  | 15000     | 0.0009 | 24    | 7500   | 0.91  | 0.95          | 0.09        | 0.45          | 8.2         | 0.98       | Average                               |
| 3          | Micro Sprint 3 / CNN  | 20000     | 0.0008 | 32    | 10000  | 0.92  | 0.9           | 0.08        | 0.4           | 14.8        | 4.12       | Good (Learned basics)                 |
| 4          | Micro Sprint 4 / CNN  | 25000     | 0.0007 | 40    | 12500  | 0.93  | 0.85          | 0.07        | 0.35          | 12.2        | 2.04       | Good (Learned basics)                 |
| 5          | Micro Sprint 5 / CNN  | 30000     | 0.0006 | 48    | 15000  | 0.94  | 0.8           | 0.06        | 0.3           | 17.0        | 4.65       | Great Performance!                    |
| 6          | Micro Sprint 6 / CNN  | 35000     | 0.0005 | 64    | 17500  | 0.95  | 1.0           | 0.05        | 0.25          | 18.4        | 3.32       | Great Performance!                    |
| 7          | Micro Sprint 7 / CNN  | 40000     | 0.0004 | 96    | 50000  | 0.96  | 1.0           | 0.04        | 0.2           | 15.4        | 3.32       | Great Performance!                    |
| 8          | Micro Sprint 8 / CNN  | 55000     | 0.0003 | 128   | 22500  | 0.97  | 1.0           | 0.03        | 0.15          | 21.0        | 5.73       | Great Performance!                    |
| 9          | Micro Sprint 9 / CNN  | 60000     | 0.0002 | 256   | 25000  | 0.98  | 1.0           | 0.02        | 0.1           | 19.8        | 1.94       | **Best balance (High mean, low std)** |
| 10         | Micro Sprint 10 / CNN | 70000     | 0.0001 | 512   | 30000  | 0.99  | 1.0           | 0.01        | 0.1           | 14.6        | 2.87       | Good (Learned basics)                 |

**Key Observations:**

* **Best model:** Experiment 9, combining **high mean reward (19.8)** with **lowest standard deviation (1.94)** among top performers.
* **Batch size & buffer:** Larger values increased stability and reduced variance.
* **Gamma:** High gamma (0.98) favored long-term rewards.
* **LR:** Moderate learning rates (0.0003–0.0005) gave best learning speed and performance.
* **CNNPolicy:** Crucial for pixel-based Atari environments; MLPPolicy showed poor performance.
* **Epsilon decay:** Balanced exploration and exploitation; too fast decay reduced learning.


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



