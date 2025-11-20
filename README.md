
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

| Exp ID | Hyperparameter Set                                                                 | Mean Reward | Std Reward | Noted Behavior         |
|--------|-------------------------------------------------------------------------------------|-------------|------------|--------------------------|
| 1      | lr=0.001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1  | 16.0        | 0.89       | Great Performance!      |
| 2      | lr=0.0005, gamma=0.98, batch=64, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.2  | 20.2        | 6.24       | Great Performance!      |
| 3      | lr=0.0001, gamma=0.95, batch=128, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=0.05 | 12.2     | 4.87       | Good (Learned basics).  |
| 4      | lr=0.0005, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1   | 17.2     | 4.31       | Great Performance!      |
| 5      | lr=0.001, gamma=0.97, batch=64, epsilon_start=0.95, epsilon_end=0.05, epsilon_decay=0.15  | 16.8       | 4.96       | Great Performance!      |
| 6      | lr=0.0002, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.05  | 15.6       | 2.87       | Great Performance!      |
| 7      | lr=0.0001, gamma=0.96, batch=32, epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=0.2   | 7.4        | 1.02       | Average.                |
| 8      | lr=0.005, gamma=0.97, batch=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1    | 15.6       | 4.08       | Great Performance!      |
| 9      | lr=0.001, gamma=0.95, batch=128, epsilon_start=0.95, epsilon_end=0.1, epsilon_decay=0.2   | 18.8       | 7.78       | Great Performance!      |
| 10     | lr=0.002, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1    | 5.2        | 3.87       | Average.                |

**Hyperparameter Tuning Results**

The DQN agent was trained with 10 different hyperparameter combinations, as shown in the table above. Key observations include:

**Learning Rate (lr):** Moderate values (0.0005–0.001) produced the highest rewards, while too low or too high learning rates slowed or destabilized learning.

**Discount Factor (gamma)**: Higher gamma (0.99) favored long-term reward optimization, but slightly lower gamma (0.95–0.98) performed well when paired with appropriate batch sizes and learning rates.

**Batch Size**: Mid-range batch sizes (64) provided consistent performance, while smaller or larger batches showed either higher variance or slower updates.

**Exploration (epsilon schedule)**: Balanced epsilon decay and end values (0.05–0.1) allowed effective exploration, with the highest performing model (Exp 2) using epsilon_end = 0.1.

**Best Model**: Experiment 2 (lr=0.0005, gamma=0.98, batch=64, epsilon_end=0.1) achieved the highest mean reward of 20.2, demonstrating stable learning and balanced exploration, and is considered the champion model.


### Loue Sauveur Christian - Experiments

| Exp ID | Focus / Policy              | Timesteps | LR     | Batch | Buffer | Mean Reward | Std Reward | Auto-Comment          |
| ------ | --------------------------- | --------- | ------ | ----- | ------ | ----------- | ---------- | --------------------- |
| 1      | Micro Sprint 1 / MlpPolicy  | 10000     | 0.001  | 16    | 5000   | 5.8         | 2.99       | MLP Limitation        |
| 2      | Micro Sprint 2 / MlpPolicy  | 15000     | 0.0009 | 24    | 7500   | 2.0         | 0.0        | MLP Limitation        |
| 3      | Micro Sprint 3 / MlpPolicy  | 20000     | 0.0008 | 32    | 10000  | 7.0         | 4.05       | MLP Limitation        |
| 4      | Micro Sprint 4 / CnnPolicy  | 25000     | 0.0007 | 40    | 12500  | 12.2        | 2.04       | Good (Learned basics) |
| 5      | Micro Sprint 5 / CnnPolicy  | 30000     | 0.0006 | 48    | 15000  | 17.0        | 4.65       | Great Performance!    |
| 6      | Micro Sprint 6 / CnnPolicy  | 35000     | 0.0005 | 64    | 17500  | 18.4        | 3.32       | Great Performance!    |
| 7      | Micro Sprint 7 / MlpPolicy  | 40000     | 0.0004 | 96    | 50000  | 4.4         | 1.74       | MLP Limitation        |
| 8      | Micro Sprint 8 / MlpPolicy  | 55000     | 0.0003 | 128   | 22500  | 4.8         | 1.94       | MLP Limitation        |
| 9      | Micro Sprint 9 / CnnPolicy  | 99000     | 0.0002 | 512   | 25000  | 22.2        | 7.83       | Great Performance!    |
| 10     | Micro Sprint 10 / MlpPolicy | 70000     | 0.0001 | 256   | 30000  | 11.4        | 2.65       | MLP Limitation        |

**Key Observations:**

* **Best model:** Experiment 5 & 6 are strong, but **Experiment 9** shows the **highest mean reward (22.2)**, despite higher std (7.83).
* **MLP vs CNN:** MLPPolicy struggles in pixel-based environments; CNNPolicy significantly improves reward.
* **Timesteps & Buffer:** Longer timesteps with proper batch and buffer (Exp 9) allow the model to explore and achieve higher rewards.
* **Learning Rate:** Moderate LR (0.0005–0.0007) with CNN gives stable, strong learning; too high or too low LR with MLP leads to poor performance.
* **Epsilon Decay:** Balances exploration and exploitation; extremely fast decay in MLP setups limits learning.
* **CNNPolicy** is superior in visual environments like Breakout because it can learn spatial and temporal features from pixel input.

* **MLPPolicy** fails to capture environment complexity, leading to low mean rewards even with long training.

* **High timesteps**, proper epsilon decay, and large buffers amplify CNN performance; MLP only benefits marginally.

### Tamanda Lynn Thumba Kaunda - Experiments 
Video Link: https://youtu.be/3r9agr-D5K8?si=OA5ojzfzn0HLMrIY

| Exp ID | Focus                         | Timesteps | LR      | Gamma | Batch | Buffer | Learn Starts | Target Update | Eps Decay | Mean Reward | Std Reward |
|-------|-------------------------------|-----------|---------|-------|-------|--------|--------------|----------------|-----------|-------------|------------|
| 1     | Baseline Short (quick-check)  | 5000      | 0.0001  | 0.99  | 64    | 5000   | 1000         | 1000           | 0.2       | 3.2         | 0.75       |
| 2     | Optimized Fast (Exp6-tuned)   | 5000      | 0.0003  | 0.95  | 32    | 10000  | 1000         | 1000           | 0.1       | 7.6         | 1.50       |
| 3     | Longer Stable (50k)           | 150000    | 0.0001  | 0.90  | 32    | 50000  | 5000         | 1000           | 0.2       | 21.6        | 7.28       |
| 4     | Aggressive Start (MLP test)   | 5000      | 0.0002  | 0.99  | 64    | 100000 | 500          | 1000           | 0.1       | 2.8         | 0.40       |
| 5     | Mid Stable (20k)              | 20000     | 0.00015 | 0.96  | 32    | 20000  | 2000         | 1000           | 0.15      | 10.0        | 2.90       |
| 6     | High Gamma Long Horizon       | 40000     | 0.0001  | 0.995 | 32    | 40000  | 2000         | 2000           | 0.2       | 7.6         | 1.50       |
| 7     | Short Buffer Fast Decay       | 10000     | 0.0002  | 0.97  | 64    | 5000   | 1000         | 1000           | 0.5       | 6.8         | 4.07       |
| 8     | Delayed Target Update (5k)    | 15000     | 0.0001  | 0.99  | 32    | 10000  | 1000         | 5000           | 0.1       | 4.2         | 1.94       |
| 9     | Balanced Optimizer            | 20000     | 0.00015 | 0.995 | 32    | 20000  | 1000         | 1000           | 0.15      | 8.0         | 4.05       |
| 10    | Fastest Decay (0.5)           | 5000      | 0.0001  | 0.99  | 64    | 50000  | 1000         | 1000           | 0.5       | 0.0         | 0.00       |

## key Results
The highest-performing configuration was the 150k-step training run (Exp 3), achieving a mean reward of 21.6. This reinforces the importance of extended training horizons, large replay buffers, and a carefully-paced exploration schedule for Atari environments. Short runs (5k–20k) consistently underperformed, confirming that Breakout requires substantial interaction data for stable learning.



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





