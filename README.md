```markdown
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
```

project_root/
│── train.py        # Script to train the DQN agent
│── play.py         # Script to run the trained DQN agent
│── dqn_model.zip   # Saved DQN policy
│── README.md       # This file
│── results/        # Experiment logs and performance tables
│── models/         # Saved models from training

````

---

## Training Instructions (`train.py`)
1. Install dependencies:
```bash
pip install gymnasium stable-baselines3 ale-py AutoROM numpy pandas
````

2. Accept the ROM license for Atari environments:

```bash
AutoROM --accept-license
```

3. Run training:

```bash
python train.py
```

* This trains multiple experiments with different hyperparameters.
* Models are saved in the `models/` directory.
* Training results, including mean rewards and standard deviations, are logged in the `results/` directory.

---

## Playing Instructions (`play.py`)

1. Load the best trained model:

```python
from stable_baselines3 import DQN
model = DQN.load("dqn_model.zip")
```

2. Run the agent in the Breakout-v5 environment:

```bash
python play.py
```

* The game will be rendered in a window.
* The agent will act using a greedy policy, selecting the action with the highest Q-value.

---

## Hyperparameter Tuning Results

### Nelly Iyabikoze - Experiments

| Hyperparameter Set                                                                       | Noted Behavior                      |
| ---------------------------------------------------------------------------------------- | ----------------------------------- |
| lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1  | Good (Learned basics)               |
| lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1  | MLP Limitation                      |
| lr=0.00025, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | Good (Learned basics)               |
| lr=5e-05, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1   | Good (Learned basics)               |
| lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1  | Great Performance!                  |
| lr=0.0001, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | Great Performance!                  |
| lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1  | Good (Learned basics)               |
| lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.2  | Average                             |
| lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1  | Good (Learned basics)               |
| lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1  | Great Performance! (Champion Model) |

### Loue Sauveur Christian - Experiments





### Tamanda Lynn Thumba Kaunda - Experiments






### Henriette Cyiza - Experiments
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

* [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
* [Gymnasium Atari Environments](https://www.gymlibrary.dev/environments/atari/)
* [AutoROM](https://github.com/mgbellemare/AutoROM)

```



