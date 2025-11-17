# Reinforcement Learning Experiment Results

## MEMBER NAME  
Nelly Iyabikoze

## Hyperparameter Set and Noted Behavior

| Hyperparameter Set                                                                                         | Noted Behavior                                                                                     |
|-----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| lr=0.001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1                      | mean_reward=195.6 (moderate, good), std_reward=8.39 (low variability, stable performance)        |
| lr=0.0005, gamma=0.98, batch=64, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.2                      | mean_reward=102.2 (low, less optimal), std_reward=2.79 (very low variability, stable)            |
| lr=0.0001, gamma=0.95, batch=128, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=0.05                   | mean_reward=165.6 (moderate, good), std_reward=7.49 (low variability, reliable)                   |
| lr=0.0005, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1                     | mean_reward=149.4 (moderate, good), std_reward=30.16 (higher variability, less stable)            |
| lr=0.001, gamma=0.97, batch=64, epsilon_start=0.95, epsilon_end=0.05, epsilon_decay=0.15                    | mean_reward=107.3 (low, less optimal), std_reward=2.57 (very low variability, stable)             |
| lr=0.0002, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.05                    | mean_reward=287.4 (high, good), std_reward=36.64 (higher variability, promising but unstable)     |
| lr=0.0001, gamma=0.96, batch=32, epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=0.2                     | mean_reward=302.9 (high, good), std_reward=92.23 (very high variability, unstable training)      |
| lr=0.005, gamma=0.97, batch=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1                      | mean_reward=500.0 (excellent), std_reward=0.00 (no variability, perfect and stable)               |
| lr=0.001, gamma=0.95, batch=128, epsilon_start=0.95, epsilon_end=0.1, epsilon_decay=0.2                     | mean_reward=329.7 (high, good), std_reward=170.48 (extremely high variability, unstable)         |
| lr=0.002, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1                      | mean_reward=397.6 (very high, very good), std_reward=133.27 (high variability, caution needed)    |

### Explanation:

- **Mean reward** indicates how well the model performs on average. Higher mean rewards are better.
- **Standard deviation (std_reward)** measures variability in performance. Lower std means consistent and stable training.
- A small std_reward with a high mean_reward indicates good and stable model performance.
- A high std_reward indicates unstable or inconsistent performance, which may be less desirable even if the mean reward is high.

The perfect score (Experiment 8) with zero std_reward indicates the best and most stable performance. Other experiments with moderate to high mean rewards but high std_reward may need further tuning to stabilize training. Low mean rewards generally suggest less effective training outcomes.
