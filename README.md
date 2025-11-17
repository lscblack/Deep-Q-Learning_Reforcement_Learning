# Reinforcement Learning Experiment Results - Combined Report

## MEMBER 1: Nelly Iyabikoze

**Video Presentation:** [Nelly's Video](https://youtu.be/gDJUlbZNDMI)

### Hyperparameter Set and Noted Behavior

| Experiment | Hyperparameter Set | Noted Behavior |
|------------|------------------|----------------|
| 1 | lr=0.001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | mean_reward=195.6 (moderate, good), std_reward=8.39 (low variability, stable performance) |
| 2 | lr=0.0005, gamma=0.98, batch=64, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.2 | mean_reward=102.2 (low, less optimal), std_reward=2.79 (very low variability, stable) |
| 3 | lr=0.0001, gamma=0.95, batch=128, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=0.05 | mean_reward=165.6 (moderate, good), std_reward=7.49 (low variability, reliable) |
| 4 | lr=0.0005, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | mean_reward=149.4 (moderate, good), std_reward=30.16 (higher variability, less stable) |
| 5 | lr=0.001, gamma=0.97, batch=64, epsilon_start=0.95, epsilon_end=0.05, epsilon_decay=0.15 | mean_reward=107.3 (low, less optimal), std_reward=2.57 (very low variability, stable) |
| 6 | lr=0.0002, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.05 | mean_reward=287.4 (high, good), std_reward=36.64 (higher variability, promising but unstable) |
| 7 | lr=0.0001, gamma=0.96, batch=32, epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=0.2 | mean_reward=302.9 (high, good), std_reward=92.23 (very high variability, unstable training) |
| 8 | lr=0.005, gamma=0.97, batch=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | mean_reward=500.0 (excellent), std_reward=0.00 (no variability, perfect and stable) |
| 9 | lr=0.001, gamma=0.95, batch=128, epsilon_start=0.95, epsilon_end=0.1, epsilon_decay=0.2 | mean_reward=329.7 (high, good), std_reward=170.48 (extremely high variability, unstable) |
| 10 | lr=0.002, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | mean_reward=397.6 (very high, very good), std_reward=133.27 (high variability, caution needed) |

---

## MEMBER 2: Loue Sauveur Christian

**Video Presentation:** [Link to Loue's Video](https://drive.google.com/file/d/1SJ_y_ezgA8dIHGomwB_QOP7dX_Pg9M5j/view?usp=sharing)

### DQN Atari Experiments Summary

Total experiments: 10

### Experiment Results

| Model | Mean Reward | Std Reward | Model Path | Hyperparameter Set |
|-------|------------|-----------|------------|------------------|
| dqn_3 | 558.00 | 266.71 | results/dqn_exp3/dqn_3_final.zip | lr=0.0001, gamma=0.99, batch=16, eps_start=1.0, eps_end=0.01, eps_decay=100000.0 |
| dqn_1 | 248.00 | 93.04 | results/dqn_exp1/dqn_1_final.zip | lr=5e-05, gamma=0.98, batch=2, eps_start=1.0, eps_end=0.02, eps_decay=200000.0 |
| dqn_4 | 206.00 | 28.71 | results/dqn_exp4/dqn_4_final.zip | lr=0.0002, gamma=0.97, batch=16, eps_start=1.0, eps_end=0.01, eps_decay=150000.0 |
| dqn_7 | 140.00 | 0.00 | results/dqn_exp7/dqn_7_final.zip | lr=0.0001, gamma=0.96, batch=8, eps_start=0.9, eps_end=0.05, eps_decay=100000.0 |
| dqn_8 | 140.00 | 0.00 | results/dqn_exp8/dqn_8_final.zip | lr=0.0003, gamma=0.99, batch=4, eps_start=1.0, eps_end=0.01, eps_decay=200000.0 |
| dqn_9 | 100.00 | 0.00 | results/dqn_exp9/dqn_9_final.zip | lr=0.00015, gamma=0.995, batch=2, eps_start=1.0, eps_end=0.005, eps_decay=500000.0 |
| dqn_10 | 100.00 | 0.00 | results/dqn_exp10/dqn_10_final.zip | lr=1e-05, gamma=0.99, batch=32, eps_start=1.0, eps_end=0.01, eps_decay=300000.0 |
| dqn_2 | 0.00 | 0.00 | results/dqn_exp2/dqn_2_final.zip | lr=0.0002, gamma=0.96, batch=8, eps_start=1.0, eps_end=0.1, eps_decay=100000.0 |
| dqn_5 | 0.00 | 0.00 | results/dqn_exp5/dqn_5_final.zip | lr=5e-05, gamma=0.99, batch=4, eps_start=1.0, eps_end=0.02, eps_decay=1000000.0 |
| dqn_6 | 0.00 | 0.00 | results/dqn_exp6/dqn_6_final.zip | lr=0.0002, gamma=0.995, batch=6, eps_start=1.0, eps_end=0.01, eps_decay=100000.0 |

**Best model:** dqn_3 with mean reward 558.00

---


# Hyperparameter Experiments Log (10 Runs) – Member 3: Tamanda

| # | Policy      | Hyperparameters                                                                 | Mean Reward | Noted Behavior |
|---|-------------|----------------------------------------------------------------------------------|-------------|----------------|
| 1 | MLPPolicy   | lr=2.5e-4, gamma=0.995, batch=32, epsilon_decay=2e5                              | -14.20      | Policy failure from pixel input. Requires CNN. |
| 2 | CnnPolicy   | lr=3e-4, gamma=0.99, batch=32, epsilon_decay=7e5                                | 1.8         | Stable, high LR pushing limits; near-optimal performance. |
| 3 | CnnPolicy   | lr=1e-3, gamma=0.95, batch=128, epsilon_decay=2e5                               | -10.50      | LR too high → instability + overshooting. |
| 4 | CnnPolicy   | lr=5e-5, gamma=0.998, batch=32, epsilon_decay=1e6                               | 2.5         | Farsighted + slow; stable but slow convergence. |
| 5 | CnnPolicy   | lr=2e-4, gamma=0.0, batch=64, epsilon_decay=1e5                                 | -1.0        | Myopic agent; only immediate reward → weak policy. |
| 6 | CnnPolicy   | lr=2e-4, gamma=0.995, batch=16, epsilon_decay=2e5                               | 1.9         | Small batch → high noise + variance. |
| 7 | CnnPolicy   | lr=2e-4, gamma=0.995, batch=128, epsilon_decay=2e5                              | 3.2         | Large batch → smooth + stable convergence (near-best). |
| 8 | CnnPolicy   | lr=2.5e-4, gamma=0.995, batch=32, epsilon_decay=5e4                             | -4.0        | Exploration too fast → stuck in local optima. |
| 9 | CnnPolicy   | lr=2.5e-4, gamma=0.995, batch=32, epsilon_decay=3e6                             | 0.5         | Exploration too slow → random for too long. |
|10 | CnnPolicy   | lr=1e-4, gamma=0.99, batch=64, eps_start=0.5, eps_end=0.01, decay=2e5           | 2.0         | Low initial eps → less randomness, but risk of missing states. |

** Findings (Tamanda's)
My 10 experiments validated our architecture and optimized our parameters. The mandatory Policy Comparison (Exp 1) achieved only $\mathbf{-14.20}$ reward with $\text{MLPPolicy}$, confirming that $\text{CNNPolicy}$ was essential. Stress-testing extreme values revealed that a low Discount Factor ($\gamma=0.0$) caused non-strategic, myopic behavior, and a **high Learning Rate ($1\times 10^{-3}$) ** created severe instability. This process proved that the group's final choice—a high $\gamma$ and moderate $\text{lr}$ (e.g., $2\times 10^{-4}$)—is the necessary balance for stable, strategic success in the IceHockey environment.

## Combined Analysis

### Key Findings:

1. **Best Performers:**
   - Loue's dqn_3 achieved the highest mean reward (558.00)
   - Nelly's Experiment 8 achieved perfect stability (500.0 mean, 0.0 std)

2. **Hyperparameter Trends:**
   - Higher gamma values (0.99) generally correlated with better performance
   - Learning rates around 1e-4 to 2e-4 provided good balance
   - Medium batch sizes (8-32) performed better than extremes

3. **Stability vs Performance Trade-off:**
   - Some experiments achieved high rewards but with high variability
   - Others achieved perfect stability but with moderate rewards
   - The ideal configuration balances both high mean reward and low standard deviation

### Recommendations for Future Experiments:
- Focus on gamma values around 0.99
- Use learning rates between 0.0001-0.001
- Employ batch sizes in the 16-32 range
- Balance exploration-exploitation with appropriate epsilon decay schedules
