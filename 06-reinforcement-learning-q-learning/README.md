# Reinforcement Learning - Q-Learning

An implementation of the Q-Learning algorithm trained on the **Gymnasium MountainCar-v0** environment. The project discretizes the environment's continuous observation space and runs a series of parameter sensitivity experiments.

## Core Features

- **State Space Discretization**:
  - Maps continuous observations: Position in $[-1.2, 0.6]$ and Velocity in $[-0.07, 0.07]$ into a $40 \times 40$ grid.
  - The resulting discrete Q-table has a shape of $(40, 40, 3)$ representing the state space and the 3 action choices (push left, no push, push right).
- **Off-Policy Q-Learning**:
  - Implements the temporal difference update formula:
    $$Q(s,a) \leftarrow Q(s,a) + \alpha \left( r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right)$$
- **Exploration Policy**:
  - Epsilon-greedy exploration with exponential decay. At each step, exploration rate decays by a multiplier (default: $0.9995$), capped at a minimum rate (default: $0.01$).
- **Hyperparameter Sensitivity Experiments**:
  - **Experiment 1 (Baseline)**: Runs 15,000 episodes with tuned params, evaluating the greedy policy on 100 test runs.
  - **Experiment 2 (Learning Rate)**: Compares $\alpha \in \{0.01, 0.1, 0.5\}$.
  - **Experiment 3 (Discount Factor)**: Compares $\gamma \in \{0.90, 0.95, 0.99\}$.
  - **Experiment 4 (Epsilon Decay)**: Compares decay rates $\in \{0.999, 0.9995, 0.9999\}$.
- **Learning Curve Visualization**: Saves two-panel training plots (episode reward moving average + epsilon decay curve) and hyperparameter comparison plots.

## Directory Structure

- [main.py](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/06-reinforcement-learning-q-learning/main.py): Primary codebase with Q-learning agent, training, evaluation, and experiments.
- [exp1_baseline.png](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/06-reinforcement-learning-q-learning/exp1_baseline.png): Training curve of baseline parameters.
- [exp2_learning_rate.png](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/06-reinforcement-learning-q-learning/exp2_learning_rate.png): Learning curve comparison across learning rates.
- [exp3_discount_factor.png](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/06-reinforcement-learning-q-learning/exp3_discount_factor.png): Learning curve comparison across discount factors.
- [exp4_epsilon_decay.png](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/06-reinforcement-learning-q-learning/exp4_epsilon_decay.png): Learning curve comparison across epsilon decay speeds.
- [lab6_instruction_2026L.pdf](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/06-reinforcement-learning-q-learning/lab6_instruction_2026L.pdf): Project constraints and specifications.
- [lab6_report.pdf](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/06-reinforcement-learning-q-learning/lab6_report.pdf): Comprehensive lab report analyzing the hyperparameter comparison results.

## Setup and Usage

### Requirements
Ensure you have the required dependencies:
```bash
pip install gymnasium numpy matplotlib
```

### How to Run
Run the baseline model and hyperparameter study:
```bash
python main.py
```
This script runs the baseline and comparative experiments. Each experiment saves its visual plot output to the current directory as a PNG file.

## Authors
- **Savy Timothée**
- **Klepacz Wiktor**
- **Group Number**: 27
