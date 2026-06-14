"""
Lab 6 - Reinforcement Learning: Q-Learning on MountainCar-v0
Variant 2: MountainCar

Requirements:
    pip install gymnasium numpy matplotlib

Usage:
    python mountain_car_qlearning.py

    To run a specific experiment, modify the EXPERIMENT variable at the bottom.
    Results (plots) are saved as PNG files in the current directory.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

# QUANTIZATION
# MountainCar observation space:
#   obs[0] = position   in [-1.2,  0.6]
#   obs[1] = velocity   in [-0.07, 0.07]
# We discretize each dimension into N_BINS buckets.

N_BINS = 40  # per dimension → Q-table shape: (40, 40, 3)

POSITION_BOUNDS = (-1.2, 0.6)
VELOCITY_BOUNDS = (-0.07, 0.07)


def discretize(observation):
    """Convert a continuous observation to a (pos_idx, vel_idx) tuple."""
    pos, vel = observation

    pos_idx = int(np.digitize(pos, np.linspace(*POSITION_BOUNDS, N_BINS - 1)))
    vel_idx = int(np.digitize(vel, np.linspace(*VELOCITY_BOUNDS, N_BINS - 1)))

    # Clamp to valid range
    pos_idx = np.clip(pos_idx, 0, N_BINS - 1)
    vel_idx = np.clip(vel_idx, 0, N_BINS - 1)

    return pos_idx, vel_idx


# Q-TABLE UPDATE

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    """
    Standard Q-Learning (off-policy) update:
        Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') − Q(s,a)]
    """
    current_q = q_table[state][action]
    max_future_q = np.max(q_table[next_state])
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
    q_table[state][action] = new_q
    return q_table


# TRAINING

def train(
    n_episodes=10_000,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.9995,
    render=False,
    seed=42,
):
    """
    Train a Q-Learning agent on MountainCar-v0.

    Parameters
    ----------
    n_episodes    : total training episodes
    alpha         : learning rate
    gamma         : discount factor
    epsilon_start : initial exploration rate
    epsilon_end   : minimum exploration rate
    epsilon_decay : multiplicative decay applied after each episode
    render        : set True to watch the car (slow!)
    seed          : random seed for reproducibility

    Returns
    -------
    q_table       : trained Q-table  (N_BINS, N_BINS, 3)
    rewards       : list of total reward per episode
    epsilons      : list of epsilon value per episode
    """
    rng = np.random.default_rng(seed)

    render_mode = "human" if render else None
    env = gym.make("MountainCar-v0", render_mode=render_mode)

    n_actions = env.action_space.n  # 3: push left, no push, push right

    # Initialize Q-table to zeros
    q_table = np.zeros((N_BINS, N_BINS, n_actions))

    epsilon = epsilon_start
    rewards = []
    epsilons = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        state = discretize(obs)
        total_reward = 0
        done = False

        while not done:
            # ε-greedy action selection
            if rng.random() < epsilon:
                action = env.action_space.sample()          # explore
            else:
                action = int(np.argmax(q_table[state]))     # exploit

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretize(next_obs)

            # Q-table update
            q_table = update_q_table(
                q_table, state, action, reward, next_state, alpha, gamma
            )

            state = next_state
            total_reward += reward

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        rewards.append(total_reward)
        epsilons.append(epsilon)

        if (episode + 1) % 1000 == 0:
            avg = np.mean(rewards[-1000:])
            print(
                f"  Episode {episode + 1:>6} | "
                f"Avg reward (last 1k): {avg:>8.1f} | "
                f"ε = {epsilon:.4f}"
            )

    env.close()
    return q_table, rewards, epsilons


# EVALUATION (greedy policy, no exploration)

def evaluate(q_table, n_episodes=100, seed=0):
    """Run greedy policy and return list of total rewards."""
    env = gym.make("MountainCar-v0")
    rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        state = discretize(obs)
        total_reward = 0
        done = False

        while not done:
            action = int(np.argmax(q_table[state]))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = discretize(obs)
            total_reward += reward

        rewards.append(total_reward)

    env.close()
    return rewards


# PLOTTING HELPERS

def smooth(data, window=200):
    """Simple moving average for plotting."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_training(rewards, epsilons, title, filename):
    """Save a 2-panel training curve plot."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle(title, fontsize=13)

    # Reward curve
    ax = axes[0]
    ax.plot(rewards, alpha=0.2, color="steelblue", linewidth=0.5, label="Episode reward")
    if len(rewards) >= 200:
        ax.plot(
            range(199, len(rewards)),
            smooth(rewards),
            color="steelblue",
            linewidth=1.5,
            label="Moving avg (200)",
        )
    ax.axhline(-200, color="red", linestyle="--", linewidth=0.8, label="Timeout (−200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Epsilon curve
    ax = axes[1]
    ax.plot(epsilons, color="orange", linewidth=1.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Exploration rate ε")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Plot saved → {filename}")


def plot_comparison(results, param_name, filename):
    """
    Compare multiple runs on one plot.
    results: list of (label, rewards)
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Hyperparameter comparison: {param_name}", fontsize=12)

    for label, rewards in results:
        ax.plot(rewards, alpha=0.15, linewidth=0.4)
        if len(rewards) >= 200:
            ax.plot(
                range(199, len(rewards)),
                smooth(rewards),
                linewidth=1.8,
                label=label,
            )
        else:
            ax.plot(rewards, linewidth=1.8, label=label)

    ax.axhline(-200, color="red", linestyle="--", linewidth=0.8, label="Timeout (−200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward (moving avg)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Plot saved → {filename}")


# EXPERIMENTS

def experiment_baseline():
    """Experiment 1 — baseline with well-tuned hyperparameters."""
    print("\n=== Experiment 1: Baseline ===")
    params = dict(
        n_episodes=15_000,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
    )
    print("  Params:", params)
    q_table, rewards, epsilons = train(**params)
    plot_training(rewards, epsilons, "Experiment 1 — Baseline", "exp1_baseline.png")

    eval_rewards = evaluate(q_table)
    print(f"  Eval (greedy, 100 eps): mean={np.mean(eval_rewards):.1f}  "
          f"success={sum(r > -200 for r in eval_rewards)}/100")
    return q_table


def experiment_learning_rate():
    """Experiment 2 — vary learning rate α."""
    print("\n=== Experiment 2: Learning Rate ===")
    alphas = [0.01, 0.1, 0.5]
    results = []

    for alpha in alphas:
        print(f"  α = {alpha}")
        _, rewards, _ = train(n_episodes=10_000, alpha=alpha, gamma=0.99,
                              epsilon_start=1.0, epsilon_end=0.01,
                              epsilon_decay=0.9995)
        results.append((f"α={alpha}", rewards))

    plot_comparison(results, "Learning Rate (α)", "exp2_learning_rate.png")


def experiment_discount_factor():
    """Experiment 3 — vary discount factor γ."""
    print("\n=== Experiment 3: Discount Factor ===")
    gammas = [0.90, 0.95, 0.99]
    results = []

    for gamma in gammas:
        print(f"  γ = {gamma}")
        _, rewards, _ = train(n_episodes=10_000, alpha=0.1, gamma=gamma,
                              epsilon_start=1.0, epsilon_end=0.01,
                              epsilon_decay=0.9995)
        results.append((f"γ={gamma}", rewards))

    plot_comparison(results, "Discount Factor (γ)", "exp3_discount_factor.png")


def experiment_epsilon_decay():
    """Experiment 4 — vary exploration decay rate."""
    print("\n=== Experiment 4: Epsilon Decay ===")
    decays = [0.999, 0.9995, 0.9999]
    results = []

    for decay in decays:
        print(f"  decay = {decay}")
        _, rewards, _ = train(n_episodes=10_000, alpha=0.1, gamma=0.99,
                              epsilon_start=1.0, epsilon_end=0.01,
                              epsilon_decay=decay)
        results.append((f"decay={decay}", rewards))

    plot_comparison(results, "Epsilon Decay", "exp4_epsilon_decay.png")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run all experiments sequentially.
    # Comment out any you don't want to run.

    experiment_baseline()
    experiment_learning_rate()
    experiment_discount_factor()
    experiment_epsilon_decay()

    print("\nAll experiments done. PNG files saved in:", os.getcwd())