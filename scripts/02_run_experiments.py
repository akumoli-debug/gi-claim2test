from __future__ import annotations

import argparse
import os
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.wrappers import FlattenObservation
from minigrid import register_minigrid_envs
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

# Register MiniGrid environments
register_minigrid_envs()
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("Warning: sb3_contrib not available. RecurrentPPO will not work.")
    RecurrentPPO = None

import torch


class ProgressCallback(BaseCallback):
    """Callback to display progress during training."""

    def __init__(self, total_timesteps: int, desc: str = "Training"):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.desc = desc
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc=self.desc, unit="step")

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(self.num_timesteps - (self.pbar.n if self.pbar.n else 0))
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()


def train_baseline_agent(env_name: str, total_timesteps: int, seed: int) -> PPO:
    """Train a baseline PPO agent with MlpPolicy."""
    print(f"Training baseline agent on {env_name}...")

    def make_env(env_name, seed):
        def _init():
            env = gym.make(env_name)
            env = ImgObsWrapper(env)  # Convert to image observation
            env = FlattenObservation(env)  # Flatten to 1D vector
            env.reset(seed=seed)
            return env
        return _init

    vec_env = DummyVecEnv([make_env(env_name, seed)])
    
    # Print observation space to verify it's correct
    test_env = make_env(env_name, seed)()
    print(f"Observation space: {test_env.observation_space}")
    print(f"Action space: {test_env.action_space}")
    test_env.close()

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        seed=seed,
        verbose=1,
    )

    callback = ProgressCallback(total_timesteps, desc=f"Baseline {env_name}")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    return model


def train_memory_agent(env_name: str, total_timesteps: int, seed: int) -> Any:
    """Train a RecurrentPPO agent with MlpLstmPolicy."""
    if RecurrentPPO is None:
        raise ImportError("sb3_contrib.RecurrentPPO is not available")

    print(f"Training memory agent on {env_name}...")

    def make_env(env_name, seed):
        def _init():
            env = gym.make(env_name)
            env = ImgObsWrapper(env)  # Convert to image observation
            env = FlattenObservation(env)  # Flatten to 1D vector
            env.reset(seed=seed)
            return env
        return _init

    vec_env = DummyVecEnv([make_env(env_name, seed)])
    
    # Print observation space to verify it's correct
    test_env = make_env(env_name, seed)()
    print(f"Observation space: {test_env.observation_space}")
    print(f"Action space: {test_env.action_space}")
    test_env.close()

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        seed=seed,
        verbose=1,
    )

    callback = ProgressCallback(total_timesteps, desc=f"Memory {env_name}")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    return model


def evaluate_agent(
    model: Any, env_name: str, n_episodes: int, seed: int
) -> dict[str, Any]:
    """Evaluate an agent and return performance metrics."""
    # Use the SAME wrappers as training
    env = gym.make(env_name)
    env = ImgObsWrapper(env)  # Convert to image observation
    env = FlattenObservation(env)  # Flatten to 1D vector
    env.reset(seed=seed)

    rewards = []
    episode_lengths = []
    successes = []
    
    # Check if env has success attribute
    has_success_attr = hasattr(env.unwrapped, 'success') or hasattr(env, 'success')

    for episode_idx in tqdm(range(n_episodes), desc=f"Evaluating on {env_name}"):
        obs, info = env.reset(seed=seed + episode_idx)
        done = False
        episode_reward = 0.0
        episode_length = 0

        if isinstance(model, RecurrentPPO):
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)

        while not done:
            if isinstance(model, RecurrentPPO):
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=episode_starts, deterministic=True
                )
                episode_starts = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        # Print episode reward as it happens
        print(f"  Episode {episode_idx + 1}: reward={episode_reward:.2f}, length={episode_length}")
        
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Determine success: check info dict, env attribute, or reward > 0
        success = False
        if "success" in info:
            success = info["success"]
        elif has_success_attr:
            success = getattr(env.unwrapped, 'success', False) or getattr(env, 'success', False)
        else:
            # Fallback: use reward > 0 as success indicator
            success = episode_reward > 0
        
        successes.append(success)

    env.close()

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "success_rate": np.mean(successes) if successes else 0.0,
        "episode_lengths": episode_lengths,
    }


def run_test_a(timesteps: int = 200000, seed: int = 0) -> None:
    """Test A: Memory test using MiniGrid-DoorKey-5x5-v0."""
    print("\n" + "=" * 60)
    print("TEST A: Memory Test")
    print("=" * 60)

    env_name = "MiniGrid-DoorKey-5x5-v0"

    # Train baseline agent
    print("\nStarting baseline training...")
    baseline_model = train_baseline_agent(env_name, timesteps, seed)
    print("Baseline training complete. Evaluating...")
    
    # Train memory agent
    print("\nStarting memory training...")
    memory_model = train_memory_agent(env_name, timesteps, seed)
    print("Memory training complete. Evaluating...")

    # Evaluate on in-distribution (same seed)
    print("\nEvaluating on in-distribution (seed={})...".format(seed))
    baseline_id = evaluate_agent(baseline_model, env_name, n_episodes=20, seed=seed)
    memory_id = evaluate_agent(memory_model, env_name, n_episodes=20, seed=seed)

    # Evaluate on OOD (different seed)
    ood_seed = seed + 1000
    print(f"\nEvaluating on OOD (seed={ood_seed})...")
    baseline_ood = evaluate_agent(baseline_model, env_name, n_episodes=20, seed=ood_seed)
    memory_ood = evaluate_agent(memory_model, env_name, n_episodes=20, seed=ood_seed)
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)
    print(f"Baseline - In-Dist: reward={baseline_id['mean_reward']:.2f}, success={baseline_id['success_rate']:.2%}")
    print(f"Baseline - OOD: reward={baseline_ood['mean_reward']:.2f}, success={baseline_ood['success_rate']:.2%}")
    print(f"Memory - In-Dist: reward={memory_id['mean_reward']:.2f}, success={memory_id['success_rate']:.2%}")
    print(f"Memory - OOD: reward={memory_ood['mean_reward']:.2f}, success={memory_ood['success_rate']:.2%}")
    print("=" * 60)

    # Save results
    results = [
        {
            "test_type": "A",
            "agent_type": "baseline",
            "split": "in-distribution",
            "mean_reward": baseline_id["mean_reward"],
            "success_rate": baseline_id["success_rate"],
        },
        {
            "test_type": "A",
            "agent_type": "baseline",
            "split": "OOD",
            "mean_reward": baseline_ood["mean_reward"],
            "success_rate": baseline_ood["success_rate"],
        },
        {
            "test_type": "A",
            "agent_type": "memory",
            "split": "in-distribution",
            "mean_reward": memory_id["mean_reward"],
            "success_rate": memory_id["success_rate"],
        },
        {
            "test_type": "A",
            "agent_type": "memory",
            "split": "OOD",
            "mean_reward": memory_ood["mean_reward"],
            "success_rate": memory_ood["success_rate"],
        },
    ]

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/runs.csv", index=False, mode="a" if os.path.exists("results/runs.csv") else "w", header=not os.path.exists("results/runs.csv"))

    # Create comparison plot
    os.makedirs("results/plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Mean reward comparison
    categories = ["In-Dist", "OOD"]
    baseline_rewards = [baseline_id["mean_reward"], baseline_ood["mean_reward"]]
    memory_rewards = [memory_id["mean_reward"], memory_ood["mean_reward"]]

    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(x - width / 2, baseline_rewards, width, label="Baseline", alpha=0.8)
    ax1.bar(x + width / 2, memory_rewards, width, label="Memory", alpha=0.8)
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Test A: Mean Reward Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Success rate comparison
    baseline_success = [baseline_id["success_rate"], baseline_ood["success_rate"]]
    memory_success = [memory_id["success_rate"], memory_ood["success_rate"]]

    ax2.bar(x - width / 2, baseline_success, width, label="Baseline", alpha=0.8)
    ax2.bar(x + width / 2, memory_success, width, label="Memory", alpha=0.8)
    ax2.set_ylabel("Success Rate")
    ax2.set_title("Test A: Success Rate Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/test_a_comparison.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot to results/plots/test_a_comparison.png")
    plt.close()


def run_test_b(timesteps: int, seed: int) -> None:
    """Test B: Generalization test using DoorKey environments."""
    print("\n" + "=" * 60)
    print("TEST B: Generalization Test")
    print("=" * 60)

    train_env = "MiniGrid-DoorKey-8x8-v0"
    test_env = "MiniGrid-DoorKey-16x16-v0"

    # Train agents on 8x8
    baseline_model = train_baseline_agent(train_env, timesteps, seed)
    memory_model = train_memory_agent(train_env, timesteps, seed)

    # Evaluate on in-distribution (8x8)
    print(f"\nEvaluating on in-distribution ({train_env})...")
    baseline_id = evaluate_agent(baseline_model, train_env, n_episodes=20, seed=seed)
    memory_id = evaluate_agent(memory_model, train_env, n_episodes=20, seed=seed)

    # Evaluate on OOD (16x16)
    print(f"\nEvaluating on OOD ({test_env})...")
    baseline_ood = evaluate_agent(baseline_model, test_env, n_episodes=20, seed=seed)
    memory_ood = evaluate_agent(memory_model, test_env, n_episodes=20, seed=seed)

    # Save results
    results = [
        {
            "test_type": "B",
            "agent_type": "baseline",
            "split": "in-distribution",
            "mean_reward": baseline_id["mean_reward"],
            "success_rate": baseline_id["success_rate"],
        },
        {
            "test_type": "B",
            "agent_type": "baseline",
            "split": "OOD",
            "mean_reward": baseline_ood["mean_reward"],
            "success_rate": baseline_ood["success_rate"],
        },
        {
            "test_type": "B",
            "agent_type": "memory",
            "split": "in-distribution",
            "mean_reward": memory_id["mean_reward"],
            "success_rate": memory_id["success_rate"],
        },
        {
            "test_type": "B",
            "agent_type": "memory",
            "split": "OOD",
            "mean_reward": memory_ood["mean_reward"],
            "success_rate": memory_ood["success_rate"],
        },
    ]

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/runs.csv", index=False, mode="a", header=False)

    # Create comparison plot
    os.makedirs("results/plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Mean reward comparison
    categories = ["In-Dist (8x8)", "OOD (16x16)"]
    baseline_rewards = [baseline_id["mean_reward"], baseline_ood["mean_reward"]]
    memory_rewards = [memory_id["mean_reward"], memory_ood["mean_reward"]]

    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(x - width / 2, baseline_rewards, width, label="Baseline", alpha=0.8)
    ax1.bar(x + width / 2, memory_rewards, width, label="Memory", alpha=0.8)
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Test B: Mean Reward Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Success rate comparison
    baseline_success = [baseline_id["success_rate"], baseline_ood["success_rate"]]
    memory_success = [memory_id["success_rate"], memory_ood["success_rate"]]

    ax2.bar(x - width / 2, baseline_success, width, label="Baseline", alpha=0.8)
    ax2.bar(x + width / 2, memory_success, width, label="Memory", alpha=0.8)
    ax2.set_ylabel("Success Rate")
    ax2.set_title("Test B: Success Rate Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/test_b_comparison.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot to results/plots/test_b_comparison.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run observational tests for RL claims")
    parser.add_argument(
        "--test",
        type=str,
        choices=["A", "B", "ALL"],
        default="ALL",
        help="Which test to run: A (memory), B (generalization), or ALL",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="Number of training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )

    args = parser.parse_args()

    if args.test == "A" or args.test == "ALL":
        try:
            run_test_a(args.timesteps, args.seed)
        except Exception as e:
            print(f"Error running Test A: {e}")
            import traceback

            traceback.print_exc()

    if args.test == "B" or args.test == "ALL":
        try:
            run_test_b(args.timesteps, args.seed)
        except Exception as e:
            print(f"Error running Test B: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
