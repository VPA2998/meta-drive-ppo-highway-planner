"""
Evaluation Module

Tests trained policies on unseen scenarios and stress tests.
"""

import numpy as np
from metadrive import MetaDriveEnv
from stable_baselines3 import PPO

# Fix: Import close_engine safely
try:
    from metadrive.engine.engineutils import close_engine
except ImportError:
    try:
        from metadrive.utils import close_engine
    except ImportError:
        def close_engine():
            print("⚠️ close_engine not found, skipping...")
            pass

from env_config import get_env_config

def evaluate_policy(model: PPO, config_type: str = "eval", episodes: int = 5) -> tuple:
    """
    Evaluate trained policy on unseen scenarios.
    
    Args:
        model: Trained PPO model
        config_type: "eval" (unseen seeds) or "stress" (heavy traffic)
        episodes: Number of evaluation episodes
    
    Returns:
        Tuple: (mean_reward, std_reward)
    """
    close_engine()
    
    # Get config
    env_config = get_env_config(config_type)
    
    # Create test environment
    test_env = MetaDriveEnv(env_config)
    
    rewards = []
    
    print(f"Evaluating on {episodes} episodes ({config_type} scenarios)...")
    
    for ep in range(episodes):
        obs, _ = test_env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        
        while not done:
            # Predict action (deterministic for evaluation)
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        rewards.append(episode_reward)
        print(f"  Episode {ep+1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    test_env.close()
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print(f"\nEvaluation Results ({config_type}):")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward, std_reward

def run_benchmark(model: PPO, random_episodes: int = 5) -> dict:
    """
    Run benchmark comparison: PPO vs Random Agent.
    
    Args:
        model: Trained PPO model
        random_episodes: Number of episodes for random baseline
    
    Returns:
        Dictionary with benchmark results
    """
    close_engine()
    env_config = get_env_config("eval")
    test_env = MetaDriveEnv(env_config)
    
    # Evaluate PPO
    ppo_rewards = []
    for _ in range(random_episodes):
        obs, _ = test_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            ep_reward += reward
        ppo_rewards.append(ep_reward)
    
    # Evaluate Random Agent
    random_rewards = []
    import gymnasium as gym
    for _ in range(random_episodes):
        obs, _ = test_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            # Random action
            action = test_env.action_space.sample()
            obs, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            ep_reward += reward
        random_rewards.append(ep_reward)
    
    test_env.close()
    
    return {
        "ppo_mean": np.mean(ppo_rewards),
        "ppo_std": np.std(ppo_rewards),
        "random_mean": np.mean(random_rewards),
        "random_std": np.std(random_rewards)
    }
