"""
PPO Training Module

Handles model initialization and training loop.
"""

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from metadrive import MetaDriveEnv

# Fix: Import close_engine safely
try:
    from metadrive.engine.engineutils import close_engine
except ImportError:
    # Fallback for different MetaDrive versions
    try:
        from metadrive.utils import close_engine
    except ImportError:
        # Last resort: define a dummy function
        def close_engine():
            print("⚠️ close_engine not found, skipping...")
            pass

from env_config import get_env_config

def create_ppo_model(env_config: dict, verbose: int = 1) -> tuple:
    """
    Initialize PPO model with default hyperparameters.
    
    Args:
        env_config: MetaDrive environment config
        verbose: Logging verbosity
    
    Returns:
        Tuple: (model, env)
    """
    # Close any existing engines
    close_engine()
    
    # Create environment
    env = MetaDriveEnv(env_config)
    env = Monitor(env)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Policy network architecture
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU
    )
    
    # Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=verbose
    )
    
    return model, env

def train_model(model: PPO, total_timesteps: int = 100000, save_path: str = None):
    """
    Train the PPO model.
    
    Args:
        model: PPO model to train
        total_timesteps: Number of timesteps to train
        save_path: Path to save the model (optional)
    
    Returns:
        Trained model
    """
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    if save_path:
        model.save(save_path)
        print(f"Model saved to {save_path}")
    
    return model
