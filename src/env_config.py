"""
MetaDrive Environment Configuration

Defines default configs for training and evaluation scenarios.
"""

# Default Training Configuration
TRAIN_CONFIG = {
    "use_render": False,
    "num_scenarios": 50,
    "start_seed": 0,
    "traffic_density": 0.1,
    "random_lane_width": True,
    "random_agent_model": True,
    "num_agents": 1,
    "horizon": 1000,
    # Freeze sensors for consistency
    "sensors": {
        "lidar": ["lidar"],
        "sidedetector": ["sidedetector"],
        "lanelinedetector": ["lanelinedetector"]
    }
}

# Evaluation Configuration (Unseen Seeds)
EVAL_CONFIG = {
    **TRAIN_CONFIG,
    "start_seed": 1000,  # Unseen scenarios
    "num_scenarios": 50
}

# Stress Test Configuration (High Difficulty)
STRESS_CONFIG = {
    **TRAIN_CONFIG,
    "start_seed": 2000,
    "traffic_density": 0.5,  # Heavy traffic
    "num_scenarios": 20
}

def get_env_config(config_type: str = "train") -> dict:
    """
    Return environment config dictionary.
    
    Args:
        config_type: "train", "eval", or "stress"
    
    Returns:
        Configuration dictionary for MetaDriveEnv
    """
    if config_type == "train":
        return TRAIN_CONFIG
    elif config_type == "eval":
        return EVAL_CONFIG
    elif config_type == "stress":
        return STRESS_CONFIG
    else:
        raise ValueError(f"Unknown config type: {config_type}")
