"""
MetaDrive Environment Configuration - Minimal Version
"""

TRAIN_CONFIG = {
    "use_render": False,
    "num_scenarios": 50,
    "start_seed": 0,
    "traffic_density": 0.1,
    "random_lane_width": True,
    "random_agent_model": True,
    "num_agents": 1,
    "horizon": 1000,
    # Remove sensors entirely - use MetaDrive defaults
    "image_observation": False,
    "discrete_action": False,
}

EVAL_CONFIG = {
    **TRAIN_CONFIG,
    "start_seed": 1000,
    "num_scenarios": 50
}

STRESS_CONFIG = {
    **TRAIN_CONFIG,
    "start_seed": 2000,
    "traffic_density": 0.5,
    "num_scenarios": 20
}

def get_env_config(config_type: str = "train") -> dict:
    if config_type == "train":
        return TRAIN_CONFIG
    elif config_type == "eval":
        return EVAL_CONFIG
    elif config_type == "stress":
        return STRESS_CONFIG
    else:
        raise ValueError(f"Unknown config type: {config_type}")
